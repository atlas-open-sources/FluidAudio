#!/usr/bin/env python3
"""Test CoreML inference for Nemotron Streaming 0.6B on LibriSpeech test-clean."""
import glob
import json
import re
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf


def load_ground_truth(librispeech_path: str) -> dict:
    """Load all ground truth transcriptions."""
    gt = {}
    for trans_file in glob.glob(f"{librispeech_path}/**/*.trans.txt", recursive=True):
        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    file_id, text = parts
                    gt[file_id] = text.lower()
    return gt


def normalize_text(text: str) -> str:
    """Normalize text for WER calculation."""
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.lower().split())


def compute_wer(reference: str, hypothesis: str) -> tuple:
    """Compute WER between reference and hypothesis."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.uint32)
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i, j] = d[i-1, j-1]
            else:
                d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + 1)

    errors = d[len(ref_words), len(hyp_words)]
    return errors, len(ref_words)


class NemotronCoreMLInference:
    """CoreML inference for Nemotron Streaming."""

    def __init__(self, model_dir: str):
        model_dir = Path(model_dir)

        # Load metadata
        with open(model_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        # Load tokenizer
        with open(model_dir / "tokenizer.json") as f:
            self.tokenizer = json.load(f)

        print("Loading CoreML models...")
        self.preprocessor = ct.models.MLModel(str(model_dir / "preprocessor.mlpackage"))
        self.encoder = ct.models.MLModel(str(model_dir / "encoder.mlpackage"))
        self.decoder = ct.models.MLModel(str(model_dir / "decoder.mlpackage"))
        self.joint = ct.models.MLModel(str(model_dir / "joint.mlpackage"))
        print("Models loaded!")

        self.sample_rate = self.metadata["sample_rate"]
        self.chunk_mel_frames = self.metadata["chunk_mel_frames"]
        self.pre_encode_cache = self.metadata["pre_encode_cache"]
        self.total_mel_frames = self.metadata["total_mel_frames"]
        self.blank_idx = self.metadata["blank_idx"]
        self.vocab_size = self.metadata["vocab_size"]
        self.decoder_hidden = self.metadata["decoder_hidden"]
        self.decoder_layers = self.metadata["decoder_layers"]

        # Cache shapes
        self.cache_channel_shape = self.metadata["cache_channel_shape"]
        self.cache_time_shape = self.metadata["cache_time_shape"]

    def _get_initial_cache(self):
        """Get initial encoder cache state."""
        cache_channel = np.zeros(self.cache_channel_shape, dtype=np.float32)
        cache_time = np.zeros(self.cache_time_shape, dtype=np.float32)
        cache_len = np.array([0], dtype=np.int32)
        return cache_channel, cache_time, cache_len

    def _get_initial_decoder_state(self):
        """Get initial decoder LSTM state."""
        h = np.zeros((self.decoder_layers, 1, self.decoder_hidden), dtype=np.float32)
        c = np.zeros((self.decoder_layers, 1, self.decoder_hidden), dtype=np.float32)
        return h, c

    def _decode_tokens(self, tokens: list) -> str:
        """Decode token IDs to text."""
        text_parts = []
        for tok in tokens:
            if tok < self.vocab_size and tok != self.blank_idx:
                text_parts.append(self.tokenizer.get(str(tok), ""))
        # Join and handle BPE
        text = "".join(text_parts)
        text = text.replace("▁", " ").strip()
        return text

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio using streaming CoreML inference."""
        # Ensure audio is float32 and has correct shape
        audio = audio.astype(np.float32)
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)

        audio_len = np.array([audio.shape[1]], dtype=np.int32)

        # Get mel spectrogram
        preproc_out = self.preprocessor.predict({
            "audio": audio,
            "audio_length": audio_len
        })
        mel = preproc_out["mel"]
        mel_len = preproc_out["mel_length"][0]

        # Initialize caches
        cache_channel, cache_time, cache_len = self._get_initial_cache()
        h, c = self._get_initial_decoder_state()

        # Initialize with blank token
        last_token = self.blank_idx
        all_tokens = []

        # Process in chunks
        chunk_start = 0
        mel_total_frames = mel.shape[2]

        while chunk_start < mel_total_frames:
            # Get chunk with pre-encode cache
            if chunk_start == 0:
                # First chunk: pad with zeros at the beginning
                chunk_end = min(self.chunk_mel_frames, mel_total_frames)
                chunk_mel = mel[:, :, :chunk_end]
                # Pad to total_mel_frames
                if chunk_mel.shape[2] < self.total_mel_frames:
                    pad_width = self.total_mel_frames - chunk_mel.shape[2]
                    chunk_mel = np.pad(chunk_mel, ((0,0), (0,0), (pad_width, 0)), mode='constant')
            else:
                # Subsequent chunks: include pre-encode cache from previous frames
                cache_start = max(0, chunk_start - self.pre_encode_cache)
                chunk_end = min(chunk_start + self.chunk_mel_frames, mel_total_frames)
                chunk_mel = mel[:, :, cache_start:chunk_end]
                # Pad if needed
                if chunk_mel.shape[2] < self.total_mel_frames:
                    pad_width = self.total_mel_frames - chunk_mel.shape[2]
                    chunk_mel = np.pad(chunk_mel, ((0,0), (0,0), (0, pad_width)), mode='constant')

            chunk_mel_len = np.array([chunk_mel.shape[2]], dtype=np.int32)

            # Run encoder
            enc_out = self.encoder.predict({
                "mel": chunk_mel.astype(np.float32),
                "mel_length": chunk_mel_len,
                "cache_channel": cache_channel,
                "cache_time": cache_time,
                "cache_len": cache_len
            })

            encoded = enc_out["encoded"]
            cache_channel = enc_out["cache_channel_out"]
            cache_time = enc_out["cache_time_out"]
            cache_len = enc_out["cache_len_out"]

            # RNNT decode loop for each encoder frame
            num_enc_frames = encoded.shape[2]
            for t in range(num_enc_frames):
                enc_step = encoded[:, :, t:t+1]

                # Greedy decode loop
                for _ in range(10):  # Max symbols per frame
                    # Run decoder
                    token_input = np.array([[last_token]], dtype=np.int32)
                    token_len = np.array([1], dtype=np.int32)

                    dec_out = self.decoder.predict({
                        "token": token_input,
                        "token_length": token_len,
                        "h_in": h,
                        "c_in": c
                    })

                    decoder_out = dec_out["decoder_out"]
                    h_new = dec_out["h_out"]
                    c_new = dec_out["c_out"]

                    # Run joint
                    joint_out = self.joint.predict({
                        "encoder": enc_step.astype(np.float32),
                        "decoder": decoder_out[:, :, :1].astype(np.float32)
                    })

                    logits = joint_out["logits"]
                    pred_token = int(np.argmax(logits[0, 0, 0, :]))

                    if pred_token == self.blank_idx:
                        # Blank: move to next encoder frame
                        break
                    else:
                        # Non-blank: emit token and continue
                        all_tokens.append(pred_token)
                        last_token = pred_token
                        h = h_new
                        c = c_new

            chunk_start += self.chunk_mel_frames

        return self._decode_tokens(all_tokens)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="nemotron_coreml")
    parser.add_argument("--dataset", type=str, default="datasets/LibriSpeech/test-clean")
    parser.add_argument("--num-files", type=int, default=10)
    args = parser.parse_args()

    print("=" * 70)
    print("NEMOTRON COREML INFERENCE TEST")
    print("=" * 70)

    # Load ground truth
    print(f"\nLoading ground truth from {args.dataset}...")
    gt = load_ground_truth(args.dataset)
    print(f"Loaded {len(gt)} transcriptions")

    # Get audio files
    audio_files = sorted(glob.glob(f"{args.dataset}/**/*.flac", recursive=True))[:args.num_files]
    print(f"Testing on {len(audio_files)} files")

    # Load models
    print()
    inference = NemotronCoreMLInference(args.model_dir)

    # Run inference
    print("\n[COREML STREAMING]")
    total_errors = 0
    total_words = 0

    for i, audio_path in enumerate(audio_files):
        file_id = Path(audio_path).stem
        print(f"  [{i+1}/{len(audio_files)}] {file_id}", end=" ", flush=True)

        audio, sr = sf.read(audio_path, dtype="float32")
        hyp = inference.transcribe(audio)

        if file_id in gt:
            errors, words = compute_wer(gt[file_id], hyp)
            total_errors += errors
            total_words += words
            current_wer = 100 * total_errors / total_words
            print(f"-> {errors} errs, WER so far: {current_wer:.2f}%")
            if errors > 0:
                print(f"       REF: {gt[file_id][:80]}...")
                print(f"       HYP: {hyp[:80]}...")
        else:
            print("-> (no ground truth)")

    wer = 100 * total_errors / total_words if total_words > 0 else 0

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Files tested:  {len(audio_files)}")
    print(f"CoreML WER:    {wer:.2f}%")
    print(f"PyTorch WER:   ~1.88% (reference)")


if __name__ == "__main__":
    main()
