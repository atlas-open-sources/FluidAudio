#!/usr/bin/env python3
"""Test CoreML inference with TRUE streaming (audio chunking) for Nemotron Streaming 0.6B."""
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


class NemotronCoreMLStreaming:
    """TRUE streaming CoreML inference - chunks audio, not just mel."""

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
        self.chunk_mel_frames = self.metadata["chunk_mel_frames"]  # 112
        self.pre_encode_cache = self.metadata["pre_encode_cache"]  # 9
        self.total_mel_frames = self.metadata["total_mel_frames"]  # 121
        self.blank_idx = self.metadata["blank_idx"]
        self.vocab_size = self.metadata["vocab_size"]
        self.decoder_hidden = self.metadata["decoder_hidden"]
        self.decoder_layers = self.metadata["decoder_layers"]
        self.mel_features = self.metadata.get("mel_features", 128)

        # Cache shapes
        self.cache_channel_shape = self.metadata["cache_channel_shape"]
        self.cache_time_shape = self.metadata["cache_time_shape"]

        # Audio chunk size: 1.12 seconds = 112 mel frames * 10ms stride
        # window_stride = 0.01s, so samples_per_chunk = 112 * 0.01 * 16000 = 17920
        self.chunk_samples = int(self.chunk_mel_frames * 0.01 * self.sample_rate)  # 17920

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
        text = "".join(text_parts)
        text = text.replace("▁", " ").strip()
        return text

    def transcribe_streaming(self, audio: np.ndarray) -> str:
        """
        TRUE streaming transcription - processes audio in chunks.

        This simulates real-time streaming where we only have access to
        1.12s of audio at a time, similar to pad_and_drop_preencoded=True.
        """
        audio = audio.astype(np.float32)
        total_samples = len(audio)

        # Initialize states
        cache_channel, cache_time, cache_len = self._get_initial_cache()
        h, c = self._get_initial_decoder_state()
        last_token = self.blank_idx
        all_tokens = []

        # Mel cache for pre_encode_cache (9 frames from previous chunk)
        mel_cache = None

        chunk_idx = 0
        audio_offset = 0

        while audio_offset < total_samples:
            # Get audio chunk
            chunk_end = min(audio_offset + self.chunk_samples, total_samples)
            audio_chunk = audio[audio_offset:chunk_end]

            # Pad if last chunk is short
            if len(audio_chunk) < self.chunk_samples:
                audio_chunk = np.pad(audio_chunk, (0, self.chunk_samples - len(audio_chunk)))

            audio_chunk = audio_chunk.reshape(1, -1)
            audio_len = np.array([audio_chunk.shape[1]], dtype=np.int32)

            # Run preprocessor on this audio chunk only
            preproc_out = self.preprocessor.predict({
                "audio": audio_chunk,
                "audio_length": audio_len
            })
            chunk_mel = preproc_out["mel"]  # [1, 128, ~112]

            # Build input mel: prepend mel_cache (9 frames) + current chunk mel
            if mel_cache is not None:
                # Prepend cached mel frames from previous chunk
                input_mel = np.concatenate([mel_cache, chunk_mel], axis=2)
            else:
                # First chunk: pad with zeros at the beginning
                pad_frames = self.pre_encode_cache
                input_mel = np.pad(chunk_mel, ((0,0), (0,0), (pad_frames, 0)), mode='constant')

            # Ensure we have exactly total_mel_frames (121)
            current_frames = input_mel.shape[2]
            if current_frames < self.total_mel_frames:
                # Pad at the end
                pad_end = self.total_mel_frames - current_frames
                input_mel = np.pad(input_mel, ((0,0), (0,0), (0, pad_end)), mode='constant')
            elif current_frames > self.total_mel_frames:
                # Trim to expected size
                input_mel = input_mel[:, :, :self.total_mel_frames]

            # Save last 9 frames for next chunk's mel cache
            mel_cache = chunk_mel[:, :, -self.pre_encode_cache:] if chunk_mel.shape[2] >= self.pre_encode_cache else chunk_mel

            # Run encoder
            enc_out = self.encoder.predict({
                "mel": input_mel.astype(np.float32),
                "mel_length": np.array([self.total_mel_frames], dtype=np.int32),
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

                    joint_out = self.joint.predict({
                        "encoder": enc_step.astype(np.float32),
                        "decoder": decoder_out[:, :, :1].astype(np.float32)
                    })

                    logits = joint_out["logits"]
                    pred_token = int(np.argmax(logits[0, 0, 0, :]))

                    if pred_token == self.blank_idx:
                        break
                    else:
                        all_tokens.append(pred_token)
                        last_token = pred_token
                        h = h_new
                        c = c_new

            chunk_idx += 1
            audio_offset += self.chunk_samples

        return self._decode_tokens(all_tokens)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="nemotron_coreml")
    parser.add_argument("--dataset", type=str, default="datasets/LibriSpeech/test-clean")
    parser.add_argument("--num-files", type=int, default=10)
    args = parser.parse_args()

    print("=" * 70)
    print("NEMOTRON COREML - TRUE STREAMING TEST")
    print("(Audio chunked at 1.12s, like pad_and_drop_preencoded=True)")
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
    inference = NemotronCoreMLStreaming(args.model_dir)

    # Run inference
    print("\n[TRUE STREAMING - 1.12s audio chunks]")
    total_errors = 0
    total_words = 0

    for i, audio_path in enumerate(audio_files):
        file_id = Path(audio_path).stem
        print(f"  [{i+1}/{len(audio_files)}] {file_id}", end=" ", flush=True)

        audio, sr = sf.read(audio_path, dtype="float32")
        hyp = inference.transcribe_streaming(audio)

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
    print(f"Files tested:      {len(audio_files)}")
    print(f"TRUE Streaming WER: {wer:.2f}%")
    print(f"Expected (PyTorch): ~3.57% (pad_and_drop=True)")
    print(f"Non-streaming WER:  ~1.88% (for comparison)")


if __name__ == "__main__":
    main()
