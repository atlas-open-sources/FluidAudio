#!/usr/bin/env python3
"""
WER Benchmark for Nemotron Streaming 0.6b on LibriSpeech test-clean
"""
import glob
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer


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
    """Normalize text for WER calculation - remove punctuation, lowercase."""
    import re
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


def calc_drop_extra_pre_encoded(model, step_num, pad_and_drop_preencoded):
    """Calculate drop_extra_pre_encoded value per NVIDIA's reference."""
    if step_num == 0 and not pad_and_drop_preencoded:
        return 0
    return model.encoder.streaming_cfg.drop_extra_pre_encoded


def transcribe_streaming(model, audio: np.ndarray, pad_and_drop_preencoded: bool = False) -> str:
    """Streaming transcription using conformer_stream_step API."""
    model.encoder.setup_streaming_params()

    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=model,
        pad_and_drop_preencoded=pad_and_drop_preencoded,
    )
    streaming_buffer.reset_buffer()
    streaming_buffer.append_audio(audio)

    cache_last_channel, cache_last_time, cache_last_channel_len = \
        model.encoder.get_initial_cache_state(batch_size=1)

    previous_hypotheses = None
    pred_out_stream = None
    final_text = ""

    with torch.inference_mode():
        for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer):
            (
                pred_out_stream,
                transcribed_texts,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
                previous_hypotheses,
            ) = model.conformer_stream_step(
                processed_signal=chunk_audio,
                processed_signal_length=chunk_lengths,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                keep_all_outputs=streaming_buffer.is_buffer_empty(),
                previous_hypotheses=previous_hypotheses,
                previous_pred_out=pred_out_stream,
                drop_extra_pre_encoded=calc_drop_extra_pre_encoded(model, step_num, pad_and_drop_preencoded),
                return_transcription=True,
            )

            if transcribed_texts and len(transcribed_texts) > 0:
                text = transcribed_texts[0]
                if hasattr(text, 'text'):
                    final_text = text.text
                else:
                    final_text = str(text)

    return final_text


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-files", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="datasets/LibriSpeech/test-clean")
    args = parser.parse_args()

    print("=" * 70)
    print("NEMOTRON STREAMING 0.6B - WER BENCHMARK")
    print("=" * 70)

    # Load ground truth
    print(f"\nLoading ground truth from {args.dataset}...")
    gt = load_ground_truth(args.dataset)
    print(f"Loaded {len(gt)} transcriptions")

    # Get audio files
    audio_files = sorted(glob.glob(f"{args.dataset}/**/*.flac", recursive=True))[:args.num_files]
    print(f"Testing on {len(audio_files)} files")

    # Load model
    print("\nLoading model...")
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/nemotron-speech-streaming-en-0.6b")
    model.eval()

    # Streaming transcription
    print("\n[STREAMING MODE]")
    stream_errors = 0
    stream_words = 0

    for i, audio_path in enumerate(audio_files):
        file_id = Path(audio_path).stem
        print(f"  [{i+1}/{len(audio_files)}] {file_id}", end=" ", flush=True)

        audio, sr = sf.read(audio_path, dtype="float32")
        hyp = transcribe_streaming(model, audio)

        if file_id in gt:
            errors, words = compute_wer(gt[file_id], hyp)
            stream_errors += errors
            stream_words += words
            current_wer = 100 * stream_errors / stream_words
            print(f"-> {errors} errs, WER so far: {current_wer:.2f}%")
        else:
            print("-> (no ground truth)")

    stream_wer = 100 * stream_errors / stream_words if stream_words > 0 else 0

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Files tested:    {len(audio_files)}")
    print(f"Streaming WER:   {stream_wer:.2f}%")
    print(f"NVIDIA claimed:  2.31%")


if __name__ == "__main__":
    main()
