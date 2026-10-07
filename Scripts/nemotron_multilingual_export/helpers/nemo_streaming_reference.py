#!/usr/bin/env python3
"""
NeMo Nemotron Streaming Reference Implementation

Streaming inference with nemotron-speech-streaming-en-0.6b using 1.12s chunks.
Uses conformer_stream_step API with CacheAwareStreamingAudioBuffer.
"""
import numpy as np
import soundfile as sf
import torch
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer


def calc_drop_extra_pre_encoded(model, step_num, pad_and_drop_preencoded):
    """Calculate drop_extra_pre_encoded value per NVIDIA's reference."""
    if step_num == 0 and not pad_and_drop_preencoded:
        return 0
    return model.encoder.streaming_cfg.drop_extra_pre_encoded


def transcribe_streaming(model, audio: np.ndarray, sr: int = 16000, pad_and_drop_preencoded: bool = False) -> str:
    """
    Streaming transcription using NeMo's conformer_stream_step API.

    Args:
        model: NeMo ASR model (must support streaming)
        audio: Audio samples as float32 numpy array
        sr: Sample rate (must be 16000)
        pad_and_drop_preencoded: Whether to pad and drop preencoded frames.
            False (default) gives better WER, True is needed for ONNX export.

    Returns:
        Transcribed text
    """
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
    parser = argparse.ArgumentParser(description="NeMo Streaming Reference")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds to transcribe")
    args = parser.parse_args()

    audio, sr = sf.read(args.audio, dtype="float32")
    if args.duration:
        audio = audio[:int(args.duration * sr)]

    print("=" * 70)
    print("NEMOTRON STREAMING")
    print("=" * 70)
    print(f"Audio: {len(audio)/sr:.1f}s @ {sr}Hz")

    print("\nLoading model...")
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/nemotron-speech-streaming-en-0.6b")
    model.eval()

    print("\n[STREAMING MODE] (1.12s chunks)")
    text = transcribe_streaming(model, audio, sr)
    print(f"  {text}")


if __name__ == "__main__":
    main()
