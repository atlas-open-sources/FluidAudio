#!/usr/bin/env python3
"""Export Nemotron Speech Streaming 0.6B to CoreML.

Exports 4 components for streaming RNNT inference:
1. Preprocessor: audio → mel
2. Encoder: mel + cache → encoded + new_cache
3. Decoder: token + state → decoder_out + new_state
4. Joint: encoder + decoder → logits
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import coremltools as ct
import numpy as np
import torch
import typer

import nemo.collections.asr as nemo_asr

from individual_components import (
    DecoderWrapper,
    EncoderStreamingWrapper,
    ExportSettings,
    JointWrapper,
    PreprocessorWrapper,
    _coreml_convert,
)

DEFAULT_MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"

# Streaming config from model:
# chunk_size=[105, 112], pre_encode_cache_size=[0, 9], valid_out_len=14
CHUNK_MEL_FRAMES = 112
PRE_ENCODE_CACHE = 9
TOTAL_MEL_FRAMES = CHUNK_MEL_FRAMES + PRE_ENCODE_CACHE  # 121


def _tensor_shape(t: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(d) for d in t.shape)


def _parse_cu(name: str) -> ct.ComputeUnit:
    mapping = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    }
    return mapping.get(name.upper(), ct.ComputeUnit.CPU_ONLY)


app = typer.Typer(add_completion=False)


@app.command()
def convert(
    output_dir: Path = typer.Option(Path("nemotron_coreml"), help="Output directory"),
    encoder_cu: str = typer.Option("CPU_AND_NE", help="Encoder compute units"),
    precision: str = typer.Option("FLOAT32", help="FLOAT32 or FLOAT16"),
    lookahead: int = typer.Option(6, help="Encoder lookahead tokens: 0=80ms, 3=320ms, 6=560ms, 13=1120ms"),
    prune_tokenizer: Optional[Path] = typer.Option(
        None,
        help=(
            "Path to a pruned tokenizer.json (id->piece map including '<blank>' at its "
            "blank idx, e.g. a shipped latin bundle's). Slices the decoder embedding and "
            "joint output rows to exactly that vocabulary, producing a latin-style ship "
            "whose token ids match the reference tokenizer."
        ),
    ),
) -> None:
    """Export Nemotron Streaming to CoreML."""
    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("Loading model...")
    # ASRModel auto-detects the concrete class. This checkpoint is
    # EncDecRNNTBPEModelWithPrompt, NOT the unprompted class the English
    # export hardcoded.
    model = nemo_asr.models.ASRModel.from_pretrained(DEFAULT_MODEL_ID, map_location="cpu")
    model.eval()

    sample_rate = int(model.cfg.preprocessor.sample_rate)
    encoder = model.encoder
    supported = [c[1] for c in model.cfg.encoder.att_context_size]
    if lookahead not in supported:
        raise typer.BadParameter(f"lookahead {lookahead} not in checkpoint supported {supported}")
    left = int(model.cfg.encoder.att_context_size[0][0])
    encoder.set_default_att_context_size([left, lookahead])
    encoder.setup_streaming_params(att_context_size=[left, lookahead])

    # DERIVED, not hardcoded. The published script baked in the 1120 ms numbers;
    # reading them back off streaming_cfg is what makes another lookahead
    # correct rather than merely exportable.
    scfg = encoder.streaming_cfg

    def _last(v):
        return int(v[-1]) if isinstance(v, (list, tuple)) else int(v)

    global CHUNK_MEL_FRAMES, PRE_ENCODE_CACHE, TOTAL_MEL_FRAMES
    CHUNK_MEL_FRAMES = _last(scfg.chunk_size)
    PRE_ENCODE_CACHE = _last(scfg.pre_encode_cache_size)
    TOTAL_MEL_FRAMES = CHUNK_MEL_FRAMES + PRE_ENCODE_CACHE
    chunk_frames = int(scfg.valid_out_len)
    typer.echo(
        f"lookahead={lookahead} att_context=[{left},{lookahead}] "
        f"chunk_mel={CHUNK_MEL_FRAMES} pre_encode_cache={PRE_ENCODE_CACHE} "
        f"total_mel={TOTAL_MEL_FRAMES} enc_frames={chunk_frames} "
        f"chunk_ms={CHUNK_MEL_FRAMES * 10}"
    )

    # Get cache shapes
    cache_channel, cache_time, cache_len = encoder.get_initial_cache_state(batch_size=1, device="cpu")
    cache_len = cache_len.to(torch.int32)

    # Transpose to [B, L, ...] for CoreML
    cache_channel_b = cache_channel.transpose(0, 1)
    cache_time_b = cache_time.transpose(0, 1)

    typer.echo(f"Cache shapes: channel={cache_channel_b.shape}, time={cache_time_b.shape}")

    # Language-prompt plumbing (EncDecRNNTBPEModelWithPrompt): the prompt is
    # applied to the ENCODED output via one-hot concat + prompt_kernel linear
    # (PromptStreamingMixin._apply_prompt_to_encoded); we fuse it into the
    # exported encoder so it takes prompt_id like the shipped bundles do.
    prompt_dict_cfg = model.cfg.model_defaults.get("prompt_dictionary", {})
    prompt_dictionary = {str(k): int(v) for k, v in dict(prompt_dict_cfg).items()}
    num_prompts = int(model.cfg.get("num_prompts", 0)) or int(
        model.cfg.model_defaults.get("num_prompts", 0)
    )
    default_prompt_id = int(prompt_dictionary.get("auto", 101))
    if not hasattr(model, "prompt_kernel") or num_prompts == 0:
        raise typer.BadParameter("checkpoint has no prompt_kernel — wrong model class?")
    typer.echo(
        f"prompt: num_prompts={num_prompts} dict_entries={len(prompt_dictionary)} "
        f"default_prompt_id={default_prompt_id}"
    )

    # Optional vocab pruning (latin-style ship): slice the decoder embedding and
    # joint output rows down to the reference tokenizer's pieces, so the bundle's
    # token ids match that tokenizer exactly. The encoder is untouched.
    vocab_pruned = prune_tokenizer is not None
    if vocab_pruned:
        ref = json.loads(Path(prune_tokenizer).read_text())
        n_out = len(ref)
        full_vocab = int(model.tokenizer.vocab_size)
        piece_to_full = {model.tokenizer.ids_to_tokens([i])[0]: i for i in range(full_vocab)}
        keep = []
        missing = []
        for nid in range(n_out):
            piece = ref[str(nid)]
            if piece == "<blank>":
                keep.append(int(model.decoder.blank_idx))
            elif piece in piece_to_full:
                keep.append(piece_to_full[piece])
            else:
                missing.append(piece)
        if missing:
            raise typer.BadParameter(
                f"{len(missing)} pieces of the reference tokenizer are absent from this "
                f"checkpoint (first: {missing[:3]}) — tokenizers are incompatible"
            )
        idx = torch.tensor(keep, dtype=torch.long)
        embed = model.decoder.prediction["embed"]
        embed.weight.data = embed.weight.data[idx].clone()
        embed.num_embeddings = n_out
        # NeMo builds the embedding with padding_idx = blank; repoint it at the
        # pruned blank or nn.Embedding asserts it is within num_embeddings.
        if getattr(embed, "padding_idx", None) is not None:
            embed.padding_idx = n_out - 1
        out_linear = model.joint.joint_net[2]
        out_linear.weight.data = out_linear.weight.data[idx].clone()
        out_linear.bias.data = out_linear.bias.data[idx].clone()
        out_linear.out_features = n_out
        blank_idx_out = n_out - 1
        vocab_size_out = n_out - 1
        pieces_out = {i: ref[str(i)] for i in range(vocab_size_out)}
        typer.echo(f"vocab pruned: {full_vocab + 1} -> {n_out} rows (blank at {blank_idx_out})")
    else:
        blank_idx_out = int(model.decoder.blank_idx)
        vocab_size_out = int(model.tokenizer.vocab_size)
        pieces_out = None  # built later from the model tokenizer

    # Create wrappers
    preprocessor = PreprocessorWrapper(model.preprocessor.eval())
    encoder_streaming = EncoderStreamingWrapper(
        encoder.eval(),
        prompt_kernel=model.prompt_kernel.eval(),
        num_prompts=num_prompts,
    )
    decoder = DecoderWrapper(model.decoder.eval())
    joint = JointWrapper(model.joint.eval())

    model.decoder._rnnt_export = True

    settings = ExportSettings(
        output_dir=output_dir,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16 if precision.upper() == "FLOAT16" else ct.precision.FLOAT32,
        max_audio_seconds=30.0,
        max_symbol_steps=1,
        chunk_size_frames=chunk_frames,
        cache_size=cache_channel.shape[2],
    )

    # === Preprocessor ===
    typer.echo("Exporting preprocessor...")
    max_samples = 30 * sample_rate
    audio = torch.randn(1, max_samples)
    audio_len = torch.tensor([max_samples], dtype=torch.int32)

    traced = torch.jit.trace(preprocessor, (audio, audio_len), strict=False)
    inputs = [
        ct.TensorType(name="audio", shape=(1, ct.RangeDim(1, max_samples)), dtype=np.float32),
        ct.TensorType(name="audio_length", shape=(1,), dtype=np.int32),
    ]
    outputs = [
        ct.TensorType(name="mel", dtype=np.float32),
        ct.TensorType(name="mel_length", dtype=np.int32),
    ]
    mlmodel = _coreml_convert(traced, inputs, outputs, settings, ct.ComputeUnit.CPU_ONLY)
    mlmodel.save(str(output_dir / "preprocessor.mlpackage"))

    # === Encoder (streaming) ===
    typer.echo("Exporting encoder...")
    mel_features = int(model.cfg.preprocessor.features)  # 128 for this model
    mel = torch.randn(1, mel_features, TOTAL_MEL_FRAMES)
    mel_len = torch.tensor([TOTAL_MEL_FRAMES], dtype=torch.int32)
    prompt_id_t = torch.tensor([default_prompt_id], dtype=torch.int32)

    traced = torch.jit.trace(
        encoder_streaming,
        (mel, mel_len, cache_channel_b, cache_time_b, cache_len, prompt_id_t),
        strict=False
    )
    inputs = [
        ct.TensorType(name="mel", shape=_tensor_shape(mel), dtype=np.float32),
        ct.TensorType(name="mel_length", shape=(1,), dtype=np.int32),
        ct.TensorType(name="cache_channel", shape=_tensor_shape(cache_channel_b), dtype=np.float32),
        ct.TensorType(name="cache_time", shape=_tensor_shape(cache_time_b), dtype=np.float32),
        ct.TensorType(name="cache_len", shape=(1,), dtype=np.int32),
        ct.TensorType(name="prompt_id", shape=(1,), dtype=np.int32),
    ]
    outputs = [
        ct.TensorType(name="encoded", dtype=np.float32),
        ct.TensorType(name="encoded_length", dtype=np.int32),
        ct.TensorType(name="cache_channel_out", dtype=np.float32),
        ct.TensorType(name="cache_time_out", dtype=np.float32),
        ct.TensorType(name="cache_len_out", dtype=np.int32),
    ]
    mlmodel = _coreml_convert(traced, inputs, outputs, settings, _parse_cu(encoder_cu))
    mlmodel.save(str(output_dir / "encoder.mlpackage"))

    # === Decoder ===
    typer.echo("Exporting decoder...")
    decoder_hidden = int(model.decoder.pred_hidden)
    decoder_layers = int(model.decoder.pred_rnn_layers)

    targets = torch.tensor([[blank_idx_out]], dtype=torch.int32)
    target_len = torch.tensor([1], dtype=torch.int32)
    h = torch.zeros(decoder_layers, 1, decoder_hidden)
    c = torch.zeros(decoder_layers, 1, decoder_hidden)

    traced = torch.jit.trace(decoder, (targets, target_len, h, c), strict=False)
    inputs = [
        ct.TensorType(name="token", shape=(1, 1), dtype=np.int32),
        ct.TensorType(name="token_length", shape=(1,), dtype=np.int32),
        ct.TensorType(name="h_in", shape=_tensor_shape(h), dtype=np.float32),
        ct.TensorType(name="c_in", shape=_tensor_shape(c), dtype=np.float32),
    ]
    outputs = [
        ct.TensorType(name="decoder_out", dtype=np.float32),
        ct.TensorType(name="h_out", dtype=np.float32),
        ct.TensorType(name="c_out", dtype=np.float32),
    ]
    mlmodel = _coreml_convert(traced, inputs, outputs, settings, ct.ComputeUnit.CPU_ONLY)
    mlmodel.save(str(output_dir / "decoder.mlpackage"))

    # === Joint ===
    typer.echo("Exporting joint...")
    with torch.no_grad():
        mel_test, _ = preprocessor(audio[:, :sample_rate], torch.tensor([sample_rate], dtype=torch.int32))
        # Run through encoder wrapper (not model.encoder directly to avoid typed method issues)
        enc_out, _, _, _, _ = encoder_streaming(
            mel_test,
            torch.tensor([mel_test.shape[2]], dtype=torch.int32),
            cache_channel_b,
            cache_time_b,
            cache_len,
            prompt_id_t,
        )
        dec_out, _, _ = decoder(targets, target_len, h, c)

    # Single step: [B, D, 1]
    enc_step = enc_out[:, :, :1].contiguous()
    dec_step = dec_out[:, :, :1].contiguous()

    traced = torch.jit.trace(joint, (enc_step, dec_step), strict=False)
    inputs = [
        ct.TensorType(name="encoder", shape=_tensor_shape(enc_step), dtype=np.float32),
        ct.TensorType(name="decoder", shape=_tensor_shape(dec_step), dtype=np.float32),
    ]
    outputs = [ct.TensorType(name="logits", dtype=np.float32)]
    mlmodel = _coreml_convert(traced, inputs, outputs, settings, ct.ComputeUnit.CPU_ONLY)
    mlmodel.save(str(output_dir / "joint.mlpackage"))

    # === Metadata ===
    vocab_size = vocab_size_out

    # Language-tag tokens (<en-US> etc.) that the model emits as a leading
    # token; FluidAudio filters them from transcripts and surfaces the first
    # as the detected language.
    import re as _re

    if pieces_out is None:
        pieces_out = {i: model.tokenizer.ids_to_tokens([i])[0] for i in range(vocab_size)}
    pieces = pieces_out
    lang_tag_token_ids = sorted(
        i
        for i, piece in pieces.items()
        if _re.fullmatch(r"<[a-z]{2,3}(-[A-Za-z]{2,4})?>", piece)
        and piece not in ("<unk>", "<pad>", "<blank>")
    )
    typer.echo(f"lang_tag_token_ids: {len(lang_tag_token_ids)} found")

    metadata = {
        "model": DEFAULT_MODEL_ID,
        "sample_rate": sample_rate,
        "mel_features": mel_features,
        "chunk_mel_frames": CHUNK_MEL_FRAMES,
        "pre_encode_cache": PRE_ENCODE_CACHE,
        "total_mel_frames": TOTAL_MEL_FRAMES,
        "vocab_size": vocab_size,
        "blank_idx": blank_idx_out,
        "cache_channel_shape": list(cache_channel_b.shape),
        "cache_time_shape": list(cache_time_b.shape),
        "decoder_hidden": decoder_hidden,
        "decoder_layers": decoder_layers,
        "encoder_dim": int(enc_out.shape[1]),
        "att_context_size": [int(left), int(lookahead)],
        "chunk_ms": int(CHUNK_MEL_FRAMES * 10),
        "num_prompts": num_prompts,
        "default_prompt_id": default_prompt_id,
        "prompt_dictionary": prompt_dictionary,
        "lang_tag_token_ids": lang_tag_token_ids,
        "model_class": type(model).__module__ + "." + type(model).__name__,
    }
    if vocab_pruned:
        metadata["vocab_pruned"] = True
        metadata["vocab_pruned_original_size"] = int(model.tokenizer.vocab_size)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Tokenizer — the shipped bundles include an explicit "<blank>" entry at
    # blank_idx (== vocab_size); FluidAudio's corruption check relies on it.
    tokenizer = {str(i): pieces[i] for i in range(vocab_size)}
    tokenizer[str(blank_idx_out)] = "<blank>"
    (output_dir / "tokenizer.json").write_text(json.dumps(tokenizer, indent=2))

    typer.echo(f"Done! Exported to {output_dir}")


if __name__ == "__main__":
    app()
