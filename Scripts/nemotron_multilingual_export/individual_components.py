#!/usr/bin/env python3
"""Export Parakeet Realtime EOU RNNT components into CoreML.

This model uses a cache-aware streaming FastConformer encoder.
The encoder requires splitting into:
1. Initial encoder (no cache, for first chunk)
2. Streaming encoder (with cache inputs/outputs)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import coremltools as ct
import torch


@dataclass
class ExportSettings:
    output_dir: Path
    compute_units: ct.ComputeUnit
    deployment_target: Optional[ct.target]
    compute_precision: Optional[ct.precision]
    max_audio_seconds: float
    max_symbol_steps: int
    # Streaming-specific settings
    chunk_size_frames: int  # Number of frames per chunk (after subsampling)
    cache_size: int  # Size of the channel cache


class PreprocessorWrapper(torch.nn.Module):
    """Wrapper for the preprocessor (mel spectrogram extraction)."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(
        self, audio_signal: torch.Tensor, length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mel, mel_length = self.module(
            input_signal=audio_signal, length=length.to(dtype=torch.long)
        )
        return mel, mel_length


class EncoderInitialWrapper(torch.nn.Module):
    """Encoder wrapper for the initial chunk (no cache input).

    This is used for the first chunk of audio where there's no previous cache.
    It outputs the encoder features and initial cache states.
    """

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(
        self, features: torch.Tensor, length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for initial chunk without cache.

        Args:
            features: Mel spectrogram [B, D, T]
            length: Sequence lengths [B]

        Returns:
            encoded: Encoder output [B, D, T_enc]
            encoded_lengths: Output lengths [B]
        """
        # Initial forward without cache
        encoded, encoded_lengths = self.module(
            audio_signal=features, length=length.to(dtype=torch.long)
        )
        return encoded, encoded_lengths


class EncoderStreamingWrapper(torch.nn.Module):
    """Encoder wrapper for streaming with cache.

    This is used for subsequent chunks where cache states are available.
    It takes cache states as input and outputs updated cache states.

    When `prompt_kernel` is provided (the prompted multilingual checkpoint,
    EncDecRNNTBPEModelWithPrompt), the wrapper takes an extra `prompt_id`
    int32 [1] input and fuses the language-prompt projection into the graph:
    one-hot(prompt_id) is concatenated to each encoded frame and passed
    through `prompt_kernel` (Linear(D + num_prompts -> D)). This replicates
    NeMo's PromptStreamingMixin._apply_prompt_to_encoded, which the shipped
    FluidInference encoder.mlmodelc also bakes in.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        prompt_kernel: Optional[torch.nn.Module] = None,
        num_prompts: int = 0,
    ) -> None:
        super().__init__()
        self.module = module
        self.prompt_kernel = prompt_kernel
        self.num_prompts = int(num_prompts)

    def forward(
        self,
        features: torch.Tensor,
        length: torch.Tensor,
        cache_last_channel: torch.Tensor,
        cache_last_time: torch.Tensor,
        cache_last_channel_len: torch.Tensor,
        prompt_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with cache for streaming."""
        # Transpose caches from [B, L, ...] to [L, B, ...] for NeMo
        cache_last_channel_t = cache_last_channel.transpose(0, 1)
        cache_last_time_t = cache_last_time.transpose(0, 1)
        cache_len_i64 = cache_last_channel_len.to(dtype=torch.int64)

        # Call encoder forward with cache parameters
        encoded, encoded_lengths, cache_ch_next, cache_t_next, cache_len_next = self.module(
            audio_signal=features,
            length=length.to(dtype=torch.long),
            cache_last_channel=cache_last_channel_t,
            cache_last_time=cache_last_time_t,
            cache_last_channel_len=cache_len_i64,
        )

        if self.prompt_kernel is not None and prompt_id is not None:
            enc_t = encoded.transpose(1, 2)  # (B, D, T) -> (B, T, D)
            onehot = torch.zeros(
                enc_t.shape[0], enc_t.shape[1], self.num_prompts, dtype=enc_t.dtype
            )
            idx = (
                prompt_id.to(torch.long)
                .view(-1, 1, 1)
                .expand(enc_t.shape[0], enc_t.shape[1], 1)
            )
            onehot = onehot.scatter(2, idx, 1.0)
            enc_t = self.prompt_kernel(torch.cat([enc_t, onehot], dim=-1)).to(encoded.dtype)
            encoded = enc_t.transpose(1, 2)  # (B, T, D) -> (B, D, T)

        # Transpose caches back from [L, B, ...] to [B, L, ...]
        cache_ch_next = cache_ch_next.transpose(0, 1)
        cache_t_next = cache_t_next.transpose(0, 1)

        return (
            encoded,
            encoded_lengths.to(dtype=torch.int32),
            cache_ch_next,
            cache_t_next,
            cache_len_next.to(dtype=torch.int32),
        )


class DecoderWrapper(torch.nn.Module):
    """Wrapper for the RNNT prediction network (decoder)."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(
        self,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        h_in: torch.Tensor,
        c_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = [h_in, c_in]
        decoder_output, _, new_state = self.module(
            targets=targets.to(dtype=torch.long),
            target_length=target_lengths.to(dtype=torch.long),
            states=state,
        )
        return decoder_output, new_state[0], new_state[1]


class JointWrapper(torch.nn.Module):
    """Wrapper for the RNNT joint network."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(
        self, encoder_outputs: torch.Tensor, decoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        # Input: encoder_outputs [B, D, T], decoder_outputs [B, D, U]
        # Transpose to match what projection layers expect
        encoder_outputs = encoder_outputs.transpose(1, 2)  # [B, T, D]
        decoder_outputs = decoder_outputs.transpose(1, 2)  # [B, U, D]

        # Apply projections
        enc_proj = self.module.enc(encoder_outputs)  # [B, T, joint_dim]
        dec_proj = self.module.pred(decoder_outputs)  # [B, U, joint_dim]

        # Explicit broadcasting along T and U
        x = enc_proj.unsqueeze(2) + dec_proj.unsqueeze(1)  # [B, T, U, joint_dim]
        x = self.module.joint_net[0](x)  # ReLU
        x = self.module.joint_net[1](x)  # Dropout (no-op in eval)
        out = self.module.joint_net[2](x)  # Linear -> logits
        return out


class MelEncoderWrapper(torch.nn.Module):
    """Fused wrapper: waveform -> mel -> encoder (no cache, initial chunk)."""

    def __init__(
        self, preprocessor: PreprocessorWrapper, encoder: EncoderInitialWrapper
    ) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.encoder = encoder

    def forward(
        self, audio_signal: torch.Tensor, audio_length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mel, mel_length = self.preprocessor(audio_signal, audio_length)
        encoded, enc_len = self.encoder(mel, mel_length.to(dtype=torch.int32))
        return encoded, enc_len


class MelEncoderStreamingWrapper(torch.nn.Module):
    """Fused wrapper: waveform -> mel -> encoder (with cache, streaming)."""

    def __init__(
        self, preprocessor: PreprocessorWrapper, encoder: EncoderStreamingWrapper
    ) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.encoder = encoder

    def forward(
        self,
        audio_signal: torch.Tensor,
        audio_length: torch.Tensor,
        cache_last_channel: torch.Tensor,
        cache_last_time: torch.Tensor,
        cache_last_channel_len: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        mel, mel_length = self.preprocessor(audio_signal, audio_length)
        return self.encoder(
            mel,
            mel_length.to(dtype=torch.int32),
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
        )


class JointDecisionWrapper(torch.nn.Module):
    """Joint + decision head: outputs label id, label prob.

    Unlike TDT, EOU models don't have duration outputs.
    They have a special EOU token that marks end of utterance.
    """

    def __init__(self, joint: JointWrapper, vocab_size: int) -> None:
        super().__init__()
        self.joint = joint
        self.vocab_with_blank = int(vocab_size) + 1

    def forward(
        self, encoder_outputs: torch.Tensor, decoder_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.joint(encoder_outputs, decoder_outputs)
        token_logits = logits[..., : self.vocab_with_blank]

        # Token selection
        token_ids = torch.argmax(token_logits, dim=-1).to(dtype=torch.int32)
        token_probs_all = torch.softmax(token_logits, dim=-1)
        token_prob = torch.gather(
            token_probs_all, dim=-1, index=token_ids.long().unsqueeze(-1)
        ).squeeze(-1)

        return token_ids, token_prob


class JointDecisionSingleStep(torch.nn.Module):
    """Single-step variant for streaming: encoder_step [1, D, 1] -> [1,1,1].

    Inputs:
      - encoder_step: [B=1, D, T=1]
      - decoder_step: [B=1, D, U=1]

    Returns:
      - token_id: [1, 1, 1] int32
      - token_prob: [1, 1, 1] float32
      - top_k_ids: [1, 1, 1, K] int32
      - top_k_logits: [1, 1, 1, K] float32
    """

    def __init__(self, joint: JointWrapper, vocab_size: int, top_k: int = 64) -> None:
        super().__init__()
        self.joint = joint
        self.vocab_with_blank = int(vocab_size) + 1
        self.top_k = int(top_k)

    def forward(
        self, encoder_step: torch.Tensor, decoder_step: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.joint(encoder_step, decoder_step)  # [1, 1, 1, V]
        token_logits = logits[..., : self.vocab_with_blank]

        token_ids = torch.argmax(token_logits, dim=-1, keepdim=False).to(
            dtype=torch.int32
        )
        token_probs_all = torch.softmax(token_logits, dim=-1)
        token_prob = torch.gather(
            token_probs_all, dim=-1, index=token_ids.long().unsqueeze(-1)
        ).squeeze(-1)

        # Top-K candidates for host-side re-ranking
        topk_logits, topk_ids_long = torch.topk(
            token_logits, k=min(self.top_k, token_logits.shape[-1]), dim=-1
        )
        topk_ids = topk_ids_long.to(dtype=torch.int32)
        return token_ids, token_prob, topk_ids, topk_logits


def _coreml_convert(
    traced: torch.jit.ScriptModule,
    inputs,
    outputs,
    settings: ExportSettings,
    compute_units_override: Optional[ct.ComputeUnit] = None,
) -> ct.models.MLModel:
    cu = (
        compute_units_override
        if compute_units_override is not None
        else settings.compute_units
    )
    kwargs = {
        "convert_to": "mlprogram",
        "inputs": inputs,
        "outputs": outputs,
        "compute_units": cu,
    }
    print("Converting:", traced.__class__.__name__)
    print("Conversion kwargs:", kwargs)
    if settings.deployment_target is not None:
        kwargs["minimum_deployment_target"] = settings.deployment_target
    if settings.compute_precision is not None:
        kwargs["compute_precision"] = settings.compute_precision
    return ct.convert(traced, **kwargs)
