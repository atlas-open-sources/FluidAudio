#!/usr/bin/env python3
"""Export the fused decoder_joint (B1 inner-loop) model.

Tier-independent — it consumes one encoder step [1, 1024, 1] — so a single
export is copied into every bundle directory passed on the CLI. Interface
matches the shipped decoder_joint.mlmodelc: token / token_length / h_in /
c_in / encoder in; logits / h_out / c_out out.

Pass --prune-tokenizer <ref tokenizer.json> to build the latin-pruned variant
(decoder embedding + joint output sliced to the reference vocab); use it only
for pruned bundle dirs.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List, Optional

import coremltools as ct
import numpy as np
import torch
import typer

import nemo.collections.asr as nemo_asr

from individual_components import DecoderJointWrapper

DEFAULT_MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"

app = typer.Typer(add_completion=False)


@app.command()
def export(
    dest: List[Path] = typer.Option(..., "--dest", help="Bundle dirs to receive decoder_joint.mlpackage"),
    precision: str = typer.Option("FLOAT16", help="FLOAT32 or FLOAT16"),
    prune_tokenizer: Optional[Path] = typer.Option(None, help="Reference tokenizer.json for latin-style pruning"),
) -> None:
    model = nemo_asr.models.ASRModel.from_pretrained(DEFAULT_MODEL_ID, map_location="cpu")
    model.eval()

    if prune_tokenizer is not None:
        ref = json.loads(prune_tokenizer.read_text())
        n_out = len(ref)
        full_vocab = int(model.tokenizer.vocab_size)
        piece_to_full = {model.tokenizer.ids_to_tokens([i])[0]: i for i in range(full_vocab)}
        keep = []
        for nid in range(n_out):
            piece = ref[str(nid)]
            keep.append(int(model.decoder.blank_idx) if piece == "<blank>" else piece_to_full[piece])
        idx = torch.tensor(keep, dtype=torch.long)
        embed = model.decoder.prediction["embed"]
        embed.weight.data = embed.weight.data[idx].clone()
        embed.num_embeddings = n_out
        if getattr(embed, "padding_idx", None) is not None:
            embed.padding_idx = n_out - 1
        out_linear = model.joint.joint_net[2]
        out_linear.weight.data = out_linear.weight.data[idx].clone()
        out_linear.bias.data = out_linear.bias.data[idx].clone()
        out_linear.out_features = n_out
        blank = n_out - 1
        typer.echo(f"vocab pruned to {n_out} rows")
    else:
        blank = int(model.decoder.blank_idx)

    model.decoder._rnnt_export = True
    fused = DecoderJointWrapper(model.decoder.eval(), model.joint.eval())

    decoder_hidden = int(model.decoder.pred_hidden)
    decoder_layers = int(model.decoder.pred_rnn_layers)
    enc_dim = 1024

    token = torch.tensor([[blank]], dtype=torch.int32)
    token_len = torch.tensor([1], dtype=torch.int32)
    h = torch.zeros(decoder_layers, 1, decoder_hidden)
    c = torch.zeros(decoder_layers, 1, decoder_hidden)
    enc_step = torch.randn(1, enc_dim, 1)

    traced = torch.jit.trace(fused, (token, token_len, h, c, enc_step), strict=False)
    inputs = [
        ct.TensorType(name="token", shape=(1, 1), dtype=np.int32),
        ct.TensorType(name="token_length", shape=(1,), dtype=np.int32),
        ct.TensorType(name="h_in", shape=tuple(h.shape), dtype=np.float32),
        ct.TensorType(name="c_in", shape=tuple(c.shape), dtype=np.float32),
        ct.TensorType(name="encoder", shape=tuple(enc_step.shape), dtype=np.float32),
    ]
    outputs = [
        ct.TensorType(name="logits", dtype=np.float32),
        ct.TensorType(name="h_out", dtype=np.float32),
        ct.TensorType(name="c_out", dtype=np.float32),
    ]
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=inputs,
        outputs=outputs,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16 if precision.upper() == "FLOAT16" else ct.precision.FLOAT32,
    )

    staging = Path("decoder_joint_staging.mlpackage")
    if staging.exists():
        shutil.rmtree(staging)
    mlmodel.save(str(staging))
    for d in dest:
        target = d / "decoder_joint.mlpackage"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(staging, target)
        typer.echo(f"-> {target}")
    typer.echo("Done!")


if __name__ == "__main__":
    app()
