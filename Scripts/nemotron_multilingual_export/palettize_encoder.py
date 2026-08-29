#!/usr/bin/env python3
"""Palettize the encoder to 4-bit LUT weights (the shipped bundles' trick).

Encoder energy per chunk is substantially memory traffic; halving weight
bytes vs int8 cuts joules per chunk without touching behavior. Applied to
the FP16 export (FP16 compute stays ANE-resident; the LUT decompresses via
constexpr_lut_to_dense like the shipped bundles).
"""
from pathlib import Path
import shutil

import typer
import coremltools as ct
from coremltools.optimize.coreml import (
    OptimizationConfig,
    OpPalettizerConfig,
    palettize_weights,
)

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def _dir_size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / (1024 * 1024)


@app.command()
def palettize(
    model_dir: Path = typer.Option(..., "--model-dir", help="Bundle dir with an FP16 encoder.mlpackage"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Output bundle dir"),
    nbits: int = typer.Option(4, help="LUT bits per weight"),
    mode: str = typer.Option("kmeans", help="kmeans | uniform"),
    granularity: str = typer.Option(
        "per_grouped_channel",
        help="per_tensor | per_grouped_channel (global per-tensor 4-bit destroys this model: 97%+ CER)",
    ),
    group_size: int = typer.Option(16, help="channels per LUT for per_grouped_channel"),
    per_channel_scale: bool = typer.Option(False, help="add per-channel scales to the LUT"),
) -> None:
    encoder_path = model_dir / "encoder.mlpackage"
    if not encoder_path.exists():
        raise typer.BadParameter(f"encoder.mlpackage not found in {model_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = _dir_size_mb(encoder_path)
    model = ct.models.MLModel(str(encoder_path))
    # per_grouped_channel LUT ops are iOS18+ (spec version 9). The exports were
    # cut at iOS17; the app requires iOS 18.2, so raising the spec is free.
    spec = model.get_spec()
    if spec.specificationVersion < 9:
        spec.specificationVersion = 9
        model = ct.models.MLModel(
            spec, weights_dir=model.weights_dir, compute_units=ct.ComputeUnit.CPU_ONLY
        )
        typer.echo("spec version raised to 9 (iOS18)")
    if granularity == "per_grouped_channel":
        op_config = OpPalettizerConfig(
            nbits=nbits,
            mode=mode,
            granularity="per_grouped_channel",
            group_size=group_size,
            enable_per_channel_scale=per_channel_scale,
        )
    else:
        op_config = OpPalettizerConfig(nbits=nbits, mode=mode)
    config = OptimizationConfig(global_config=op_config)
    typer.echo(f"palettizing to {nbits}-bit ({mode}, {granularity})... this can take a while")
    palettized = palettize_weights(model, config)
    out_encoder = output_dir / "encoder.mlpackage"
    if out_encoder.exists():
        shutil.rmtree(out_encoder)
    palettized.save(str(out_encoder))
    typer.echo(f"Encoder: {baseline:.0f} MB -> {_dir_size_mb(out_encoder):.0f} MB")

    for name in [
        "decoder.mlpackage",
        "joint.mlpackage",
        "preprocessor.mlpackage",
        "decoder_joint.mlpackage",
        "metadata.json",
        "tokenizer.json",
    ]:
        src = model_dir / name
        dst = output_dir / name
        if src.exists() and not dst.exists():
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
    typer.echo("Done!")


if __name__ == "__main__":
    app()
