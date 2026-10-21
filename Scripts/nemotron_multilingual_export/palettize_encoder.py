#!/usr/bin/env python3
"""Palettize a verified FP16 parent; measure numerical changes on real input."""
from pathlib import Path
import shutil
from qualification_provenance import begin_derivative, finish_derivative, validate_compute_units

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
    compute_units: str = typer.Option("CPU_ONLY", help="Explicit compute units for parent and transformed model validation"),
) -> None:
    encoder_path = model_dir / "encoder.mlpackage"
    if not encoder_path.exists():
        raise typer.BadParameter(f"encoder.mlpackage not found in {model_dir}")
    if nbits not in {1, 2, 3, 4, 6, 8} or mode not in {"kmeans", "uniform"}:
        raise ValueError("Invalid palette bits or mode")
    if granularity not in {"per_tensor", "per_grouped_channel"} or group_size < 1:
        raise ValueError("Invalid granularity/group size")
    units = getattr(ct.ComputeUnit, validate_compute_units(compute_units))
    receipt = begin_derivative(model_dir, output_dir, {"compute_units": compute_units, "nbits": nbits, "mode": mode, "granularity": granularity, "group_size": group_size, "per_channel_scale": per_channel_scale})

    baseline = _dir_size_mb(encoder_path)
    model = ct.models.MLModel(str(encoder_path), compute_units=units)
    # per_grouped_channel LUT ops are iOS18+ (spec version 9). The exports were
    # cut at iOS17; the app requires iOS 18.2, so raising the spec is free.
    spec = model.get_spec()
    if spec.specificationVersion < 9:
        spec.specificationVersion = 9
        model = ct.models.MLModel(
            spec, weights_dir=model.weights_dir, compute_units=units
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
        "decoder-joint-receipt.json",
    ]:
        src = model_dir / name
        dst = output_dir / name
        if src.exists() and not dst.exists():
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
    palettized = ct.models.MLModel(str(out_encoder), compute_units=units)
    finish_derivative(model_dir, output_dir, receipt, palettized)
    typer.echo("Done!")


if __name__ == "__main__":
    app()
