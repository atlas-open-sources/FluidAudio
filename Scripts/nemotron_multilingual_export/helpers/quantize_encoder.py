#!/usr/bin/env python3
"""Quantize Nemotron encoder to int8 for smaller model size.

Quality must be measured on a matched parent and held-out real speech.
"""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qualification_provenance import begin_derivative, finish_derivative, validate_compute_units
import shutil

import typer
import coremltools as ct
from coremltools.optimize.coreml import (
    OptimizationConfig,
    OpLinearQuantizerConfig,
    linear_quantize_weights,
)

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def _dir_size_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)


@app.command()
def quantize(
    model_dir: Path = typer.Option(
        Path("nemotron_coreml"),
        "--model-dir",
        help="Directory containing the baseline encoder.mlpackage",
    ),
    output_dir: Path = typer.Option(
        Path("nemotron_coreml_int8"),
        "--output-dir",
        help="Output directory for quantized models",
    ),
    granularity: str = typer.Option(
        "per_channel",
        "--granularity",
        help="Quantization granularity: per_channel or per_tensor",
    ),
    parent_precision: str = typer.Option("FLOAT16", help="Required precision of the qualified parent: FLOAT16 or FLOAT32"),
    compute_units: str = typer.Option("CPU_ONLY", help="Explicit compute units for parent and transformed model validation"),
) -> None:
    """Quantize the Nemotron encoder to int8."""

    encoder_path = model_dir / "encoder.mlpackage"
    if not encoder_path.exists():
        typer.echo(f"Error: encoder.mlpackage not found at {encoder_path}")
        raise typer.Exit(code=1)

    if granularity not in {"per_channel", "per_tensor"}:
        raise ValueError("Invalid quantization granularity")
    units = getattr(ct.ComputeUnit, validate_compute_units(compute_units))
    receipt = begin_derivative(model_dir, output_dir, {"parent_precision": parent_precision, "weight_storage": "int8 weights, inherited floating compute", "compute_units": compute_units, "method": "int8_linear", "granularity": granularity})

    # Get baseline size
    baseline_size = _dir_size_mb(encoder_path)
    typer.echo(f"Baseline encoder size: {baseline_size:.1f} MB")

    # Load encoder
    typer.echo("Loading encoder model...")
    encoder = ct.models.MLModel(str(encoder_path), compute_units=units)

    # Set minimum deployment target for proper quantization ops
    try:
        encoder.minimum_deployment_target = ct.target.iOS17
    except Exception:
        pass

    # Configure int8 quantization
    typer.echo(f"Quantizing to int8 ({granularity})...")
    config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(
            mode="linear",
            granularity=granularity,
        )
    )

    # Quantize
    encoder_q = linear_quantize_weights(encoder, config)

    # Save quantized encoder
    encoder_q_path = output_dir / "encoder.mlpackage"
    typer.echo(f"Saving quantized encoder to {encoder_q_path}...")
    encoder_q.short_description = "Nemotron Streaming Encoder (int8 quantized)"
    encoder_q.author = "Fluid Inference"
    encoder_q.save(str(encoder_q_path))

    # Get quantized size
    quantized_size = _dir_size_mb(encoder_q_path)
    compression_ratio = baseline_size / quantized_size

    typer.echo(f"\nQuantized encoder size: {quantized_size:.1f} MB")
    typer.echo(f"Compression ratio: {compression_ratio:.2f}x")
    typer.echo(f"Size reduction: {baseline_size - quantized_size:.1f} MB")

    # Copy other models (they're already small enough)
    typer.echo("\nCopying other models...")
    for model_name in ["preprocessor.mlpackage", "decoder.mlpackage", "joint.mlpackage", "decoder_joint.mlpackage"]:
        src = model_dir / model_name
        dst = output_dir / model_name
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            typer.echo(f"  Copied {model_name}")

    # Copy metadata and tokenizer
    for file_name in ["metadata.json", "tokenizer.json", "decoder-joint-receipt.json"]:
        src = model_dir / file_name
        dst = output_dir / file_name
        if src.exists():
            shutil.copy2(src, dst)
            typer.echo(f"  Copied {file_name}")

    # Update metadata with quantization info
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        metadata["quantization"] = {
            "encoder": {
                "method": "int8_linear",
                "granularity": granularity,
                "baseline_size_mb": round(baseline_size, 1),
                "quantized_size_mb": round(quantized_size, 1),
                "compression_ratio": round(compression_ratio, 2),
            }
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        typer.echo("  Updated metadata.json with quantization info")

    encoder_q = ct.models.MLModel(str(encoder_q_path), compute_units=units)
    finish_derivative(model_dir, output_dir, receipt, encoder_q)
    typer.echo(f"\nDone! Quantized models saved to {output_dir}")

    # Print summary
    typer.echo("\n" + "=" * 50)
    typer.echo("SUMMARY")
    typer.echo("=" * 50)
    typer.echo(f"Encoder: {baseline_size:.1f} MB → {quantized_size:.1f} MB ({compression_ratio:.2f}x)")

    total_baseline = baseline_size
    total_quantized = quantized_size
    for model_name in ["preprocessor.mlpackage", "decoder.mlpackage", "joint.mlpackage", "decoder_joint.mlpackage"]:
        src = model_dir / model_name
        if src.exists():
            size = _dir_size_mb(src)
            total_baseline += size
            total_quantized += size

    typer.echo(f"Total pipeline: {total_baseline:.1f} MB → {total_quantized:.1f} MB")


if __name__ == "__main__":
    app()
