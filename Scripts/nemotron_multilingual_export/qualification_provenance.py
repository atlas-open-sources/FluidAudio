"""Fail-closed provenance and real-input component checks for local exports."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import os
import platform


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_provenance():
    source = Path(__file__).resolve().parent
    git = ['git', '-C', str(source)]
    return {
        'git_commit': subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip(),
        'git_dirty': bool(subprocess.check_output(git + ['status', '--porcelain'], text=True).strip()),
        'argv': list(sys.argv), 'python': sys.version,
        'platform': platform.platform(), 'macos': platform.mac_ver()[0],
        'machine': platform.machine(),
        'hardware': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
        'offline_environment': {key: os.environ.get(key) for key in
                                ['HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'WANDB_MODE']},
    }


def validate_compute_units(name):
    if name not in {'CPU_ONLY', 'CPU_AND_NE', 'CPU_AND_GPU', 'ALL'}:
        raise ValueError('Invalid compute units')
    return name


def begin_export(output, checkpoint, expected_hash, audio, seed):
    if output.exists():
        raise ValueError('Output already exists; use a new isolated directory')
    if not checkpoint.is_file() or sha256(checkpoint) != expected_hash:
        raise ValueError('Checkpoint missing or SHA256 mismatch')
    if not audio.is_file():
        raise ValueError('Real validation WAV is required')
    receipt = {
        'schema_version': 1, 'status': 'started',
        'checkpoint': {'path': str(checkpoint.resolve()), 'sha256': expected_hash},
        'validation_audio': {'path': str(audio.resolve()), 'sha256': sha256(audio)},
        'seed': seed, 'argv': sys.argv,
        **runtime_provenance(),
        'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        'source_hashes': {str(p.relative_to(Path(__file__).parent)): sha256(p) for p in Path(__file__).parent.rglob('*.py')},
    }
    source_receipt = checkpoint.parent / 'copy-receipt.json'
    if source_receipt.is_file():
        source = json.loads(source_receipt.read_text())
        if source.get('sha256') != expected_hash or source.get('verified') is not True:
            raise ValueError('Checkpoint source-copy receipt mismatch')
        receipt['checkpoint']['source_receipt'] = source
        receipt['checkpoint']['source_receipt_sha256'] = sha256(source_receipt)
    output.mkdir(parents=True)
    (output / 'export-receipt.json').write_text(json.dumps(receipt, indent=2))
    return receipt


def compare_outputs(mlmodel, wrapper, args, inputs, outputs, capture=None):
    import numpy as np
    import torch
    with torch.no_grad():
        expected = wrapper(*args)
    if isinstance(expected, torch.Tensor):
        expected = (expected,)
    if len(expected) != len(outputs) or len(args) != len(inputs):
        raise ValueError('Component input/output count mismatch')
    input_arrays = {spec.name: value.detach().numpy() for spec, value in zip(inputs, args)}
    if capture is not None:
        np.savez(capture, **{'input_' + k: v for k, v in input_arrays.items()},
                 **{'expected_' + spec.name: t.detach().numpy() for spec, t in zip(outputs, expected)})
    actual = mlmodel.predict(input_arrays)
    result = {}
    for spec, tensor in zip(outputs, expected):
        ref = tensor.detach().numpy()
        got = actual[spec.name]
        result[spec.name] = measure_difference(ref, got)
    # Record every mismatch before raising in finish_export. This check measures
    # a single real-audio component input, not end-to-end transcription quality.
    return result


def finish_export(output, receipt, checks, options):
    if set(checks) != {'preprocessor', 'encoder', 'decoder', 'joint'} or any(not rows for rows in checks.values()):
        raise ValueError('All four component checks are required')
    passed = all(row['passed'] for component in checks.values() for row in component.values())
    receipt.update(status='passed' if passed else 'numerical_check_failed', checks=checks, options=options,
                   numerical_gate='Single real-audio input; floats normalized RMSE <=0.05; integers exact; not WER acceptance')
    receipt['output_sha256'] = {str(p.relative_to(output)): sha256(p) for p in output.rglob('*')
                                if p.is_file() and p.name != 'export-receipt.json'}
    (output / 'export-receipt.json').write_text(json.dumps(receipt, indent=2))
    if not passed:
        raise ValueError('Numerical component comparison failed; inspect export-receipt.json')


def begin_derivative(model_dir, output_dir, options):
    import numpy as np
    import random
    if output_dir.exists():
        raise ValueError('Output already exists; use a new isolated directory')
    parent_path = model_dir / 'export-receipt.json'
    validate_compute_units(options.get('compute_units'))
    parent = json.loads(parent_path.read_text())
    precision = options.get('parent_precision', 'FLOAT16')
    if precision not in {'FLOAT16', 'FLOAT32'}:
        raise ValueError('Invalid declared parent precision')
    if parent['status'] != 'passed' or parent['options']['precision'] != precision:
        raise ValueError('Derivative requires a passed parent with declared precision (default passed FP16)')
    required = ['encoder.mlpackage', 'preprocessor.mlpackage', 'decoder.mlpackage', 'joint.mlpackage']
    hashes = parent.get('output_sha256', {})
    if not hashes or any(not any(k.startswith(name + '/') for k in hashes) for name in required):
        raise ValueError('Incomplete parent component hash manifest')
    if any(name not in hashes for name in ['metadata.json', 'tokenizer.json', 'encoder-validation.npz']):
        raise ValueError('Incomplete parent companion hash manifest')
    for relative, expected in hashes.items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('Unsafe parent hash path')
        if sha256(model_dir / relative) != expected:
            raise ValueError(f'Parent changed: {relative}')
    fused = model_dir / 'decoder_joint.mlpackage'
    if fused.exists():
        fused_receipt = json.loads((model_dir / 'decoder-joint-receipt.json').read_text())
        if fused_receipt['status'] != 'passed' or fused_receipt['checkpoint_sha256'] != parent['checkpoint']['sha256']:
            raise ValueError('Fused model lineage mismatch')
        for relative, expected in fused_receipt['output_hashes'].items():
            if sha256(fused / relative) != expected:
                raise ValueError('Fused model changed')
    np.random.seed(42)
    random.seed(42)
    output_dir.mkdir(parents=True)
    receipt = {'schema_version': 1, 'status': 'started', 'parent_receipt_sha256': sha256(parent_path), 'parent_directory': str(model_dir.resolve()),
               **runtime_provenance(),
               'source_hashes': {str(p.relative_to(Path(__file__).parent)): sha256(p) for p in Path(__file__).parent.rglob('*.py')},
               'checkpoint': parent['checkpoint'], 'options': options, 'seed': 42,
               'parent_precision': precision,
               'compute_precision': 'Inherited parent graph; weight compression does not lower activation/op precision',
               'weight_storage': options.get('weight_storage'),
               'parent_output_sha256': parent['output_sha256'],
               'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}
    (output_dir / 'derivative-receipt.json').write_text(json.dumps(receipt, indent=2))
    return receipt


def finish_derivative(model_dir, output_dir, receipt, mlmodel):
    import numpy as np
    with np.load(model_dir / 'encoder-validation.npz', allow_pickle=False) as arrays:
        inputs = {k[len('input_'):]: arrays[k] for k in arrays.files if k.startswith('input_')}
        actual = mlmodel.predict(inputs)
        checks = {}
        for key in arrays.files:
            if not key.startswith('expected_'):
                continue
            name = key[len('expected_'):]
            ref, got = arrays[key], actual[name]
            checks[name] = measure_difference(ref, got)
    if not checks:
        raise ValueError('Missing derivative numerical reference outputs')
    passed = all(v['passed'] for v in checks.values())
    metadata_path = output_dir / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata.update(checkpoint_sha256=receipt['checkpoint']['sha256'],
                    parent_precision=receipt['parent_precision'],
                    compute_precision=receipt['compute_precision'],
                    weight_storage=receipt['weight_storage'])
    metadata_path.write_text(json.dumps(metadata, indent=2))
    receipt.update(checks=checks, status='passed' if passed else 'numerical_check_failed')
    receipt['output_sha256'] = {str(p.relative_to(output_dir)): sha256(p) for p in output_dir.rglob('*')
                                if p.is_file() and p.name != 'derivative-receipt.json'}
    (output_dir / 'derivative-receipt.json').write_text(json.dumps(receipt, indent=2))
    if not passed:
        raise ValueError('Derivative numerical check failed; see retained receipt')


def measure_difference(ref, got):
    import numpy as np
    if ref.shape != got.shape or ref.size == 0:
        raise ValueError('Output shape mismatch or empty output')
    if not np.isfinite(got).all() or not np.isfinite(ref).all():
        raise ValueError('Non-finite numerical output')
    delta = got.astype(np.float64) - ref
    rmse = float(np.sqrt(np.mean(delta ** 2)))
    ref_rms = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2)))
    relative = rmse / max(ref_rms, 1e-8)
    exact_integer = bool(np.array_equal(got, ref)) if np.issubdtype(ref.dtype, np.integer) else None
    passed = exact_integer if exact_integer is not None else relative <= 0.05
    return {'shape': list(ref.shape), 'rmse': rmse, 'normalized_rmse': relative,
            'max_abs_error': float(np.max(np.abs(delta))), 'exact_integer': exact_integer,
            'passed': bool(passed)}
