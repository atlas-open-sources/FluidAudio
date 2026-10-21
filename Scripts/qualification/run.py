#!/usr/bin/env python3
"""Fail-closed, provenance-locked wrapper for the real FluidAudio benchmark CLI."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tree_digest(root):
    rows = [(str(p.relative_to(root)), digest(p)) for p in sorted(root.rglob('*')) if p.is_file()]
    if not rows:
        raise ValueError('empty model directory')
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def validate_rows(samples, rows):
    expected = {(s['language'], s['sample_id']): s for s in samples}
    if len(expected) != len(samples):
        raise ValueError('duplicate manifest IDs')
    seen = set()
    for row in rows:
        key = (row['language'], row['sampleId'])
        if key not in expected or key in seen:
            raise ValueError(f'unexpected or repeated sample: {key}')
        seen.add(key)
        if row['ref_raw'] != expected[key]['reference']:
            raise ValueError(f'reference mismatch: {key}')
        if row['status'] != 'ok' or row['selected_metric'] not in ('wer', 'cer'):
            raise ValueError(f'invalid sample status or metric: {key}')
        for field in ('processing_seconds', 'audio_duration'):
            value = row[field]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'invalid {field}: {key}')
        for field, minimum in (('selected_errors', 0), ('selected_reference_units', 1)):
            if type(row[field]) is not int or row[field] < minimum:
                raise ValueError(f'invalid {field}: {key}')
    if seen != set(expected):
        raise ValueError(f'missing {len(set(expected) - seen)} samples')


def run(args):
    corpus = args.corpus.resolve()
    manifest_path = corpus / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    samples = manifest['samples']
    languages = list(dict.fromkeys(s['language'] for s in samples))
    for sample in samples:
        path = (corpus / sample['path']).resolve()
        if not path.is_relative_to(corpus) or digest(path) != sample['sha256']:
            raise ValueError('corpus path or hash mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    model_hash = tree_digest(args.model.resolve())
    rows_file = args.output / 'samples.jsonl'
    cmd = [str(args.binary.resolve()), 'nemotron-multilingual-benchmark', '--model-dir', str(args.model.resolve()),
           '--cache-dir', str(corpus), '--languages', ','.join(languages), '--samples', 'all',
           '--dump-samples', str(rows_file.resolve()), '--output', str((args.output / 'summary.json').resolve()),
           '--compute-units', 'cpuAndNeuralEngine']
    if args.prompt == 'auto':
        cmd += ['--prompt', 'auto']
    receipt = {'schema': 1, 'revision': args.revision, 'instrumentation': args.instrumentation,
               'build': 'release', 'binary_sha256': digest(args.binary), 'model_sha256': model_hash,
               'corpus_sha256': digest(manifest_path), 'platform': platform.platform(), 'machine': platform.machine(),
               'prompt': args.prompt, 'compute_units': 'cpuAndNeuralEngine',
               'measurement': 'whole-utterance throughput; not real-time streaming latency',
               'expected_samples': len(samples), 'command': cmd, 'status': 'running'}
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    start = time.monotonic()
    with (args.output / 'run.log').open('w') as log:
        measured_cmd = ['/usr/bin/time', '-l'] + cmd if platform.system() == 'Darwin' else cmd
        process = subprocess.Popen(measured_cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            rc = 124
    receipt.update(exit_code=rc, wall_seconds=time.monotonic() - start, status='failed')
    rss = re.search(r'(\d+)\s+maximum resident set size', (args.output / 'run.log').read_text())
    receipt['peak_resident_bytes'] = int(rss.group(1)) if rss else None
    try:
        if rc:
            raise ValueError(f'benchmark exited {rc}')
        rows = [json.loads(line) for line in rows_file.read_text().splitlines()]
        validate_rows(samples, rows)
        if tree_digest(args.model.resolve()) != model_hash or digest(manifest_path) != receipt['corpus_sha256']:
            raise ValueError('inputs changed during run')
        # Loader must not silently add or rewrite any selected audio.
        for sample in samples:
            if digest(corpus / sample['path']) != sample['sha256']:
                raise ValueError('audio changed during run')
        receipt.update(status='passed', observed_samples=len(rows))
    except Exception as error:
        receipt['error'] = str(error)
        raise
    finally:
        (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for option in ('binary', 'corpus', 'model', 'output'):
        p.add_argument('--' + option, required=True, type=Path)
    p.add_argument('--revision', required=True)
    p.add_argument('--instrumentation', required=True)
    p.add_argument('--prompt', choices=('known', 'auto'), default='known')
    p.add_argument('--timeout', type=int, default=3600)
    run(p.parse_args())
