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


def resolve_prompts(languages, metadata, mode):
    dictionary = metadata.get('prompt_dictionary', {})
    special = {'cmn_hans_cn': 'zh-CN', 'es_419': 'es-ES', 'pt_br': 'pt-BR', 'ar_eg': 'ar-EG'}
    resolved = {}
    for language in languages:
        parts = language.split('_')
        prompt = 'auto' if mode == 'auto' else special.get(
            language, parts[0] + '-' + parts[1].upper() if len(parts) == 2 else language)
        dashed = prompt.replace('_', '-')
        pieces = dashed.split('-', 1)
        keys = [prompt, dashed]
        if len(pieces) == 2:
            keys.append(pieces[0].lower() + '-' + pieces[1].upper())
        keys.append(pieces[0].lower())
        matched = next((key for key in keys if key in dictionary), None)
        if matched is None:
            raise ValueError(f'unsupported {mode} prompt for {language}: {prompt}')
        prompt_id = dictionary[matched]
        if type(prompt_id) is not int or prompt_id < 0:
            raise ValueError(f'invalid prompt ID for {matched}')
        resolved[language] = {'requested': prompt, 'matched': matched, 'id': prompt_id}
    return resolved


def build_provenance(args, binary_hash):
    result = {'configuration': getattr(args, 'build_configuration', 'release'), 'verification': 'declared_unverified'}
    path = getattr(args, 'build_receipt', None)
    if path is not None:
        receipt = json.loads(path.read_text())
        for key, expected in [('binary_sha256', binary_hash), ('revision', args.revision),
                              ('instrumentation', args.instrumentation), ('configuration', result['configuration'])]:
            if receipt.get(key) != expected:
                raise ValueError(f'build receipt mismatch: {key}')
        result.update(verification='receipt_matched', receipt_path=str(path.resolve()), receipt_sha256=digest(path))
    return result


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
    if not samples:
        raise ValueError('empty corpus')
    resolved_prompts = resolve_prompts(languages, json.loads((args.model / 'metadata.json').read_text()), args.prompt)
    binary_hash = digest(args.binary)
    build = build_provenance(args, binary_hash)
    args.output.mkdir(parents=True, exist_ok=False)
    model_hash = tree_digest(args.model.resolve())
    rows_file = args.output / 'samples.jsonl'
    cmd = [str(args.binary.resolve()), 'nemotron-multilingual-benchmark', '--model-dir', str(args.model.resolve()),
           '--cache-dir', str(corpus), '--languages', ','.join(languages), '--samples', 'all',
           '--dump-samples', str(rows_file.resolve()), '--output', str((args.output / 'summary.json').resolve()),
           '--compute-units', 'cpuAndNeuralEngine']
    if args.prompt == 'auto':
        cmd += ['--prompt', 'auto']
    receipt = {'schema': 2, 'revision': args.revision, 'instrumentation': args.instrumentation,
               'build': build['configuration'], 'build_provenance': build, 'binary_sha256': binary_hash, 'model_sha256': model_hash,
               'corpus_sha256': digest(manifest_path), 'platform': platform.platform(), 'machine': platform.machine(),
               'resolved_prompts': resolved_prompts, 'timing_control': getattr(args, 'timing_control', 'uncontrolled'),
               'host_id': getattr(args, 'host_id', None),
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
        if digest(args.binary) != binary_hash:
            raise ValueError('binary changed during run')
        receipt.update(status='passed', observed_samples=len(rows), samples_sha256=digest(rows_file))
    except Exception as error:
        receipt['error'] = str(error)
        raise
    finally:
        (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


def audit_existing(root, corpus, model, binary=None):
    """Create a later attestation; never rewrite or upgrade the historical receipt."""
    original_path = root / 'receipt.json'
    original = json.loads(original_path.read_text())
    if original.get('schema') != 1 or original.get('status') != 'passed':
        raise ValueError('legacy audit requires a passed schema1 receipt')
    target = root / 'audit.json'
    if target.exists():
        raise ValueError('audit already exists; preserve prior attestation')
    manifest_path = corpus / 'manifest.json'
    if digest(manifest_path) != original['corpus_sha256'] or tree_digest(model) != original['model_sha256']:
        raise ValueError('legacy model/corpus differs from captured original hashes')
    samples = json.loads(manifest_path.read_text())['samples']
    for sample in samples:
        path = (corpus / sample['path']).resolve()
        if not path.is_relative_to(corpus.resolve()) or digest(path) != sample['sha256']:
            raise ValueError('legacy corpus audio hash/path mismatch')
    rows_path = root / 'samples.jsonl'
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    validate_rows(samples, rows)
    if len(rows) != original.get('observed_samples') or len(samples) != original.get('expected_samples'):
        raise ValueError('legacy sample count mismatch')
    prompts = resolve_prompts(list(dict.fromkeys(s['language'] for s in samples)),
                              json.loads((model / 'metadata.json').read_text()), original['prompt'])
    current_binary = digest(binary) if binary is not None and binary.is_file() else None
    attestation = dict(original, schema=2, samples_sha256=digest(rows_path), resolved_prompts=prompts,
                       build_provenance={'configuration': original['build'], 'verification': 'declared_unverified'},
                       timing_control='uncontrolled', host_id=None,
                       legacy_audit={'receipt_sha256': digest(original_path),
                                     'audited_at_unix': time.time(),
                                     'corpus_path': str(corpus.resolve()), 'model_path': str(model.resolve()),
                                     'binary_current_sha256': current_binary,
                                     'binary_verification': 'matches_original' if current_binary == original['binary_sha256']
                                         else 'original_hash_recorded_only',
                                     'sample_integrity_scope': 'validated now; original receipt did not hash sample output'})
    target.write_text(json.dumps(attestation, indent=2) + '\n')
    return attestation


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for option in ('binary', 'corpus', 'model', 'output'):
        p.add_argument('--' + option, type=Path)
    p.add_argument('--revision')
    p.add_argument('--instrumentation')
    p.add_argument('--prompt', choices=('known', 'auto'), default='known')
    p.add_argument('--build-configuration', choices=('release', 'debug'), default='release')
    p.add_argument('--build-receipt', type=Path)
    p.add_argument('--host-id', help='Operator-defined identifier for paired timing host')
    p.add_argument('--timing-control', choices=('uncontrolled', 'declared-sequential'), default='uncontrolled')
    p.add_argument('--timeout', type=int, default=3600)
    p.add_argument('--audit-existing', type=Path, help='Write immutable legacy audit sidecar, not a new run')
    args = p.parse_args()
    if args.audit_existing is not None:
        if args.corpus is None or args.model is None:
            p.error('--audit-existing requires --corpus and --model')
        print(json.dumps(audit_existing(args.audit_existing, args.corpus, args.model, args.binary), indent=2))
    else:
        if any(getattr(args, name) is None for name in ('binary', 'corpus', 'model', 'output', 'revision', 'instrumentation')):
            p.error('run requires --binary, --corpus, --model, --output, --revision and --instrumentation')
        run(args)
