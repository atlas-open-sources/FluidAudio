#!/usr/bin/env python3
"""Paired utterance bootstrap; source-parity requires identical model hashes."""
import argparse
import json
from pathlib import Path
import random
from run import digest, validate_rows


def load_receipt(root):
    receipt = json.loads((root / 'receipt.json').read_text())
    if receipt.get('schema') != 1:
        return receipt
    audit_path = root / 'audit.json'
    if not audit_path.is_file():
        raise ValueError('schema1 receipt requires separate legacy audit sidecar')
    audit = json.loads(audit_path.read_text())
    if audit.get('legacy_audit', {}).get('receipt_sha256') != digest(root / 'receipt.json'):
        raise ValueError('legacy receipt changed after audit')
    for key, value in receipt.items():
        if key != 'schema' and audit.get(key) != value:
            raise ValueError(f'legacy audit contradicts original receipt: {key}')
    audit['audit_sha256'] = digest(audit_path)
    return audit


def weighted_delta(pairs):
    numerator_a = sum(a['selected_errors'] for a, b in pairs)
    numerator_b = sum(b['selected_errors'] for a, b in pairs)
    denominator = sum(a['selected_reference_units'] for a, b in pairs)
    if denominator <= 0:
        raise ValueError('empty comparison denominator')
    return (numerator_b - numerator_a) / denominator


def compare(a, b, allow_model_change=False, repetitions=2000):
    if a.resolve() == b.resolve():
        raise ValueError('self-comparison is not evidence')
    if repetitions < 100:
        raise ValueError('at least 100 bootstrap repetitions required')
    ra = load_receipt(a)
    rb = load_receipt(b)
    for root, r in ((a, ra), (b, rb)):
        if r.get('samples_sha256') != digest(root / 'samples.jsonl'):
            raise ValueError('missing or mismatched sample receipt hash; historical runs require separate audit')
        if r['status'] != 'passed':
            raise ValueError('cannot compare incomplete/failed runs')
    for key in ('corpus_sha256', 'prompt', 'compute_units', 'instrumentation', 'build', 'resolved_prompts'):
        if ra[key] != rb[key]:
            raise ValueError(f'non-comparable {key}')
    if ra['model_sha256'] != rb['model_sha256'] and not allow_model_change:
        raise ValueError('source-parity requires identical models')
    def load(root):
        rows = [json.loads(line) for line in (root / 'samples.jsonl').read_text().splitlines()]
        validate_rows([{'language': r['language'], 'sample_id': r['sampleId'], 'reference': r['ref_raw']} for r in rows], rows)
        result = {(r['language'], r['sampleId']): r for r in rows}
        if len(result) != len(rows) or not rows:
            raise ValueError('duplicate or empty samples')
        return result
    if all(ra.get(key) == rb.get(key) for key in ('revision', 'instrumentation', 'binary_sha256', 'model_sha256', 'prompt')):
        raise ValueError('same source, binary, model and prompt; repeated runs are timing controls, not source comparison')
    aa, bb = load(a), load(b)
    if set(aa) != set(bb):
        raise ValueError('sample sets differ')
    groups = {}
    for key, left in aa.items():
        right = bb[key]
        for field in ('ref_raw', 'selected_metric', 'selected_reference_units', 'audio_duration'):
            if left[field] != right[field]:
                raise ValueError(f'non-comparable scoring at {key}: {field}')
        groups.setdefault(key[0], []).append((left, right))
    timing_reasons = []
    if not ra.get('host_id') or ra.get('host_id') != rb.get('host_id'):
        timing_reasons.append('host identity absent or mismatched')
    if any(ra.get(key) != rb.get(key) for key in ('platform', 'machine')):
        timing_reasons.append('host platform mismatch')
    if any(r.get('timing_control') != 'declared-sequential' for r in (ra, rb)):
        timing_reasons.append('concurrency uncontrolled')
    def provenance(root, receipt):
        return {'run_directory': str(root.resolve()), 'receipt_sha256': digest(root / 'receipt.json'), **receipt}
    result = {'baseline': provenance(a, ra), 'candidate': provenance(b, rb),
              'timing': {'status': 'exploratory' if timing_reasons else 'operator-declared paired conditions',
                         'reasons': timing_reasons, 'repeatability_verified': False},
              'comparison': 'model-variant' if allow_model_change else 'source-parity',
              'uncertainty': 'paired utterance bootstrap, not speaker bootstrap; no speaker IDs in repackaged corpus',
              'bootstrap_seed': 20261021, 'repetitions': repetitions, 'languages': {}}
    rng = random.Random(20261021)
    for lang, pairs in sorted(groups.items()):
        n = len(pairs)
        bootstrap = sorted(weighted_delta([pairs[rng.randrange(n)] for _ in range(n)])
                           for _ in range(repetitions))
        result['languages'][lang] = {
            'metric': pairs[0][0]['selected_metric'], 'samples': n,
            'candidate_minus_baseline': weighted_delta(pairs),
            'ci95': [bootstrap[int(repetitions * .025)], bootstrap[int(repetitions * .975)]],
            'identical_hypotheses': sum(x['hyp_raw'] == y['hyp_raw'] for x, y in pairs),
            'baseline_rtfx': sum(x['audio_duration'] for x, y in pairs) / sum(x['processing_seconds'] for x, y in pairs),
            'candidate_rtfx': sum(y['audio_duration'] for x, y in pairs) / sum(y['processing_seconds'] for x, y in pairs)}
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('baseline', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('--allow-model-change', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.write_text(json.dumps(compare(args.baseline, args.candidate, args.allow_model_change), indent=2) + '\n')
