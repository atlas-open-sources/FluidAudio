#!/usr/bin/env python3
"""Paired utterance bootstrap; source-parity requires identical model hashes."""
import argparse
import json
from pathlib import Path
import random


def weighted_delta(pairs):
    numerator_a = sum(a['selected_errors'] for a, b in pairs)
    numerator_b = sum(b['selected_errors'] for a, b in pairs)
    denominator = sum(a['selected_reference_units'] for a, b in pairs)
    if denominator <= 0:
        raise ValueError('empty comparison denominator')
    return (numerator_b - numerator_a) / denominator


def compare(a, b, allow_model_change=False, repetitions=2000):
    if repetitions < 100:
        raise ValueError('at least 100 bootstrap repetitions required')
    ra = json.loads((a / 'receipt.json').read_text())
    rb = json.loads((b / 'receipt.json').read_text())
    for r in (ra, rb):
        if r['status'] != 'passed':
            raise ValueError('cannot compare incomplete/failed runs')
    for key in ('corpus_sha256', 'prompt', 'compute_units', 'instrumentation', 'build'):
        if ra[key] != rb[key]:
            raise ValueError(f'non-comparable {key}')
    if ra['model_sha256'] != rb['model_sha256'] and not allow_model_change:
        raise ValueError('source-parity requires identical models')
    def load(root):
        rows = [json.loads(line) for line in (root / 'samples.jsonl').read_text().splitlines()]
        result = {(r['language'], r['sampleId']): r for r in rows}
        if len(result) != len(rows) or not rows:
            raise ValueError('duplicate or empty samples')
        return result
    aa, bb = load(a), load(b)
    if set(aa) != set(bb):
        raise ValueError('sample sets differ')
    groups = {}
    for key, left in aa.items():
        right = bb[key]
        for field in ('ref_raw', 'selected_metric', 'selected_reference_units'):
            if left[field] != right[field]:
                raise ValueError(f'non-comparable scoring at {key}: {field}')
        groups.setdefault(key[0], []).append((left, right))
    result = {'comparison': 'model-variant' if allow_model_change else 'source-parity',
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
