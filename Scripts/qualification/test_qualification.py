import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from run import validate_rows, digest, resolve_prompts
from compare import weighted_delta, compare, load_receipt


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.samples = [{'language': 'en_us', 'sample_id': 'one', 'reference': 'real reference'}]
        self.rows = [{'language': 'en_us', 'sampleId': 'one', 'ref_raw': 'real reference', 'status': 'ok',
                      'selected_metric': 'wer', 'selected_errors': 0, 'selected_reference_units': 2,
                      'processing_seconds': 1.0, 'audio_duration': 2.0}]

    def test_exact_rows_pass(self):
        validate_rows(self.samples, self.rows)

    def test_missing_duplicate_unexpected_fail(self):
        for rows in ([], self.rows * 2, [dict(self.rows[0], sampleId='other')]):
            with self.assertRaises(ValueError):
                validate_rows(self.samples, rows)

    def test_reference_status_counts_timing_fail(self):
        for key, value in [('ref_raw', 'wrong'), ('status', 'skipped'), ('selected_errors', -1),
                           ('selected_reference_units', 0), ('processing_seconds', float('nan')),
                           ('audio_duration', 0), ('selected_metric', 'accuracy')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_rows(self.samples, [dict(self.rows[0], **{key: value})])

    def test_weighted_delta_not_mean_of_rates(self):
        a = dict(self.rows[0], selected_errors=1, selected_reference_units=1)
        b = dict(a, selected_errors=0)
        c = dict(a, selected_errors=0, selected_reference_units=9)
        self.assertEqual(weighted_delta([(a, b), (c, c)]), -0.1)

    def test_duplicate_manifest_fails(self):
        with self.assertRaises(ValueError):
            validate_rows(self.samples * 2, self.rows)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        scratch = os.environ.get('AGENT_SCRATCH_DIR')
        if not scratch:
            self.fail('AGENT_SCRATCH_DIR must name a project-owned scratch directory')
        self.directory = tempfile.TemporaryDirectory(dir=scratch, prefix='qualification-tests-')
        self.addCleanup(self.directory.cleanup)
        self.a = Path(self.directory.name) / 'baseline'
        self.b = Path(self.directory.name) / 'candidate'
        for index, root in enumerate((self.a, self.b)):
            root.mkdir()
            row = {'language': 'en_us', 'sampleId': 'one', 'ref_raw': 'test reference', 'hyp_raw': 'test reference',
                   'status': 'ok', 'selected_metric': 'wer', 'selected_errors': 0, 'selected_reference_units': 2,
                   'processing_seconds': 1.0, 'audio_duration': 2.0}
            (root / 'samples.jsonl').write_text(json.dumps(row) + '\n')
            receipt = {'status': 'passed', 'revision': str(index), 'binary_sha256': str(index),
                       'model_sha256': 'model', 'corpus_sha256': 'corpus', 'prompt': 'known',
                       'compute_units': 'ane', 'instrumentation': 'same', 'build': 'release',
                       'resolved_prompts': {'en_us': {'id': 0}}, 'platform': 'same', 'machine': 'arm64',
                       'samples_sha256': digest(root / 'samples.jsonl')}
            (root / 'receipt.json').write_text(json.dumps(receipt))

    def update_receipt(self, root, **changes):
        path = root / 'receipt.json'
        receipt = json.loads(path.read_text())
        receipt.update(changes)
        path.write_text(json.dumps(receipt))

    def test_valid_comparison_includes_provenance_and_exploratory_timing(self):
        result = compare(self.a, self.b, repetitions=100)
        self.assertEqual(result['baseline']['revision'], '0')
        self.assertEqual(result['candidate']['revision'], '1')
        self.assertEqual(result['timing']['status'], 'exploratory')
        self.assertEqual(result['languages']['en_us']['candidate_minus_baseline'], 0)

    def test_rejects_tampered_samples_and_missing_hash(self):
        with (self.b / 'samples.jsonl').open('a') as f:
            f.write('\n')
        with self.assertRaisesRegex(ValueError, 'sample receipt hash'):
            compare(self.a, self.b, repetitions=100)
        self.update_receipt(self.b, samples_sha256=None)
        with self.assertRaises(ValueError):
            compare(self.a, self.b, repetitions=100)

    def test_rejects_self_and_identical_run_configuration(self):
        with self.assertRaisesRegex(ValueError, 'self-comparison'):
            compare(self.a, self.a, repetitions=100)
        self.update_receipt(self.b, revision='0', binary_sha256='0')
        with self.assertRaisesRegex(ValueError, 'repeated runs'):
            compare(self.a, self.b, repetitions=100)

    def test_model_change_requires_explicit_axis(self):
        self.update_receipt(self.b, model_sha256='different')
        with self.assertRaisesRegex(ValueError, 'identical models'):
            compare(self.a, self.b, repetitions=100)
        self.assertEqual(compare(self.a, self.b, True, 100)['comparison'], 'model-variant')

    def test_failed_runs_and_mismatched_prompt_mapping_rejected(self):
        self.update_receipt(self.b, status='failed')
        with self.assertRaises(ValueError):
            compare(self.a, self.b, repetitions=100)
        self.update_receipt(self.b, status='passed', resolved_prompts={'en_us': {'id': 1}})
        with self.assertRaisesRegex(ValueError, 'resolved_prompts'):
            compare(self.a, self.b, repetitions=100)

    def test_legacy_sidecar_binds_original_receipt_and_preserves_claims(self):
        self.update_receipt(self.a, schema=1)
        with self.assertRaisesRegex(ValueError, 'legacy audit'):
            load_receipt(self.a)
        path = self.a / 'receipt.json'
        original = json.loads(path.read_text())
        audited = dict(original, schema=2, legacy_audit={'receipt_sha256': digest(path)})
        (self.a / 'audit.json').write_text(json.dumps(audited))
        self.assertEqual(load_receipt(self.a)['revision'], '0')
        audited['revision'] = 'changed'
        (self.a / 'audit.json').write_text(json.dumps(audited))
        with self.assertRaisesRegex(ValueError, 'contradicts'):
            load_receipt(self.a)
        self.update_receipt(self.a, revision='later')
        with self.assertRaisesRegex(ValueError, 'changed after audit'):
            load_receipt(self.a)

    def test_host_mismatch_remains_exploratory_with_declared_sequential_runs(self):
        self.update_receipt(self.a, host_id='host-a', timing_control='declared-sequential')
        self.update_receipt(self.b, host_id='host-b', timing_control='declared-sequential')
        result = compare(self.a, self.b, repetitions=100)
        self.assertEqual(result['timing']['status'], 'exploratory')
        self.assertIn('host identity absent or mismatched', result['timing']['reasons'])

    def test_invalid_counts_rejected_even_when_sample_hash_matches(self):
        path = self.b / 'samples.jsonl'
        row = json.loads(path.read_text())
        row['selected_errors'] = -1
        path.write_text(json.dumps(row) + '\n')
        self.update_receipt(self.b, samples_sha256=digest(path))
        with self.assertRaisesRegex(ValueError, 'selected_errors'):
            compare(self.a, self.b, repetitions=100)

    def test_prompt_mapping_matches_swift_special_cases_and_bare_fallback(self):
        metadata = {'prompt_dictionary': {'zh-CN': 4, 'es-ES': 2, 'pt-BR': 12, 'ar': 7, 'en-US': 0}}
        rows = resolve_prompts(['cmn_hans_cn', 'es_419', 'pt_br', 'ar_eg', 'en_us'], metadata, 'known')
        self.assertEqual([x['id'] for x in rows.values()], [4, 2, 12, 7, 0])
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            resolve_prompts(['ja_jp'], metadata, 'known')


if __name__ == '__main__':
    unittest.main()
