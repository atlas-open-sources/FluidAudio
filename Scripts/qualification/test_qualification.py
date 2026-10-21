import copy
import unittest
from run import validate_rows
from compare import weighted_delta


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


if __name__ == '__main__':
    unittest.main()
