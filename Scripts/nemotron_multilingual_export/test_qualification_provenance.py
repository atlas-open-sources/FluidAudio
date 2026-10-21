import hashlib
from pathlib import Path
import unittest
from qualification_provenance import begin_export, sha256, measure_difference, finish_export
import numpy as np


class ProvenanceTests(unittest.TestCase):
    def test_hash_matches_standard_library(self):
        source = Path(__file__)
        self.assertEqual(sha256(source), hashlib.sha256(source.read_bytes()).hexdigest())

    def test_existing_output_rejected_before_loading_checkpoint(self):
        with self.assertRaisesRegex(ValueError, 'already exists'):
            begin_export(Path(__file__).parent, Path('absent.nemo'), 'invalid', Path('absent.wav'), 42)

    def test_invalid_checkpoint_rejected_without_creating_output(self):
        output = Path(__file__).parent / 'must-not-be-created'
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            begin_export(output, Path(__file__), 'invalid', Path(__file__), 42)
        self.assertFalse(output.exists())


class NumericalGateTests(unittest.TestCase):
    def test_missing_components_cannot_pass_vacuously(self):
        with self.assertRaisesRegex(ValueError, "All four component checks"):
            finish_export(Path(__file__).parent, {}, {}, {})

    def test_integer_lengths_must_match_exactly(self):
        result = measure_difference(np.array([100]), np.array([101]))
        self.assertFalse(result['passed'])

    def test_small_float_rounding_is_accepted(self):
        result = measure_difference(np.array([1.0, 2.0]), np.array([1.001, 2.001]))
        self.assertTrue(result['passed'])
        self.assertGreater(result['rmse'], 0)

    def test_large_float_drift_is_rejected(self):
        self.assertFalse(measure_difference(np.array([1.0]), np.array([1.5]))['passed'])

    def test_nonfinite_reference_or_output_is_rejected(self):
        for reference, result in [(np.array([np.nan]), np.array([1.])), (np.array([1.]), np.array([np.inf]))]:
            with self.assertRaisesRegex(ValueError, 'Non-finite'):
                measure_difference(reference, result)

    def test_empty_or_mismatched_shape_is_rejected(self):
        for reference, result in [(np.array([]), np.array([])), (np.zeros(2), np.zeros(3))]:
            with self.assertRaisesRegex(ValueError, 'shape mismatch or empty'):
                measure_difference(reference, result)


if __name__ == '__main__':
    unittest.main()
