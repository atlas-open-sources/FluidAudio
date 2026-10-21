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


class DerivativeGuardTests(unittest.TestCase):
    def setUp(self):
        import os
        import tempfile
        import json
        from qualification_provenance import sha256
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('AGENT_SCRATCH_DIR', Path(__file__).parent))
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name) / 'parent'
        self.parent.mkdir()
        self.output = Path(self.temp.name) / 'derived'
        for name in ['encoder', 'preprocessor', 'decoder', 'joint']:
            folder = self.parent / (name + '.mlpackage')
            folder.mkdir()
            (folder / 'model.bin').write_bytes(b'fixture-model')
        for name in ['metadata.json', 'tokenizer.json', 'encoder-validation.npz']:
            (self.parent / name).write_bytes(b'fixture-companion')
        self.receipt = {'status': 'passed', 'options': {'precision': 'FLOAT16'},
                        'checkpoint': {'sha256': 'fixture'},
                        'output_sha256': {str(p.relative_to(self.parent)): sha256(p) for p in self.parent.rglob('*') if p.is_file()}}
        self.write_receipt()

    def write_receipt(self):
        import json
        (self.parent / 'export-receipt.json').write_text(json.dumps(self.receipt))

    def begin(self):
        from qualification_provenance import begin_derivative
        return begin_derivative(self.parent, self.output, {'compute_units': 'CPU_ONLY'})

    def test_failed_parent_rejected(self):
        self.receipt['status'] = 'numerical_check_failed'
        self.write_receipt()
        with self.assertRaisesRegex(ValueError, 'passed FP16'):
            self.begin()
        self.assertFalse(self.output.exists())

    def test_fp32_parent_not_relabelled_fp16(self):
        self.receipt['options']['precision'] = 'FLOAT32'
        self.write_receipt()
        with self.assertRaisesRegex(ValueError, 'passed FP16'):
            self.begin()

    def test_changed_companion_rejected(self):
        (self.parent / 'tokenizer.json').write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'Parent changed'):
            self.begin()
        self.assertFalse(self.output.exists())

    def test_missing_component_hashes_rejected(self):
        self.receipt['output_sha256'].pop('joint.mlpackage/model.bin')
        self.write_receipt()
        with self.assertRaisesRegex(ValueError, 'Incomplete parent component'):
            self.begin()

    def test_unverified_fused_model_rejected(self):
        (self.parent / 'decoder_joint.mlpackage').mkdir()
        with self.assertRaises(FileNotFoundError):
            self.begin()
        self.assertFalse(self.output.exists())

    def test_compute_units_required_and_validated(self):
        from qualification_provenance import begin_derivative
        with self.assertRaisesRegex(ValueError, 'Invalid compute units'):
            begin_derivative(self.parent, self.output, {'compute_units': 'MAGIC'})
        self.assertFalse(self.output.exists())

    def test_explicit_fp32_parent_preserves_precision_in_receipt(self):
        from qualification_provenance import begin_derivative
        self.receipt['options']['precision'] = 'FLOAT32'
        self.write_receipt()
        result = begin_derivative(self.parent, self.output, {'compute_units': 'CPU_ONLY', 'parent_precision': 'FLOAT32', 'weight_storage': 'int8'})
        self.assertEqual(result['parent_precision'], 'FLOAT32')
        self.assertEqual(result['weight_storage'], 'int8')
        self.assertIn('Inherited parent', result['compute_precision'])
