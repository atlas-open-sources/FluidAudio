"""Integrity tests use text bytes; no model or audio fixtures are fabricated."""
import hashlib
import unittest

from corpus import sha, verify_object


class CorpusIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.data = b'example transcript\n'
        self.item = {'path': 'en_us/en_us.trans.txt', 'size': len(self.data)}

    def git_item(self):
        header = f'blob {len(self.data)}\0'.encode('ascii')
        return {**self.item, 'oid': hashlib.sha1(header + self.data).hexdigest()}

    def lfs_item(self):
        return {**self.item, 'lfs': {'oid': sha(self.data)}}

    def test_valid_git_blob(self):
        self.assertEqual(verify_object(self.data, self.git_item()), sha(self.data))

    def test_raw_payload_sha1_is_not_git_object_hash(self):
        item = {**self.item, 'oid': hashlib.sha1(self.data).hexdigest()}
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            verify_object(self.data, item)

    def test_valid_lfs_blob(self):
        self.assertEqual(verify_object(self.data, self.lfs_item()), sha(self.data))

    def test_same_size_corruption_rejected_for_both_storage_modes(self):
        corrupted = b'X' + self.data[1:]
        for item in (self.git_item(), self.lfs_item()):
            with self.subTest(item=item), self.assertRaisesRegex(ValueError, 'hash mismatch'):
                verify_object(corrupted, item)

    def test_length_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, 'length mismatch'):
            verify_object(self.data + b'x', self.lfs_item())

    def test_missing_or_malformed_hash_rejected(self):
        for extra in ({}, {'oid': 'abc'}, {'lfs': {}}, {'lfs': {'oid': 'abc'}}):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, 'missing or invalid'):
                verify_object(self.data, {**self.item, **extra})


if __name__ == '__main__':
    unittest.main()
