import unittest
import numpy as np
from verify_streaming_rollout import prepend_history, next_prediction_state, validated_frame_count


class StreamingConventionTests(unittest.TestCase):
    def test_first_chunk_left_history_is_zero_and_audio_is_not_shifted_away(self):
        mel = np.arange(6, dtype=np.float32).reshape(1, 1, 6)
        out, history = prepend_history(mel, None, 5, 2)
        np.testing.assert_array_equal(out, [[[0, 0, 0, 1, 2, 3, 4]]])
        np.testing.assert_array_equal(history, [[[3, 4]]])

    def test_warmed_history_excludes_unconsumed_preprocessor_frame(self):
        out, history = prepend_history(np.array([[[6, 7, 8, 9, 10, 11]]], np.float32),
                                       np.array([[[4, 5]]], np.float32), 5, 2)
        np.testing.assert_array_equal(out, [[[4, 5, 6, 7, 8, 9, 10]]])
        np.testing.assert_array_equal(history, [[[9, 10]]])

    def test_short_chunk_retains_previous_history_tail(self):
        previous = np.arange(9, dtype=np.float32).reshape(1, 1, 9)
        mel = np.arange(9, 18, dtype=np.float32).reshape(1, 1, 9)
        _, history = prepend_history(mel, previous, 8, 9)
        np.testing.assert_array_equal(history, np.arange(8, 17).reshape(1, 1, 9))
        _, first = prepend_history(mel, None, 8, 9)
        np.testing.assert_array_equal(first, [[[0, 9, 10, 11, 12, 13, 14, 15, 16]]])

    def test_bad_history_shape_rejected(self):
        with self.assertRaisesRegex(ValueError, 'History shape'):
            prepend_history(np.zeros((1, 2, 6)), np.zeros((1, 1, 2)), 5, 2)

    def test_blank_preserves_recurrent_state(self):
        previous = (7, np.array([1]), np.array([2]))
        self.assertIs(next_prediction_state(9, np.array([3]), np.array([4]), previous, 9), previous)
        updated = next_prediction_state(8, np.array([3]), np.array([4]), previous, 9)
        self.assertEqual(updated[0], 8)
        np.testing.assert_array_equal(updated[1], [3])

    def test_frame_count_must_be_in_tensor(self):
        for length in [0, 5, -1]:
            with self.assertRaisesRegex(ValueError, 'encoded length'):
                validated_frame_count(np.zeros((1, 10, 4)), np.array([length]))
        self.assertEqual(validated_frame_count(np.zeros((1, 10, 4)), np.array([4])), 4)
