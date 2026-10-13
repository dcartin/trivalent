"""Regression tests for trivalent.states's intended public API.

These tests deliberately exercise boundary cases absent from typical small-state
examples. After each implementation change, run:
    python -m unittest discover -s tests -p 'test_states*.py' -v

Assumed public contract:
  * States are nonnegative integers fitting in 64 unsigned bits.
  * Ordinary Python integers and NumPy integer scalars are accepted.
  * 'num_edges' is a positive integer; orientation, parity states for |E| = 64,
    and combined states for |E| = 32 only with graph ID zero.
  * Signed permutations are 1-based, have abs values 1..E exactly once,
    and map old edge i to abs(perm[i])-1; negative sign flips orientation.
  * Empty/malformed symmetry groups and permutations are rejected.
"""

import unittest
import numpy as np
from trivalent import states

UINT64_MAX = (1 << 64) - 1


def reference_permute(state, perm, E, kind):
    """Independent Python-integer reference, no NumPy/Numba arithmetic."""
    state = int(state)
    perm = [int(x) for x in perm]
    assert sorted(abs(x) for x in perm) == list(range(1, E + 1))
    mask = (1 << E) - 1
    width = 2 * E if kind == 'combined' else E
    graph_id = state >> width
    if kind == 'combined':
        orient = (state >> E) & mask
        parity = state & mask
    else:
        orient = state & mask if kind == 'orient' else 0
        parity = state & mask if kind == 'parity' else 0
    out_o = out_p = 0
    for i, target in enumerate(perm):
        j = abs(target) - 1
        if kind in ('orient', 'combined'):
            bit = (orient >> i) & 1
            if target < 0:
                bit ^= 1
            out_o |= bit << j
        if kind in ('parity', 'combined'):
            out_p |= ((parity >> i) & 1) << j
    if kind == 'combined':
        return (graph_id << (2 * E)) | (out_o << E) | out_p
    return (graph_id << E) | (out_o if kind == 'orient' else out_p)


class TestStateBoundaries(unittest.TestCase):

    def test_single_encoders_preserve_bit_63(self):
        expected = 1 << 63
        self.assertEqual(int(states.encode_parity_state(1 << 31, 0, 32)), expected)
        self.assertEqual(int(states.encode_orient_state(1 << 31, 0, 32)), expected)

    def test_single_encoders_reject_graph_id_overflow(self):
        for encode in (states.encode_parity_state, states.encode_orient_state):
            with self.subTest(encode=encode.__name__):
                with self.assertRaises(ValueError):
                    encode(1 << 58, 0, 6)

    def test_combined_encoder_rejects_graph_id_at_E32(self):
        with self.assertRaises(ValueError):
            states.encode_combined_state(1, 0, 0, 32)

    def test_combined_E32_permutation_clears_old_bits(self):
        E = 32
        perm = np.arange(1, E + 1, dtype=np.int64)
        perm[0], perm[1] = perm[1], perm[0]
        # One parity bit on old edge 0 moves to edge 1.
        self.assertEqual(int(states.permute_combined_state(1, perm, E)), 2)

    def test_combined_E32_decoder_does_not_invent_graph_id(self):
        E = 32
        self.assertEqual(
            tuple(map(int, states.decode_combined_state(1, E))),
            (0, 0, 1)
        )

    def test_public_decode_accepts_entire_uint64_range(self):
        state = UINT64_MAX  # ordinary Python int
        g, b = states.decode_orient_state(state, 6)
        self.assertEqual((int(g), int(b)), (state >> 6, 63))

    def test_public_permutations_accept_python_int_above_int64(self):
        E = 6
        identity = np.arange(1, E + 1, dtype=np.int64)
        for func in (states.permute_orient_state, states.permute_parity_state):
            with self.subTest(func=func.__name__):
                self.assertEqual(int(func(UINT64_MAX, identity, E)), UINT64_MAX)
        self.assertEqual(
            int(states.permute_combined_state(UINT64_MAX, identity, E)), UINT64_MAX
        )

    def test_public_canonicalization_accepts_python_high_bit(self):
        E = 6
        identity = np.arange(1, E + 1, dtype=np.int64).reshape(1, -1)
        for func in (states.find_min_orient_state,
                     states.find_min_parity_state,
                     states.find_min_combined_state):
            with self.subTest(func=func.__name__):
                got = func(UINT64_MAX, identity, E)
                self.assertEqual(int(got), UINT64_MAX)
                self.assertNotIsInstance(got, (float, np.floating))

    def test_reject_invalid_edge_count(self):
    
        for decode, invalid_counts in (
            (states.decode_orient_state, (0, -1, 65)),
            (states.decode_parity_state, (0, -1, 65)),
            (states.decode_combined_state, (0, -1, 33)),
        ):
            for E in invalid_counts:
                with self.subTest(function=decode.__name__, E=E):
                    with self.assertRaises(ValueError):
                        decode(0, E)

    def test_reject_out_of_range_or_negative_states(self):
        for value in (-1, 1 << 64):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    states.decode_parity_state(value, 6)

    def test_reject_bad_signed_permutations(self):
        E = 3
        bad = (
            [],                 # wrong length
            [1, 1, 3],          # duplicate target
            [1, 0, 3],          # zero invalid
            [1, 2, 4],          # outside 1..E
            [1.9, 2, 3],        # float silently truncated if cast too early
        )
        for perm in bad:
            with self.subTest(perm=perm):
                with self.assertRaises((ValueError, TypeError)):
                    states.permute_parity_state(1, np.asarray(perm), E)

    def test_reject_bad_groups(self):
        E = 3
        bad_groups = (
            np.empty((0, E), dtype=np.int64),
            np.array([[1, 1, 3]], dtype=np.int64),
            np.array([[1, 2]], dtype=np.int64),
            np.array([[1.0, 2.1, 3.0]]),
        )
        for group in bad_groups:
            with self.subTest(group=group.tolist()):
                with self.assertRaises((ValueError, TypeError)):
                    states.find_min_parity_state(1, group, E)

    def test_combined_signed_permutation_matches_reference(self):
        E = 6
        perm = np.array([-3, 2, 1, 6, -5, 4], dtype=np.int64)
        graph_id, orient, parity = 5, 0b101101, 0b010111
        packed = (graph_id << (2 * E)) | (orient << E) | parity
        expected = reference_permute(packed, perm, E, 'combined')
        self.assertEqual(int(states.permute_combined_state(packed, perm, E)), expected)
        self.assertEqual(expected >> (2 * E), graph_id)

    def test_canonical_minimum_matches_reference_on_high_bits(self):
        E = 6
        identity = np.arange(1, E + 1, dtype=np.int64)
        involution = np.array([-2, -1, 3, 4, 5, 6], dtype=np.int64)
        group = np.stack([identity, involution])
        for kind, fn in (
            ('orient', states.find_min_orient_state),
            ('parity', states.find_min_parity_state),
            ('combined', states.find_min_combined_state),
        ):
            expected = min(reference_permute(UINT64_MAX - 17, p, E, kind) for p in group)
            got = int(fn(UINT64_MAX - 17, group, E))
            with self.subTest(kind=kind):
                self.assertEqual(got, expected)
                self.assertEqual(int(fn(got, group, E)), got)

    def test_reverse_orientation_is_not_silent_placeholder(self):
        E = 6
        state = (5 << E) | 0b001101
        result = states.reverse_orientation(state, E, canonical=False)
        self.assertIsNotNone(result)
        self.assertEqual(int(result), (5 << E) | (0b001101 ^ 0b111111))
        self.assertEqual(
            int(states.reverse_orientation(result, E, canonical=False)), state
        )


if __name__ == '__main__':
    unittest.main()
