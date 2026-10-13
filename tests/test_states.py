# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 12:59:53 2026

@author: cartin

Tests for packed orientation, parity, and combined states.

Conventions
-----------
- Number of graph edges |E| = 3n, where n is a positive integer.
- States are non-negative integers fitting within 64 unsigned bits.
- Edge decorations for a single property are represented by a bitmask of length
|E| bits. Order is given in terms of canonical edge ordering for each graph.
- States consist of graph ID followed by an appropriate decoration bitmask.
- Ordinary Python integers and Numpy integer scalars are accepted as states.
- Edge permutations perm[idx] are 1-based and signed, with abs values 1..|E|
appearing exactly once, mapping old edge idx to abs(perm[idx]) - 1. Negative
signs in perm[idx] reverse edge orientation, but not parity.
"""

# Go to trivalent folder and run
#
#   conda activate notebooks
#   python -m unittest tests.test_states
#
# The notebooks environment is needed to run the Hypothesis unittests

import unittest
import numpy as np

from trivalent.states import (
        decode_parity_state,
        decode_orient_state,
        decode_combined_state,
        
        encode_parity_state,
        encode_orient_state,
        encode_combined_state,
        
        permute_parity_bits,
        permute_orient_bits,
        
        permute_parity_state,
        permute_orient_state,
        permute_combined_state,
        
        find_min_parity_bits,
        find_min_orient_bits,
        find_min_combined_bits,
        
        find_min_parity_state,
        find_min_orient_state,
        find_min_combined_state,
        
        reverse_orientation
    )

# ================================================================
# INDEPENDENT REFERENCE IMPLEMENTATIONS
# ================================================================

def reference_bits(bits, perm, E, orient=False):
    """Apply a signed edge permutation using ordinary Python."""
    result = 0

    assert sorted(abs(int(x)) for x in perm) == list(range(1, E + 1))

    for old_idx, target in enumerate(perm):
        target = int(target)
        new_idx = abs(target) - 1

        bit = (bits >> old_idx) & 1

        if orient and target < 0:
            bit ^= 1

        result |= bit << new_idx

    return result


def reference_state(state, perm, E, kind):
    """Independent reference for the specified packed-state layout."""
    mask = (1 << E) - 1

    if kind == "combined":
        graph_id = state >> (2 * E)
        orient = (state >> E) & mask
        parity = state & mask

        new_o = reference_bits(orient, perm, E, orient=True)
        new_p = reference_bits(parity, perm, E, orient=False)

        return (graph_id << (2 * E)) | (new_o << E) | new_p

    graph_id = state >> E
    bits = state & mask

    new_bits = reference_bits(
        bits, perm, E, orient=(kind == "orient")
    )

    return (graph_id << E) | new_bits


def compose_signed(a, b):
    """Return b composed with a (apply a first, then b)."""
    result = []

    for target in a:
        target = int(target)

        intermediate = abs(target) - 1
        second = int(b[intermediate])

        sign = 1 if target > 0 else -1
        result.append(sign * second)

    return np.asarray(result, dtype=np.int8)


def inverse_signed(perm):
    """Construct the inverse 1-based signed permutation."""
    E = len(perm)
    inverse = np.zeros(E, dtype=np.int8)

    for old_idx, target in enumerate(perm):
        target = int(target)
        new_idx = abs(target) - 1

        inverse[new_idx] = (
            old_idx + 1 if target > 0 else -(old_idx + 1)
        )

    return inverse


# ================================================================
# TESTS
# ================================================================

class TestStates(unittest.TestCase):

    def test_encode_decode_roundtrip(self):
        """All three state encodings round-trip correctly."""

        for E in (3, 6, 9, 21):
            mask = (1 << E) - 1

            for graph_id in (0, 1, 7):
                for orient in (0, 1, mask // 3, mask):
                    parity = mask ^ orient

                    with self.subTest(E=E, g=graph_id, b=orient):

                        so = encode_orient_state(graph_id, orient, E)
                        sp = encode_parity_state(graph_id, parity, E)
                        sc = encode_combined_state(graph_id, orient, parity, E)

                        self.assertEqual(
                            decode_orient_state(so, E),
                            (graph_id, orient)
                        )

                        self.assertEqual(
                            decode_parity_state(sp, E),
                            (graph_id, parity)
                        )

                        self.assertEqual(
                            decode_combined_state(sc, E),
                            (graph_id, orient, parity)
                        )

                        # Verify exact storage format.
                        self.assertEqual(
                            so, (graph_id << E) | orient
                        )
                        self.assertEqual(
                            sp, (graph_id << E) | parity
                        )
                        self.assertEqual(
                            sc,
                            (graph_id << (2 * E))
                            | (orient << E)
                            | parity
                        )

    def test_identity_permutation(self):
        """Identity edge permutation must preserve every state."""

        E = 6
        identity = np.arange(1, E + 1, dtype=np.int8)

        g, o, p = 5, 0b101011, 0b110100

        cases = (
            (permute_orient_state, encode_orient_state(g, o, E)),
            (permute_parity_state, encode_parity_state(g, p, E)),
            (permute_combined_state, encode_combined_state(g, o, p, E)),
        )

        for func, state in cases:
            self.assertEqual(
                func(state, identity, E),
                state
            )

    def test_known_signed_permutation(self):
        """Check one small example with results computed by hand."""

        E = 3
        g = 2
        o = 0b101
        p = 0b101

        # Old edges 0,1,2 map to new edges 2,0,1.
        # The old edge 1 changes orientation.
        perm = np.array([3, -1, 2], dtype=np.int8)

        self.assertEqual(
            permute_orient_state(encode_orient_state(g, o, E), perm, E),
            (g << E) | 0b111
        )

        self.assertEqual(
            permute_parity_state(encode_parity_state(g, p, E), perm, E),
            (g << E) | 0b110
        )

        self.assertEqual(
            permute_combined_state(encode_combined_state(g, o, p, E), perm, E),
            (g << (2 * E)) | (0b111 << E) | 0b110
        )

    def test_permutation_against_reference(self):
        """Compare all transformations against independent Python code."""

        E = 6
        g, o, p = 7, 0b101001, 0b011101

        perm = np.array([-3, 2, 1, 6, -5, 4], dtype=np.int8)

        cases = (
            ("orient", permute_orient_state, encode_orient_state(g, o, E)),
            ("parity", permute_parity_state, encode_parity_state(g, p, E)),
            ("combined", permute_combined_state, encode_combined_state(g, o, p, E)),
        )

        for kind, func, state in cases:
            self.assertEqual(
                func(state, perm, E),
                reference_state(state, perm, E, kind)
            )

    def test_combined_consistency(self):
        """Combined transformations agree with separate transformations."""

        E = 6
        g, o, p = 7, 0b101101, 0b010111

        perm = np.array([-3, 2, 1, 6, -5, 4], dtype=np.int8)

        so = permute_orient_state(encode_orient_state(g, o, E), perm, E)
        sp = permute_parity_state(encode_parity_state(g, p, E), perm, E)
        sc = permute_combined_state(encode_combined_state(g, o, p, E), perm, E)

        go, new_o = decode_orient_state(so, E)
        gp, new_p = decode_parity_state(sp, E)

        self.assertEqual(go, g)
        self.assertEqual(gp, g)

        self.assertEqual(
            decode_combined_state(sc, E),
            (g, new_o, new_p)
        )

    def test_permutation_composition_and_inverse(self):
        """The state operations respect signed-permutation composition."""

        E = 6
        g, o, p = 5, 0b101011, 0b011001

        a = np.array([-3, 2, 1, 6, -5, 4], dtype=np.int8)
        b = np.array([2, -1, -4, 3, 6, 5], dtype=np.int8)

        composed = compose_signed(a, b)
        inverse = inverse_signed(a)

        cases = (
            (permute_orient_state, encode_orient_state(g, o, E)),
            (permute_parity_state, encode_parity_state(g, p, E)),
            (permute_combined_state, encode_combined_state(g, o, p, E)),
        )

        for func, state in cases:

            # Apply a, then b.
            sequential = func(func(state, a, E), b, E)

            # Apply their composition directly.
            direct = func(state, composed, E)

            self.assertEqual(sequential, direct)

            # Apply a and then its inverse.
            restored = func(func(state, a, E), inverse, E)

            self.assertEqual(restored, state)

    def test_canonical_minimum(self):
        """Minimization agrees with direct orbit enumeration."""

        E = 6
        g, o, p = 5, 0b101001, 0b010111

        identity = np.arange(1, E + 1, dtype=np.int8)

        # A signed involution: applying it twice is identity.
        involution = np.array(
            [-2, -1, 3, 4, 5, 6],
            dtype=np.int8
        )

        group = np.stack((identity, involution))

        cases = (
            (permute_orient_state, find_min_orient_state, encode_orient_state(g, o, E)),
            (permute_parity_state, find_min_parity_state, encode_parity_state(g, p, E)),
            (permute_combined_state, find_min_combined_state, encode_combined_state(g, o, p, E)),
        )

        for permute, minimum, state in cases:

            orbit = [
                permute(state, sigma, E)
                for sigma in group
            ]

            expected = min(orbit)
            actual = minimum(state, group, E)

            self.assertEqual(actual, expected)

            # Canonicalization should be idempotent.
            self.assertEqual(
                minimum(actual, group, E), actual
            )

            # Both orbit elements must have the same minimum.
            for related_state in orbit:
                self.assertEqual(
                    minimum(related_state, group, E),
                    expected
                )
                
    def test_single_encoder_high_bit(self):
        """
        Single-decoration states must retain their unsigned value
        when the highest bit (bit 63) is set.
        """
    
        E = 32
        graph_num = 1 << 31
        expected = 1 << 63
    
        for encode, decode in (
            (encode_orient_state, decode_orient_state),
            (encode_parity_state, decode_parity_state),
        ):
            state = encode(graph_num, 0, E)
    
            self.assertEqual(int(state), expected)
            self.assertGreaterEqual(int(state), 0)
    
            decoded_graph, decoded_bits = decode(state, E)
    
            self.assertEqual(int(decoded_graph), graph_num)
            self.assertEqual(int(decoded_bits), 0)
            
    def test_single_decode_bit_boundary(self):
        """
        Decoding must work correctly when the state occupies
        the highest bits of uint64.
        """
    
        state = (1 << 64) - 1
    
        for decode in (
            decode_orient_state,
            decode_parity_state,
        ):
            # E = 63: one bit remains for the graph ID
            graph_num, bits = decode(state, 63)
    
            self.assertEqual(int(graph_num), 1)
            self.assertEqual(int(bits), (1 << 63) - 1)
    
            # E = 64: the entire word contains decoration bits
            graph_num, bits = decode(state, 64)
    
            self.assertEqual(int(graph_num), 0)
            self.assertEqual(int(bits), state)
            
    def test_single_permutation_high_bit(self):
        """
        Identity permutations must preserve states occupying
        the highest bit of uint64.
        """
    
        E = 32
        graph_num = 1 << 31
    
        identity = np.arange(1, E + 1, dtype=np.int64)
    
        for encode, permute in (
            (encode_orient_state, permute_orient_state),
            (encode_parity_state, permute_parity_state),
        ):
    
            state = encode(graph_num, 0, E)
            result = permute(state, identity, E)
    
            self.assertEqual(int(result), int(state))
            self.assertEqual(int(result), 1 << 63)

    def test_combined_permutation_edge_limit(self):
        """
        A combined state cannot hold two 33-bit
        decorations in a single uint64.
        """
    
        E = 33
        identity = np.arange(1, E + 1, dtype=np.int64)
    
        with self.assertRaises(ValueError):
            permute_combined_state(0, identity, E)
            
    def test_symmetry_group_validation(self):
        """
        Canonicalization must reject malformed symmetry groups.
        """
    
        E = 3
    
        valid = np.array([
            [1, 2, 3],
            [-2, -1, 3]
        ], dtype=np.int8)
    
        invalid_groups = [
            np.array([[1, 1, 3]], dtype=np.int8),  # Repeated edge
            np.array([[1, 2]], dtype=np.int8),     # Wrong length
            np.array([1, 2, 3], dtype=np.int8),    # Not 2D
            np.empty((0, E), dtype=np.int8),      # Empty group
            np.array([[1., 2., 3.]]),             # Float labels
            np.array([[0, 2, 3]], dtype=np.int8),  # Zero label
            np.array([[1, 2, 4]], dtype=np.int8),  # Out of range
        ]
    
        for find_min in (
            find_min_orient_state,
            find_min_parity_state,
            find_min_combined_state,
        ):
    
            # A valid group must be accepted.
            result = find_min(0, valid, E)
            self.assertIsInstance(int(result), int)
    
            # Every invalid group must be rejected.
            for group in invalid_groups:
    
                with self.subTest(
                    function=find_min.__name__,
                    group=repr(group)
                ):
                    with self.assertRaises((ValueError, TypeError)):
                        find_min(0, group, E)
                        
    def test_signed_permutation_uint64_overflow(self):
        """
        Invalid unsigned permutation labels must not wrap
        into valid negative signed labels.
        """
    
        E = 6
    
        bad_perm = np.arange(1, E + 1, dtype=np.uint64)
        bad_perm[0] = np.iinfo(np.uint64).max
    
        for permute in (
            permute_orient_state,
            permute_parity_state,
            permute_combined_state,
        ):
            with self.subTest(function=permute.__name__):
                with self.assertRaises(ValueError):
                    permute(0, bad_perm, E)
                    
    def test_known_signed_bit_permutation(self):
        """
        Verify the bitmask permutation independently
        of state encoding and decoding.
        """
    
        E = 3
    
        # Old edges 0,1,2 map to new edges 2,0,1.
        # Old edge 1 also reverses direction.
        perm = np.array([3, -1, 2], dtype=np.int64)
    
        orient_bits = 0b101
        parity_bits = 0b101
    
        new_orient = permute_orient_bits(
            orient_bits, perm, E
        )
    
        new_parity = permute_parity_bits(
            parity_bits, perm, E
        )
    
        self.assertEqual(new_orient, 0b111)
        self.assertEqual(new_parity, 0b110)
        
    def test_bitmask_canonicalization(self):
    
        E = 3
        graph_num = 5
    
        identity = np.array([1, 2, 3], dtype=np.int8)
        involution = np.array([-2, -1, 3], dtype=np.int8)
        group = np.stack((identity, involution))
    
        orient_bits = 0b011
        parity_bits = 0b010
    
        # Canonical bitmasks
        min_o = find_min_orient_bits(orient_bits, group, E)
        min_p = find_min_parity_bits(parity_bits, group, E)
    
        self.assertEqual(min_o, 0b000)
        self.assertEqual(min_p, 0b001)
    
        # Related bitmasks must have the same minimum
        self.assertEqual(
            find_min_orient_bits(
                permute_orient_bits(orient_bits, involution, E),
                group, E
            ),
            min_o
        )
    
        self.assertEqual(
            find_min_parity_bits(
                permute_parity_bits(parity_bits, involution, E),
                group, E
            ),
            min_p
        )
    
        # Packed-state minimization must agree
        orient_state = encode_orient_state(graph_num, orient_bits, E)
        parity_state = encode_parity_state(graph_num, parity_bits, E)
    
        self.assertEqual(
            int(find_min_orient_state(orient_state, group, E)),
            int(encode_orient_state(graph_num, min_o, E))
        )
    
        self.assertEqual(
            int(find_min_parity_state(parity_state, group, E)),
            int(encode_parity_state(graph_num, min_p, E))
        )

    def test_combined_bitmask_canonicalization(self):
    
        E = 3
        graph_num = 5
    
        group = np.array([
            [1, 2, 3],
            [-2, -1, 3]
        ], dtype=np.int8)
    
        orient_bits = 0b011
        parity_bits = 0b001
    
        combined_bits = (orient_bits << E) | parity_bits
    
        # Identity: (011, 001) -> combined value 25
        # Involution: (000, 010) -> combined value 2
    
        minimum = find_min_combined_bits(
            combined_bits, group, E
        )
    
        self.assertEqual(minimum, 0b010)
    
        # The packed-state operation must give the same result.
    
        state = encode_combined_state(
            graph_num, orient_bits, parity_bits, E
        )
    
        min_state = find_min_combined_state(
            state, group, E
        )
    
        expected = encode_combined_state(
            graph_num, 0b000, 0b010, E
        )
    
        self.assertEqual(int(min_state), int(expected))
    
        # Independent minimization would give (000, 001),
        # which is NOT a member of the combined symmetry orbit.
    
        independent = (
            (find_min_orient_bits(orient_bits, group, E) << E)
            | find_min_parity_bits(parity_bits, group, E)
        )
    
        self.assertNotEqual(independent, minimum)
            
    def test_minimization_invalid_edge_counts(self):
        """
        Canonicalization functions must validate num_edges
        before processing bitmasks or symmetry groups.
        """
    
        group = np.array([[1]], dtype=np.int64)
    
        cases = (
            (find_min_parity_bits, (0, -1, 65)),
            (find_min_orient_bits, (0, -1, 65)),
            (find_min_combined_bits, (0, -1, 33)),
            (find_min_parity_state, (0, -1, 65)),
            (find_min_orient_state, (0, -1, 65)),
            (find_min_combined_state, (0, -1, 33)),
        )
    
        for function, invalid_counts in cases:
    
            for E in invalid_counts:
    
                with self.subTest(
                    function=function.__name__,
                    num_edges=E
                ):
                    with self.assertRaisesRegex(
                        ValueError, "Number of edges"
                    ):
                        function(0, group, E)

if __name__ == "__main__":
    unittest.main()
