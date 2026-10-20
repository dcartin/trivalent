# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 07:44:00 2026

@author: cartin
"""

import operator

import numba as nb
import numpy as np

#=============================================================================#
# Helper functions
#=============================================================================#

UINT64_MAX = (1 << 64) - 1

def _as_uint64(state):
    """
        Normalize Python/Numpy integet to uint64
    """
    
    value = operator.index(state)
    
    if not 0 <= value <= UINT64_MAX:
        raise ValueError("State must fit in an unsigned 64-bit integer")
        
    return np.uint64(value)

#-----------------------------------------------------------------------------#

def _check_num_edges(num_edges, max_edges):
    """
        Checks num_edges against maximum size for bitmask/state type
    """

    num_edges = operator.index(num_edges)

    if not 1 <= num_edges <= max_edges:
        raise ValueError(f"Number of edges must be between 1 and {max_edges}")

    return num_edges

#-----------------------------------------------------------------------------#

def _check_edge_bits(bits, num_edges):
    """
        Validate a binary edge decoration of length num_edges, return as uint64
    """
    
    if not 1 <= num_edges <= 64:
        raise ValueError("Bitmasks must have between 1 and 64 edges")
        
    bits = _as_uint64(bits)
    
    if int(bits) >= (1 << num_edges):
        raise ValueError("Bitmask exceeds given number of edges")
        
    return bits

#-----------------------------------------------------------------------------#

def _check_signed_perm(perm, num_edges):
    
    # Casting a floating-point array immediately with np.asarray(..., dtype =
    # np.int64) silently truncates fractional labels. Reject non-integer labels
    # first, then normalize dtype for Numba.
    
    raw = np.asarray(perm)
    
    if raw.dtype.kind not in "iu":
        raise TypeError("Permutation labels must be integers")
        
    if raw.ndim != 1 or raw.shape[0] != num_edges:
        raise TypeError("Permutation must be 1D array of length |E|")
        
    # Reject out-of-range values before casting to int64. This also prevents
    # large uint64 values from wrapping to apparently valid negative labels.
    
    if (np.any(raw > num_edges) or (raw.dtype.kind == "i" and np.any(raw < -num_edges))):
        raise ValueError("Permutation labels are out of range")
        
    p = np.asarray(raw, dtype = np.int64)
    
    if not np.array_equal(np.sort(np.abs(p)), np.arange(1, num_edges + 1)):
        raise ValueError("Absolute value of labels must be exactly 1..|E|")
        
    return np.ascontiguousarray(p)

#-----------------------------------------------------------------------------#

def _check_sym_grp(sym_grp, num_edges):
    """
        Check that sym_grp is non-empty 2D array of valid signed edge permutations
    """
    
    raw = np.asarray(sym_grp)
    
    # Check array dimensions
    
    if (raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] != num_edges):
        raise ValueError("Symmetry group must have shape (|G|, |E|) with |G| > 0")
        
    # Check integer types before conversion

    if raw.dtype.kind not in "iu":
        raise TypeError("Symmetry labels must be integers")
        
    # Reject out-of-range values before casting to int64. This also prevents
    # large uint64 values from wrapping to apparently valid negative labels.
    
    if (np.any(raw > num_edges) or (raw.dtype.kind == "i" and np.any(raw < -num_edges))):
        raise ValueError("Symmetry labels are out of range")
        
    # Check individual group elements using previous permutation validator;
    # this casts labels to int64.
    
    checked_sym_grp = [_check_signed_perm(elem, num_edges) for elem in raw]
    
    return np.ascontiguousarray(np.stack(checked_sym_grp))

#=============================================================================#
# Bitmask  utility functions
#=============================================================================#

#-----------------------------------------------------------------------------#
# Compiled canonicalization kernels
#-----------------------------------------------------------------------------#

@nb.njit(inline = "always")
def _combine_bits(orient_bits, parity_bits, num_edges):
    return np.uint64((orient_bits << num_edges) | parity_bits)
    
#-----------------------------------------------------------------------------#

@nb.njit(inline = "always")
def _separate_bits(combined_bits, num_edges):

    # Ensure unsigned integer arithmetic throughout, even if function is called
    # using ordinary Python integer
    
    combined_bits = np.uint64(combined_bits)

    # Edge mask is set this way, rather than (1 <'< num_edges) - 1, to avoid
    # intermediate overflow issue for uint64 when |E| = 64.
    
    edge_mask = np.uint64(0xFFFFFFFFFFFFFFFF) >> (64 - num_edges)
    
    orient_bits = ((combined_bits >> num_edges) & edge_mask)
    parity_bits = (combined_bits & edge_mask)
    
    return orient_bits, parity_bits

#-----------------------------------------------------------------------------#
# Public-facing functions with argument compatibility checks
#-----------------------------------------------------------------------------#

def combine_bits(orient_bits, parity_bits, num_edges):

    # Ensure that given arguments are integers
    
    orient_bits = operator.index(orient_bits)
    parity_bits = operator.index(parity_bits)
    num_edges = _check_num_edges(num_edges, 32)
    
    return _combine_bits(orient_bits, parity_bits, num_edges)

#-----------------------------------------------------------------------------#

def separate_bits(combined_bits, num_edges):
    
    combined_bits = operator.index(combined_bits)
    num_edges = _check_num_edges(num_edges, 32)
    
    return _separate_bits(combined_bits, num_edges)

#=============================================================================#
# Encode/decode graph states
#=============================================================================#

# Signed overflow in single encoder
#
# Current width checks allow values occupying bit 63, which Numba can
# return as a negative signed integer, e.g. _encode_parity_state(1 << 31, 0, 32).
# After validating graph_num, bits, num_edges in Python, perform packing using
# Python integer shifts and return ordinary Python integer. Do not send large
# graph IDs to signed Numba packing helper.

#-----------------------------------------------------------------------------#
# Compiled canonicalization kernels
#-----------------------------------------------------------------------------#

@nb.njit(inline = "always")
def _decode_single_state(state, num_edges):

    state = np.uint64(state)
    
    # Edge mask is set this way, rather than (1 << num_edges) - 1, to avoid
    # intermediate overflow issue for uint64 when |E| = 64.
    
    edge_mask = np.uint64(0xFFFFFFFFFFFFFFFF) >> (64 - num_edges)

    if num_edges == 64:
        graph_num = np.uint64(0)
    else:
        graph_num = (state >> num_edges)
        
    bits = (state & edge_mask)
    
    return graph_num, bits

#-----------------------------------------------------------------------------#

@nb.njit(inline = "always")
def _encode_single_state(graph_num, bits, num_edges):
    return (np.uint64(graph_num) << num_edges) | np.uint64(bits)

#-----------------------------------------------------------------------------#

@nb.njit(inline = "always")
def _decode_parity_state(parity_state, num_edges):
    """
        Parity states in form (graph ID << E bits) + (parity bits)
    """
    
    return _decode_single_state(parity_state, num_edges)

#-----------------------------------------------------------------------------#

@nb.njit(inline = "always")
def _decode_orient_state(orient_state, num_edges):
    """
        Orientation states in form (graph ID << E bits) + (orientation bits)
    """
    
    return _decode_single_state(orient_state, num_edges)

#-----------------------------------------------------------------------------#

@nb.njit(inline = "always")
def _decode_combined_state(combo_state, num_edges):
    """
        Combined states in form (graph ID << 2E bits) + (orientation bits) + (parity bits)
    """
    
    # Ensure unsigned integer arithmetic throughout, even if function is called
    # using ordinary Python integer
    
    combo_state = np.uint64(combo_state)
    
    # When |E| = 32, there is no room for a graph ID, so use zero; to ensure
    # that both branches give the same data type, use np.uint64()
    
    if num_edges == 32:
        graph_num = np.uint64(0)
    else:
        graph_num = (combo_state >> (2 * num_edges))
            
    # Edge mask is set this way, rather than (1 << num_edges) - 1, to avoid
    # intermediate overflow issue for uint64 when |E| = 64.

    edge_mask = np.uint64(0xFFFFFFFFFFFFFFFF) >> (64 - num_edges)
    
    orient_bits = ((combo_state >> num_edges) & edge_mask)
    parity_bits = (combo_state & edge_mask)
    
    return graph_num, orient_bits, parity_bits

#-----------------------------------------------------------------------------#

@nb.njit(inline = "always")
def _encode_parity_state(graph_num, parity_bits, num_edges):
    """
        Parity states in form (graph ID << E bits) + (parity bits)
    """
    
    return _encode_single_state(graph_num, parity_bits, num_edges)

#-----------------------------------------------------------------------------#

@nb.njit(inline = "always")
def _encode_orient_state(graph_num, orient_bits, num_edges):
    """
        Orientation states in form (graph ID << E bits) + (orientation bits)
    """
    
    return _encode_single_state(graph_num, orient_bits, num_edges)

#-----------------------------------------------------------------------------#

@nb.njit(inline = "always")
def _encode_combined_state(graph_num, orient_bits, parity_bits, num_edges):
    """
        Combined states in form (graph ID << 2E bits) + (orientation bits) + (parity bits)
    """
    
    return np.uint64((graph_num << (2 * num_edges)) | (orient_bits << num_edges) | parity_bits)

#-----------------------------------------------------------------------------#
# Public-facing functions with argument compatibility checks
#-----------------------------------------------------------------------------#

def decode_parity_state(parity_state, num_edges):
    return _decode_parity_state(_as_uint64(parity_state), _check_num_edges(num_edges, 64))

#-----------------------------------------------------------------------------#

def decode_orient_state(orient_state, num_edges):
    return _decode_orient_state(_as_uint64(orient_state), _check_num_edges(num_edges, 64))

#-----------------------------------------------------------------------------#

def decode_combined_state(combined_state, num_edges, separate = True):
    graph_num, orient_bits, parity_bits = _decode_combined_state(_as_uint64(combined_state), _check_num_edges(num_edges, 32))

    if separate:
        return graph_num, orient_bits, parity_bits
    else:
        return graph_num, (orient_bits << num_edges) | parity_bits

#-----------------------------------------------------------------------------#

def encode_parity_state(graph_num, parity_bits, num_edges):
    """
        Parity states in form (graph ID << E bits) + (parity bits)
    """
    
    # Ensure that given arguments are integers
    
    graph_num = operator.index(graph_num)
    parity_bits = operator.index(parity_bits)
    num_edges = _check_num_edges(num_edges, 64)
    
    # Check all compatibility tests for given arguments
        
    if parity_bits < 0:
        raise ValueError('Parity bitmask must be non-negative')
        
    if parity_bits >= (1 << num_edges):
        raise ValueError('Parity bitmask longer than number of edges')
    
    if graph_num < 0:
        raise ValueError('Graph ID must be non-negative')
        
    if graph_num >= (1 << (64 - num_edges)):
        raise ValueError('Graph ID does not fit in 64 bit state')
        
    # Return encoded information as single state integer
    
    return (graph_num << num_edges) | parity_bits

#-----------------------------------------------------------------------------#

def encode_orient_state(graph_num, orient_bits, num_edges):
    """
        Orientation states in form (graph ID << E bits) + (orientation bits)
    """
    
    # Ensure that given arguments are integers
    
    graph_num = operator.index(graph_num)
    orient_bits = operator.index(orient_bits)
    num_edges = _check_num_edges(num_edges, 64)
    
    # Check all compatibility tests for given arguments
        
    if orient_bits < 0:
        raise ValueError('Orientation bitmask must be non-negative')
        
    if orient_bits >= (1 << num_edges):
        raise ValueError('Orientation bitmask longer than number of edges')
    
    if graph_num < 0:
        raise ValueError('Graph ID must be non-negative')
        
    if graph_num >= (1 << (64 - num_edges)):
        raise ValueError('Graph ID does not fit in 64 bit state')
        
    # Return encoded information as single state integer
    
    return (graph_num << num_edges) | orient_bits

#-----------------------------------------------------------------------------#

def encode_combined_state(graph_num, orient_bits, parity_bits, num_edges):
    """
        Combined states in form (graph ID << 2E bits) + (orientation << E bits) + (parity bits)
    """
    
    # Ensure that given arguments are integers
    
    graph_num = operator.index(graph_num)
    orient_bits = operator.index(orient_bits)
    parity_bits = operator.index(parity_bits)
    num_edges = _check_num_edges(num_edges, 32)
    
    # Check all compatibility tests for given arguments
        
    if orient_bits < 0 or parity_bits < 0:
        raise ValueError('Given bitmasks must be non-negative')
    
    if orient_bits >= (1 << num_edges):
        raise ValueError('Orientation bitmask longer than number of edges')
        
    if parity_bits >= (1 << num_edges):
        raise ValueError('Parity bitmask longer than number of edges')
        
    if graph_num < 0:
        raise ValueError('Graph ID must be non-negative')
        
    if graph_num >= (1 << (64 - 2 * num_edges)):
        raise ValueError('Graph ID does not fit in 64 bit combined state')
        
    # Return encoded information as single state integer
    
    return (graph_num << (2 * num_edges)) | (orient_bits << num_edges) | parity_bits

#-----------------------------------------------------------------------------#

def encode_combined_bits_state(graph_num, combined_bits, num_edges):
    """
        Optional version if combined bitmask is easier than separated bitmasks
    """
    
    num_edges = _check_num_edges(num_edges, 32)
    orient_bits, parity_bits = separate_bits(combined_bits, num_edges)
    
    return encode_combined_state(graph_num, orient_bits, parity_bits, num_edges)

#=============================================================================#
# Permutations of graph states
#=============================================================================#

@nb.njit(cache = True)
def _permute_parity_bits(parity_bits, signed_perm, num_edges):
    """
        Use signed permutation on parity bits to maximize number of edges possible
    """
    
    # Make sure all variables are unsigned integers, rather than default signed,
    # to avoid float64 variables arising from later Numba promotion in
    # perm_parity_bits from reconciling int64 and uint64
    
    perm_parity_bits = np.uint64(0)
    
    # There is no check of edge orientation flip necessary, so no XOR needed,
    # simply move old parity value to its new location; because left shift here
    # can potentially exceed 64-bit limit, wrap in np.uint64().
    
    for idx, target in enumerate(signed_perm):
        target_idx = abs(target) - 1
        current_bit = (parity_bits >> idx) & 1
        
        perm_parity_bits |= np.uint64(current_bit << target_idx)
        
    return perm_parity_bits

#-----------------------------------------------------------------------------#

@nb.njit(cache = True)
def _permute_orient_bits(orient_bits, signed_perm, num_edges):
    """
        Use signed permutation on orientation bitmask (graph ID << E bits) + (orientation bits)
    """
    
    # Make sure all variables are unsigned integers, rather than default signed,
    # to avoid float64 variables arising from later Numba promotion in
    # perm_orient_bits from reconciling int64 and uint64
    
    perm_orient_bits = np.uint64(0)
    
    # The signed permutation is 1-based and records changes in edge orientation
    # due to the symmetry element; thus, if the orientation flips, the resulting
    # orientation direction must flip as well, done by XOR on orientation value;
    # because left shift here can potentially exceed 64-bit limit, wrap in
    # np.uint64().
    
    for idx, target in enumerate(signed_perm):
        target_idx = abs(target) - 1
        current_bit = (orient_bits >> idx) & 1
        
        if target < 0:
            current_bit ^= 1
            
        perm_orient_bits |= np.uint64(current_bit << target_idx)
        
    return perm_orient_bits

#-----------------------------------------------------------------------------#

@nb.njit(cache = True)
def _permute_combined_bits(combined_bits, signed_perm, num_edges):
    
    orient_bits, parity_bits = _separate_bits(combined_bits, num_edges)
    
    perm_orient_bits = _permute_orient_bits(orient_bits, signed_perm, num_edges)
    perm_parity_bits = _permute_parity_bits(parity_bits, signed_perm, num_edges)
    
    return _combine_bits(perm_orient_bits, perm_parity_bits, num_edges)

#-----------------------------------------------------------------------------#
# Public-facing functions with argument compatibility checks
#-----------------------------------------------------------------------------#

def permute_parity_bits(parity_bits, signed_perm, num_edges):
    """
        Use signed permutation on parity bitmask, independent of graph ID
    """
    
    num_edges = _check_num_edges(num_edges, 64)
    
    parity_bits = _check_edge_bits(parity_bits, num_edges)
    signed_perm = _check_signed_perm(signed_perm, num_edges)
    
    return int(_permute_parity_bits(parity_bits, signed_perm, num_edges))
    
#-----------------------------------------------------------------------------#

def permute_orient_bits(orient_bits, signed_perm, num_edges):
    """
        Use signed permutation on orientation bitmask, independent of graph ID
    """
    
    num_edges = _check_num_edges(num_edges, 64)
    
    orient_bits = _check_edge_bits(orient_bits, num_edges)
    signed_perm = _check_signed_perm(signed_perm, num_edges)
    
    return int(_permute_orient_bits(orient_bits, signed_perm, num_edges))
    
#-----------------------------------------------------------------------------#
    
def permute_combined_bits(combined_bits, signed_perm, num_edges):
    """
        Use signed permutation on combined bitmask, independent of graph ID;
        note it is assumed only 32 bits max, since must fit in combined state
    """
    
    num_edges = _check_num_edges(num_edges, 32)
    
    signed_perm = _check_signed_perm(signed_perm, num_edges)
    
    return _permute_combined_bits(combined_bits, signed_perm, num_edges)

#-----------------------------------------------------------------------------#

def permute_parity_state(parity_state, signed_perm, num_edges):
    """
        Use signed permutation on parity bitmask (graph ID << E bits) + (parity bits)
    """
    
    # Ensure that given arguments are integers
    
    parity_state = _as_uint64(parity_state)
    num_edges = _check_num_edges(num_edges, 64)
    
    signed_perm = _check_signed_perm(signed_perm, num_edges)
        
    # Return permuted state
    
    graph_num, parity_bits = _decode_parity_state(parity_state, num_edges)
    parity_bits = _permute_parity_bits(parity_bits, signed_perm, num_edges)
    
    return _encode_parity_state(graph_num, parity_bits, num_edges)

#-----------------------------------------------------------------------------#

def permute_orient_state(orient_state, signed_perm, num_edges):
    """
        Use signed permutation on orientation bitmask (graph ID << E bits) + (orientation bits)
    """
    
    # Ensure that given arguments are integers
    
    orient_state = _as_uint64(orient_state)
    num_edges = _check_num_edges(num_edges, 64)
    
    signed_perm = _check_signed_perm(signed_perm, num_edges)
        
    # Return permuted state
    
    graph_num, orient_bits = _decode_orient_state(orient_state, num_edges)
    orient_bits = _permute_orient_bits(orient_bits, signed_perm, num_edges)
    
    return _encode_orient_state(graph_num, orient_bits, num_edges)

#-----------------------------------------------------------------------------#

def permute_combined_state(combined_state, signed_perm, num_edges):
    """
        Use signed permutation on combined bitmask (graph ID << 2E bits) +
        (orientation << E bits) + (parity bits)
    """
    
    # Ensure that given arguments are integers
    
    combined_state = _as_uint64(combined_state)
    num_edges = _check_num_edges(num_edges, 32)
    
    # Check all compatibility tests for given arguments, to ensure two
    # individual bitmasks can fit inside combined np.uint64 bitmask
        
    signed_perm = _check_signed_perm(signed_perm, num_edges)
        
    # Return permuted state
    
    graph_num, orient_bits, parity_bits = _decode_combined_state(combined_state, num_edges)
    orient_bits = _permute_orient_bits(orient_bits, signed_perm, num_edges)
    parity_bits = _permute_parity_bits(parity_bits, signed_perm, num_edges)
    
    return _encode_combined_state(graph_num, orient_bits, parity_bits, num_edges)

#=============================================================================#
# Find lexicographic minimum graph state
#=============================================================================#

#-----------------------------------------------------------------------------#
# Compiled canonicalization kernels
#-----------------------------------------------------------------------------#

@nb.njit(cache = True)
def _find_min_parity_bits(parity_bits, sym_grp, num_edges):
    
    min_parity_bits = np.uint64(parity_bits)
    min_grp_idx = -1
    
    for idx in range(sym_grp.shape[0]):
        temp_parity_bits = _permute_parity_bits(parity_bits, sym_grp[idx], num_edges)
        if temp_parity_bits < min_parity_bits:
            min_parity_bits = temp_parity_bits
            min_grp_idx = idx
            
    return min_parity_bits, min_grp_idx

#-----------------------------------------------------------------------------#

@nb.njit(cache = True)
def _find_min_orient_bits(orient_bits, sym_grp, num_edges):
    
    min_orient_bits = np.uint64(orient_bits)
    min_grp_idx = -1
    
    for idx in range(sym_grp.shape[0]):
        temp_orient_bits = _permute_orient_bits(orient_bits, sym_grp[idx], num_edges)
        if temp_orient_bits < min_orient_bits:
            min_orient_bits = temp_orient_bits
            min_grp_idx = idx
            
    return min_orient_bits, min_grp_idx

#-----------------------------------------------------------------------------#

@nb.njit(cache = True)
def _find_min_combined_bits(combined_bits, sym_grp, num_edges):
    
    edge_mask = (np.uint64(0xFFFFFFFFFFFFFFFF) >> (64 - num_edges))
    
    orient_bits = (combined_bits >> num_edges) & edge_mask
    parity_bits = combined_bits & edge_mask
    
    min_combined_bits = combined_bits
    min_grp_idx = -1
    
    for idx in range(sym_grp.shape[0]):
        perm = sym_grp[idx]
        
        temp_orient_bits = _permute_orient_bits(orient_bits, perm, num_edges)
        temp_parity_bits = _permute_parity_bits(parity_bits, perm, num_edges)
        
        temp_combined_bits = (temp_orient_bits << num_edges) | temp_parity_bits
        
        if temp_combined_bits < min_combined_bits:
            min_combined_bits = temp_combined_bits
            min_grp_idx = idx
            
    return min_combined_bits, min_grp_idx
    
#-----------------------------------------------------------------------------#

@nb.njit(cache = True)
def _find_min_parity_state(parity_state, sym_grp, num_edges):
    
    graph_num, parity_bits = _decode_parity_state(parity_state, num_edges)
    min_parity_bits, min_grp_idx = _find_min_parity_bits(parity_bits, sym_grp, num_edges)
    min_parity_state = _encode_parity_state(graph_num, min_parity_bits, num_edges)

    return min_parity_state, min_grp_idx

#-----------------------------------------------------------------------------#

@nb.njit(cache = True)
def _find_min_orient_state(orient_state, sym_grp, num_edges):
    
    graph_num, orient_bits = _decode_orient_state(orient_state, num_edges)
    min_orient_bits, min_grp_idx = _find_min_orient_bits(orient_bits, sym_grp, num_edges)
    min_orient_state = _encode_orient_state(graph_num, min_orient_bits, num_edges)
    
    return min_orient_state, min_grp_idx

#-----------------------------------------------------------------------------#

@nb.njit(cache = True)
def _find_min_combined_state(combined_state, sym_grp, num_edges):
    
    graph_num, orient_bits, parity_bits = _decode_combined_state(combined_state, num_edges)
    combined_bits = (np.uint64(orient_bits) << num_edges) | np.uint64(parity_bits)
    
    min_combined_bits, min_grp_idx = _find_min_combined_bits(combined_bits, sym_grp, num_edges)
    
    edge_mask = (np.uint64(0xFFFFFFFFFFFFFFFF) >> (64 - num_edges))
    
    min_orient_bits = min_combined_bits >> num_edges
    min_parity_bits = min_combined_bits & edge_mask
            
    min_combined_state = _encode_combined_state(graph_num, min_orient_bits, min_parity_bits, num_edges)
    
    return min_combined_state, min_grp_idx

#-----------------------------------------------------------------------------#
# Public-facing functions with normalized states
#-----------------------------------------------------------------------------#

def find_min_parity_bits(parity_bits, sym_grp, num_edges, return_perm = False):
    """
        Find minimum parity bitmask, with option to return signed permutation
    """
    
    parity_bits = _check_edge_bits(parity_bits, num_edges)
    sym_grp = _check_sym_grp(sym_grp, num_edges)
    num_edges = _check_num_edges(num_edges, 64)
    
    min_parity_bits, min_grp_idx = _find_min_parity_bits(
        _check_edge_bits(parity_bits, num_edges),
        _check_sym_grp(sym_grp, num_edges),
        num_edges
        )
    
    if not return_perm:
        return int(min_parity_bits)
    
    if min_grp_idx == -1:   # No minimum permutation found
        min_perm = np.arange(1, num_edges + 1, dtype = np.int64)
    else:
        min_perm = sym_grp[min_grp_idx].copy()
        
    return int(min_parity_bits), min_perm

#-----------------------------------------------------------------------------#

def find_min_orient_bits(orient_bits, sym_grp, num_edges, return_perm = False):
    """
        Find minimum orientation bitmask, with option to return signed permutation
    """
    
    orient_bits = _check_edge_bits(orient_bits, num_edges)
    sym_grp = _check_sym_grp(sym_grp, num_edges)
    num_edges = _check_num_edges(num_edges, 64)
    
    min_orient_bits, min_grp_idx = _find_min_orient_bits(
        _check_edge_bits(orient_bits, num_edges),
        _check_sym_grp(sym_grp, num_edges),
        num_edges
        )
    
    if not return_perm:
        return int(min_orient_bits)
    
    if min_grp_idx == -1:   # No minimum permutation found
        min_perm = np.arange(1, num_edges + 1, dtype = np.int64)
    else:
        min_perm = sym_grp[min_grp_idx].copy()
        
    return int(min_orient_bits), min_perm

#-----------------------------------------------------------------------------#

def find_min_combined_bits(combined_bits, sym_grp, num_edges, return_perm = False):
    
    # Check all compatibility tests for given arguments, to ensure two
    # individual bitmasks can fit inside combined np.uint64 bitmask
    
    num_edges = _check_num_edges(num_edges, 32)
    combined_bits = _as_uint64(combined_bits)
    
    if int(combined_bits) >= (1 << (2 * num_edges)):
       raise ValueError("Combined bitmask exceeds its allotted edge bits")

    sym_grp = _check_sym_grp(sym_grp, num_edges)
    
    min_combined_bits, min_grp_idx = _find_min_combined_bits(combined_bits, sym_grp, num_edges)
    
    if not return_perm:
        return int(min_combined_bits)
    
    if min_grp_idx == -1:   # No minimum permutation found
        min_perm = np.arange(1, num_edges + 1, dtype = np.int64)
    else:
        min_perm = sym_grp[min_grp_idx].copy()
        
    return int(min_combined_bits), min_perm

#-----------------------------------------------------------------------------#

def find_min_parity_state(parity_state, sym_grp, num_edges, return_perm = False):
    
    num_edges = _check_num_edges(num_edges, 64)
    
    min_parity_state, min_grp_idx = _find_min_parity_state(
        _as_uint64(parity_state),
        _check_sym_grp(sym_grp, num_edges), 
        num_edges
        )
    
    if not return_perm:
        return int(min_parity_state)
    
    if min_grp_idx == -1:   # No minimum permutation found
        min_perm = np.arange(1, num_edges + 1, dtype = np.int64)
    else:
        min_perm = sym_grp[min_grp_idx].copy()
        
    return int(min_parity_state), min_perm

#-----------------------------------------------------------------------------#

def find_min_orient_state(orient_state, sym_grp, num_edges, return_perm = False):
    
    num_edges = _check_num_edges(num_edges, 64)
    
    min_orient_state, min_grp_idx = _find_min_orient_state(
        _as_uint64(orient_state),
        _check_sym_grp(sym_grp, num_edges), 
        num_edges
        )
    
    if not return_perm:
        return int(min_orient_state)
    
    if min_grp_idx == -1:   # No minimum permutation found
        min_perm = np.arange(1, num_edges + 1, dtype = np.int64)
    else:
        min_perm = sym_grp[min_grp_idx].copy()
        
    return int(min_orient_state), min_perm

#-----------------------------------------------------------------------------#

def find_min_combined_state(combined_state, sym_grp, num_edges, return_perm = False):
    
    num_edges = _check_num_edges(num_edges, 32)
    
    min_combined_state, min_grp_idx = _find_min_combined_state(
        _as_uint64(combined_state),
        _check_sym_grp(sym_grp, num_edges), 
        num_edges
        )
        
    if not return_perm:
        return int(min_combined_state)
    
    if min_grp_idx == -1:   # No minimum permutation found
        min_perm = np.arange(1, num_edges + 1, dtype = np.int64)
    else:
        min_perm = sym_grp[min_grp_idx].copy()
        
    return int(min_combined_state), min_perm

#=============================================================================#
# Orientation reversed state
#=============================================================================#

#-----------------------------------------------------------------------------#
# Compiled canonicalization kernels
#-----------------------------------------------------------------------------#

@nb.njit(cache=True)
def _reverse_orient_bits(orient_bits, num_edges):
    """
        Reverse all orientation bits by taking complement
    """

    edge_mask = (np.uint64(0xFFFFFFFFFFFFFFFF) >> (64 - num_edges))

    return np.uint64(orient_bits) ^ edge_mask

#-----------------------------------------------------------------------------#
# Public-facing functions with normalized states
#-----------------------------------------------------------------------------#

def reverse_orientation(orient_state, num_edges, canonical = True, sym_grp = None):
    """
        Use signed permutation on orientation bitmask (graph ID << E bits) + (orientation bits);
        if canonical = True, canonicalize the reversed orientation using given
        graph symmetry group
    """

    num_edges = _check_num_edges(num_edges, 64)
    orient_state = _as_uint64(orient_state)

    # Extract graph ID and orientation bits
    
    graph_num, orient_bits = _decode_orient_state(orient_state, num_edges)

    # Reverse all edge orientations
    
    reversed_bits = _reverse_orient_bits(orient_bits, num_edges)

    # Optionally canonicalize the reversed decoration
    
    if canonical:

        if sym_grp is None:
            raise ValueError("sym_grp is required when canonical = True")

        sym_grp = _check_sym_grp(sym_grp, num_edges)

        reversed_bits = _find_min_orient_bits(reversed_bits, sym_grp, num_edges)

    # Reassemble the packed state using Python integers
    
    return (int(graph_num) << num_edges) | int(reversed_bits)