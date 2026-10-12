# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 07:44:00 2026

@author: cartin
"""

import numba as nb
import numpy as np

#=============================================================================#
# Encode/decode graph states
#=============================================================================#

@nb.njit(inline = "always")
def _decode_single_state(state, num_edges):

    edge_mask = (1 << num_edges) - 1
    graph_num = (state >> num_edges)
    bits = (state & edge_mask)
    
    return graph_num, bits

#-----------------------------------------------------------------------------#

@nb.njit(inline = "always")
def _encode_single_state(graph_num, bits, num_edges):
    return ((graph_num << num_edges) | bits)

#-----------------------------------------------------------------------------#

@nb.njit(inline = "always")
def decode_parity_state(parity_state, num_edges):
    """
        Parity states in form (graph ID << E bits) + (parity bits)
    """
    
    return _decode_single_state(parity_state, num_edges)

#-----------------------------------------------------------------------------#

@nb.njit(inline = "always")
def decode_orient_state(orient_state, num_edges):
    """
        Orientation states in form (graph ID << E bits) + (orientation bits)
    """
    
    return _decode_single_state(orient_state, num_edges)

#-----------------------------------------------------------------------------#

@nb.njit(inline = "always")
def decode_combined_state(combo_state, num_edges):
    """
        Combined states in form (graph ID << 2E bits) + (orientation bits) + (parity bits)
    """
    
    graph_num = (combo_state >> (2 * num_edges))
    
    edge_mask = (1 << num_edges) - 1
    orient_bits = ((combo_state >> num_edges) & edge_mask)
    parity_bits = (combo_state & edge_mask)
    
    return graph_num, orient_bits, parity_bits

#-----------------------------------------------------------------------------#

def encode_parity_state(graph_num, parity_bits, num_edges):
    """
        Parity states in form (graph ID << E bits) + (parity bits)
    """
    
    if parity_bits >= (1 << num_edges):
        raise ValueError('Parity bitmask longer than number of edges')
    
    return _encode_single_state(graph_num, parity_bits, num_edges)

#-----------------------------------------------------------------------------#

def encode_orient_state(graph_num, orient_bits, num_edges):
    """
        Orientation states in form (graph ID << E bits) + (orientation bits)
    """
    
    if orient_bits >= (1 << num_edges):
        raise ValueError('Orientation bitmask longer than number of edges')
    
    return _encode_single_state(graph_num, orient_bits, num_edges)

#-----------------------------------------------------------------------------#

def encode_combined_state(graph_num, orient_bits, parity_bits, num_edges):
    """
        Combined states in form (graph ID << 2E bits) + (orientation << E bits) + (parity bits)
    """
    
    if orient_bits >= (1 << num_edges):
        raise ValueError('Orientation bitmask longer than number of edges')
        
    if parity_bits >= (1 << num_edges):
        raise ValueError('Parity bitmask longer than number of edges')
    
    return (graph_num << (2 * num_edges)) | (orient_bits << num_edges) | parity_bits

#=============================================================================#
# Permutations of graph states
#=============================================================================#

@nb.njit(cache = True)
def permute_parity_state(parity_state, signed_perm, num_edges):
    """
        Use signed permutation on parity bitmask (graph ID << E bits) + (parity bits)
    """
    
    # Preserve the state ID by wiping the remaining E bits
    
    perm_parity_state = (parity_state >> num_edges) << num_edges
    
    # There is no check of edge orientation flip necessary, so no XOR needed,
    # simply move old parity value to its new location; because left shift here
    # can potentially exceed 64-bit limit, wrap in np.uint64().
    
    for idx, target in enumerate(signed_perm):
        target_idx = abs(target) - 1
        current_bit = (parity_state >> idx) & 1
        
        perm_parity_state |= np.uint64(current_bit << target_idx)
        
    return perm_parity_state

#-----------------------------------------------------------------------------#

@nb.njit(cache = True)
def permute_orient_state(orient_state, signed_perm, num_edges):
    """
        Use signed permutation on orientation bitmask (graph ID << E bits) + (orientation bits)
    """
    
    # Preserve the state ID by wiping the remaining E bits
    
    perm_orient_state = (orient_state >> num_edges) << num_edges
    
    # The signed permutation is 1-based and records changes in edge orientation
    # due to the symmetry element; thus, if the orientation flips, the resulting
    # orientation direction must flip as well, done by XOR on orientation value;
    # because left shift here can potentially exceed 64-bit limit, wrap in
    # np.uint64().
    
    for idx, target in enumerate(signed_perm):
        target_idx = abs(target) - 1
        current_bit = (orient_state >> idx) & 1
        
        if target < 0:
            current_bit ^= 1
            
        perm_orient_state |= np.uint64(current_bit << target_idx)
        
    return perm_orient_state

#-----------------------------------------------------------------------------#

@nb.njit(cache = True)
def permute_combined_state(combined_state, signed_perm, num_edges):
    """
        Use signed permutation on combined bitmask (graph ID << 2E bits) +
        (orientation << E bits) + (parity bits)
    """
    
    # Preserve the state ID by wiping the remaining 2E bits
    
    dbl_edge_shift = 2 * num_edges
    perm_combined_state = np.uint64(combined_state >> dbl_edge_shift) << dbl_edge_shift
    
    # The signed permutation is 1-based and records changes in edge orientation
    # due to the symmetry element; thus, if the orientation flips, the resulting
    # orientation direction must flip as well, done by XOR on orientation value;
    # because left shift here can potentially exceed 64-bit limit, wrap in
    # np.uint64().
    
    for idx, target in enumerate(signed_perm):
        tgt_parity_idx = np.uint64(abs(target) - 1)
        tgt_orient_idx = tgt_parity_idx + num_edges
        
        curr_orient_bit = np.uint64((combined_state >> (idx + num_edges)) & 1)
        curr_parity_bit = np.uint64((combined_state >> idx) & 1)
        
        if target < 0:
            curr_orient_bit ^= 1
            
        perm_combined_state |= (np.uint64(curr_orient_bit) << tgt_orient_idx)
        perm_combined_state |= (np.uint64(curr_parity_bit) << tgt_parity_idx)
        
    return perm_combined_state

#=============================================================================#
# Find lexicographic minimum graph state
#=============================================================================#

@nb.njit(cache = True)
def find_min_parity_state(parity_state, sym_grp, num_edges):
    
    min_parity_state = parity_state
    
    for idx in range(sym_grp.shape[0]):
        temp_state = permute_parity_state(parity_state, sym_grp[idx], num_edges)
        if temp_state < min_parity_state:
            min_parity_state = temp_state
            
    return min_parity_state

#-----------------------------------------------------------------------------#

@nb.njit(cache = True)
def find_min_orient_state(orient_state, sym_grp, num_edges):
    
    min_orient_state = orient_state
    
    for idx in range(sym_grp.shape[0]):
        temp_state = permute_orient_state(orient_state, sym_grp[idx], num_edges)
        if temp_state < min_orient_state:
            min_orient_state = temp_state
            
    return min_orient_state

#-----------------------------------------------------------------------------#

@nb.njit(cache = True)
def find_min_combined_state(combined_state, sym_grp, num_edges):
    
    min_combined_state = combined_state
    
    for idx in range(sym_grp.shape[0]):
        temp_state = permute_combined_state(combined_state, sym_grp[idx], num_edges)
        if temp_state < min_combined_state:
            min_combined_state = temp_state
            
    return min_combined_state

#=============================================================================#
# Orientation reversed state
#=============================================================================#

@nb.njit(cache = True)
def reverse_orientation(orient_state, num_edges, canonical = True, sym_grp = None):
    """
        Use signed permutation on orientation bitmask (graph ID << E bits) + (orientation bits)
    """
    
    pass
