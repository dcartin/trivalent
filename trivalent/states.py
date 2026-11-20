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
    r"""
        Combine orientation and parity bitmasks into a single integer.
        
        Parameters
        ----------
        orient_bits : int
            Orientation bitmask containing $|E|$ bits.
        
        parity_bits : int
            Parity bitmask containing $|E|$ bits.
        
        num_edges : int
            Number of graph edges $|E|$, between 1 and 32.
        
        Returns
        -------
        numpy.uint64
            Combined decoration bitmask.
        
        Notes
        -----
        The combined bitmask is
        
        $$
        c = 2^{|E|}d + p,
        $$
        
        where $d$ and $p$ are the orientation and parity
        bitmasks, respectively.
        
        The parity bits occupy the lowest $|E|$ positions,
        followed by the orientation bits.
        
        No graph ID is included in the result.
        
        The input bitmasks are assumed to be non-negative
        and to fit within $|E|$ bits.
    """

    # Ensure that given arguments are integers
    
    orient_bits = operator.index(orient_bits)
    parity_bits = operator.index(parity_bits)
    num_edges = _check_num_edges(num_edges, 32)
    
    return _combine_bits(orient_bits, parity_bits, num_edges)

#-----------------------------------------------------------------------------#

def separate_bits(combined_bits, num_edges):
    r"""
        Separate a combined decoration bitmask into its components.
        
        Parameters
        ----------
        combined_bits : int
            Combined orientation and parity bitmask containing
            $2|E|$ bits.
        
        num_edges : int
            Number of graph edges $|E|$, between 1 and 32.
        
        Returns
        -------
        orient_bits : numpy.uint64
            Orientation bitmask extracted from the upper
            $|E|$ decoration bits.
        
        parity_bits : numpy.uint64
            Parity bitmask extracted from the lower
            $|E|$ decoration bits.
        
        Notes
        -----
        For a valid combined bitmask
        
        $$
        c = 2^{|E|}d + p,
        $$
        
        the components are recovered using
        
        $$
        d = \left\lfloor \frac{c}{2^{|E|}} \right\rfloor,
        \qquad
        p = c \bmod 2^{|E|}.
        $$
        
        No graph ID is extracted.
        
        This function is the inverse of `combine_bits()`
        for bitmasks within the permitted ranges.
    """
    
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
    r"""
        Decode a parity state into its graph ID and parity bitmask.
        
        Parameters
        ----------
        parity_state : int
            Encoded parity state.
        
        num_edges : int
            Number of graph edges $|E|$, between 1 and 64.
        
        Returns
        -------
        graph_num : numpy.uint64
            Graph ID extracted from the encoded state.
        
        parity_bits : numpy.uint64
            Parity bitmask occupying the lowest $|E|$ bits.
        
        Notes
        -----
        This is the inverse of `encode_parity_state()` for
        valid encoded states.
        
        When $|E|=64$, the graph ID is zero.
    """

    return _decode_parity_state(_as_uint64(parity_state), _check_num_edges(num_edges, 64))

#-----------------------------------------------------------------------------#

def decode_orient_state(orient_state, num_edges):
    r"""
        Decode an orientation state into its graph ID and bitmask.
        
        Parameters
        ----------
        orient_state : int
            Encoded orientation state.
        
        num_edges : int
            Number of graph edges $|E|$, between 1 and 64.
        
        Returns
        -------
        graph_num : numpy.uint64
            Graph ID extracted from the encoded state.
        
        orient_bits : numpy.uint64
            Orientation bitmask occupying the lowest $|E|$ bits.
        
        Notes
        -----
        This is the inverse of `encode_orient_state()` for
        valid encoded states.
        
        When $|E|=64$, the graph ID is zero.
    """

    return _decode_orient_state(_as_uint64(orient_state), _check_num_edges(num_edges, 64))

#-----------------------------------------------------------------------------#

def decode_combined_state(combined_state, num_edges, separate = True):
    r"""
        Decode a combined state into its graph ID and decorations.
        
        Parameters
        ----------
        combined_state : int
            Encoded combined state.
        
        num_edges : int
            Number of graph edges $|E|$, between 1 and 32.
        
        separate : bool, default=True
            If `True`, return the orientation and parity
            bitmasks separately.
        
            If `False`, return a single combined decoration bitmask.
        
        Returns
        -------
        tuple of numpy.uint64
            If `separate=True`, return
            `(graph_num, orient_bits, parity_bits)`.
        
            Otherwise, return `(graph_num, combined_bits)`,
            where
        
            $$
            \mathrm{combined\_bits} = 2^{|E|}d + p.
            $$
        
        Notes
        -----
        This is the inverse of `encode_combined_state()` for
        valid encoded states.
        
        When $|E|=32$, the graph ID is zero.
    """

    graph_num, orient_bits, parity_bits = _decode_combined_state(_as_uint64(combined_state), _check_num_edges(num_edges, 32))

    if separate:
        return graph_num, orient_bits, parity_bits
    else:
        return graph_num, (orient_bits << num_edges) | parity_bits

#-----------------------------------------------------------------------------#

def encode_parity_state(graph_num, parity_bits, num_edges):
    r"""
        Encode a graph ID and parity bitmask as a single integer.
        
        Parameters
        ----------
        graph_num : int
            Non-negative graph ID supplied by the caller.
        
        parity_bits : int
            Parity bitmask containing $|E|$ bits.
        
        num_edges : int
            Number of graph edges $|E|$, between 1 and 64.
        
        Returns
        -------
        int
            Encoded parity state.
        
        Raises
        ------
        ValueError
            If either input is negative, or the encoded state
            would exceed 64 bits.
        
        Notes
        -----
        The encoding is
        
        $$
        s_p = 2^{|E|}g + p,
        $$
        
        where $g$ is the graph ID and $p$ is the parity bitmask.
        
        Bit $e$ of $p$ represents edge index $e$.
        
        No parity admissibility conditions are checked.
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
    r"""
        Encode a graph ID and orientation bitmask as a single integer.
        
        Parameters
        ----------
        graph_num : int
            Non-negative graph ID supplied by the caller.
        
        orient_bits : int
            Orientation bitmask containing $|E|$ bits.
        
        num_edges : int
            Number of graph edges $|E|$, between 1 and 64.
        
        Returns
        -------
        int
            Encoded orientation state.
        
        Raises
        ------
        ValueError
            If either input is negative, or the encoded state
            would exceed 64 bits.
        
        Notes
        -----
        The encoding is
        
        $$
        s_d = 2^{|E|}g + d,
        $$
        
        where $g$ is the graph ID and $d$ is the orientation bitmask.
        
        Bit $e$ of $d$ represents the orientation of edge index $e$
        relative to its reference direction.
        
        No orientation admissibility conditions are checked.
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
    r"""
        Encode a graph ID, orientation bitmask, and parity bitmask.
        
        Parameters
        ----------
        graph_num : int
            Non-negative graph ID supplied by the caller.
        
        orient_bits : int
            Orientation bitmask containing $|E|$ bits.
        
        parity_bits : int
            Parity bitmask containing $|E|$ bits.
        
        num_edges : int
            Number of graph edges $|E|$, between 1 and 32.
        
        Returns
        -------
        int
            Encoded combined state.
        
        Raises
        ------
        ValueError
            If any input is negative, or the encoded state
            would exceed 64 bits.
        
        Notes
        -----
        The encoding is
        
        $$
        s_c = 2^{2|E|}g + 2^{|E|}d + p,
        $$
        
        where $g$ is the graph ID, $d$ is the orientation
        bitmask, and $p$ is the parity bitmask.
        
        The parity bits occupy the lowest $|E|$ positions,
        followed by the orientation bits, then the graph ID.
        
        No decoration admissibility conditions are checked.
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
    r"""
        Encode a graph ID and combined decoration bitmask.
        
        Parameters
        ----------
        graph_num : int
            Non-negative graph ID supplied by the caller.
        
        combined_bits : int
            Combined orientation and parity bitmask containing
            $2|E|$ bits.
        
        num_edges : int
            Number of graph edges $|E|$, between 1 and 32.
        
        Returns
        -------
        int
            Encoded combined state.
        
        Notes
        -----
        The combined decoration bitmask has the form
        
        $$
        c = 2^{|E|}d + p,
        $$
        
        where $d$ and $p$ are orientation and parity bitmasks.
        
        The encoded state is
        
        $$
        s_c = 2^{2|E|}g + c.
        $$
        
        This function is equivalent to separating the
        decoration bitmasks and calling `encode_combined_state()`.
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
    r"""
        Apply a signed edge permutation to a parity bitmask.
        
        Parameters
        ----------
        parity_bits : int
            Parity bitmask containing $|E|$ bits.
        
        signed_perm : array_like of int
            Signed edge permutation of length $|E|$, using
            one-based edge labels.
        
        num_edges : int
            Number of graph edges $|E|$, between 1 and 64.
        
        Returns
        -------
        int
            Permuted parity bitmask.
        
        Notes
        -----
        For each original edge index $i$, let $q$ be a signed
        edge permutation, with
        
        $$
        q[i] = s_i(j+1),
        $$
        
        where $j$ is the image edge index and $s_i = \pm 1$.
        
        The parity bits $p_i$ transform according to
        
        $$
        p'_j = p_i.
        $$
        
        The signs of the permutation entries do not affect
        the parity values.
        
        The input bitmask and signed permutation are validated.
        The permutation is not required to be a graph automorphism.
        
        No graph ID is included in the result.
    """
    
    num_edges = _check_num_edges(num_edges, 64)
    
    parity_bits = _check_edge_bits(parity_bits, num_edges)
    signed_perm = _check_signed_perm(signed_perm, num_edges)
    
    return int(_permute_parity_bits(parity_bits, signed_perm, num_edges))
    
#-----------------------------------------------------------------------------#

def permute_orient_bits(orient_bits, signed_perm, num_edges):
    r"""
    Apply a signed edge permutation to an orientation bitmask.
    
    Parameters
    ----------
    orient_bits : int
        Orientation bitmask containing $|E|$ bits.
    
    signed_perm : array_like of int
        Signed edge permutation of length $|E|$, using
        one-based edge labels.
    
    num_edges : int
        Number of graph edges $|E|$, between 1 and 64.
    
    Returns
    -------
    int
        Permuted orientation bitmask.
    
    Notes
    -----
    For each original edge index $i$, let $q$ be a signed
    edge permutation, with
    
    $$
    q[i] = s_i(j+1),
    $$
    
    where $j$ is the image edge index and $s_i = \pm 1$.
    
    The orientation bits $d_i$ transform according to
    
    $$
    d'_j =
    \begin{cases}
    d_i, & s_i = +1,\\
    1-d_i, & s_i = -1.
    \end{cases}
    $$
    
    Thus, a negative permutation entry complements the
    orientation bit when moving it to its new position.
    
    The input bitmask and signed permutation are validated.
    The permutation is not required to be a graph automorphism.
    
    No graph ID is included in the result.
    """
    
    num_edges = _check_num_edges(num_edges, 64)
    
    orient_bits = _check_edge_bits(orient_bits, num_edges)
    signed_perm = _check_signed_perm(signed_perm, num_edges)
    
    return int(_permute_orient_bits(orient_bits, signed_perm, num_edges))
    
#-----------------------------------------------------------------------------#
    
def permute_combined_bits(combined_bits, signed_perm, num_edges):
    r"""
        Apply a signed edge permutation to combined decorations.
        
        The same permutation acts simultaneously on the
        orientation and parity bitmasks.
        
        Parameters
        ----------
        combined_bits : int
            Combined decoration bitmask containing $2|E|$ bits.
        
        signed_perm : array_like of int
            Signed edge permutation of length $|E|$, using
            one-based edge labels.
        
        num_edges : int
            Number of graph edges $|E|$, between 1 and 32.
        
        Returns
        -------
        numpy.uint64
            Permuted combined decoration bitmask.
        
        Notes
        -----
        The combined bitmask is represented as
        
        $$
        c = 2^{|E|}d+p,
        $$
        
        where $d$ and $p$ are the orientation and parity
        bitmasks, respectively.
        
        For a signed permutation $q[i]=s_i(j+1)$,
        the decorations transform according to
        
        $$
        p'_j=p_i,
        $$
        
        and
        
        $$
        d'_j =
        \begin{cases}
        d_i, & s_i=+1,\\
        1-d_i, & s_i=-1.
        \end{cases}
        $$
        
        The result is the combined bitmask
        
        $$
        c'=2^{|E|}d'+p'.
        $$
        
        The signed permutation is validated, but is not
        required to be a graph automorphism.
        
        The combined bitmask is assumed to be non-negative
        and to fit within $2|E|$ bits.
        
        No graph ID is included in the result.
    """
    
    num_edges = _check_num_edges(num_edges, 32)
    
    signed_perm = _check_signed_perm(signed_perm, num_edges)
    
    return _permute_combined_bits(combined_bits, signed_perm, num_edges)

#-----------------------------------------------------------------------------#

def permute_parity_state(parity_state, signed_perm, num_edges):
    r"""
        Apply a signed edge permutation to an encoded parity state.
        
        Parameters
        ----------
        parity_state : int
            Encoded parity state containing a graph ID and
            $|E|$ parity bits.
        
        signed_perm : array_like of int
            Signed edge permutation of length $|E|$, using
            one-based edge labels.
        
        num_edges : int
            Number of graph edges $|E|$, between 1 and 64.
        
        Returns
        -------
        numpy.uint64
            Encoded state with permuted parity bits.
        
        Notes
        -----
        For an original edge index $i$ mapped to edge $j$,
        the parity bits transform as
        
        $$
        p'_j = p_i.
        $$
        
        The signs of the permutation entries are ignored.
        
        The graph ID is preserved, and only the edge
        decoration is transformed.
        
        The supplied permutation is validated, but is not
        required to be a positive graph automorphism.
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
    r"""
        Apply a signed edge permutation to an encoded orientation state.
        
        Parameters
        ----------
        orient_state : int
            Encoded orientation state containing a graph ID and
            $|E|$ orientation bits.
        
        signed_perm : array_like of int
            Signed edge permutation of length $|E|$, using
            one-based edge labels.
        
        num_edges : int
            Number of graph edges $|E|$, between 1 and 64.
        
        Returns
        -------
        numpy.uint64
            Encoded state with permuted orientation bits.
        
        Notes
        -----
        For a signed permutation $q[i]=s_i(j+1)$,
        the orientation bits transform as
        
        $$
        d'_j =
        \begin{cases}
        d_i, & s_i=+1,\\
        1-d_i, & s_i=-1.
        \end{cases}
        $$
        
        Thus, a negative permutation entry complements
        the corresponding orientation bit.
        
        The graph ID is preserved, and only the edge
        decoration is transformed.
        
        The supplied permutation is validated, but is not
        required to be a positive graph automorphism.
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
    r"""
        Apply a signed edge permutation to an encoded combined state.
        
        The same permutation acts simultaneously on both
        orientation and parity decorations.
        
        Parameters
        ----------
        combined_state : int
            Encoded combined state containing a graph ID,
            $|E|$ orientation bits, and $|E|$ parity bits.
        
        signed_perm : array_like of int
            Signed edge permutation of length $|E|$, using
            one-based edge labels.
        
        num_edges : int
            Number of graph edges $|E|$, between 1 and 32.
        
        Returns
        -------
        numpy.uint64
            Encoded state with both decorations permuted.
        
        Notes
        -----
        The combined state is encoded as
        
        $$
        s_c = 2^{2|E|}g + 2^{|E|}d + p,
        $$
        
        where $g$ is the graph ID, $d$ is the orientation
        bitmask, and $p$ is the parity bitmask.
        
        Both decorations undergo the same edge permutation.
        Negative permutation entries complement orientation
        bits but do not change parity values.
        
        The graph ID is preserved.
        
        The supplied permutation is validated, but is not
        required to be a positive graph automorphism.
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
    r"""
        Find the minimum parity bitmask under a symmetry group.
        
        Parameters
        ----------
        parity_bits : int
            Parity bitmask containing $|E|$ bits.
        
        sym_grp : numpy.ndarray
            Non-empty array of shape $(N_{\mathrm{sym}}, |E|)$
            containing signed edge permutations.
        
        num_edges : int
            Number of graph edges $|E|$, between 1 and 64.
        
        return_perm : bool, default=False
            If `True`, also return a permutation attaining
            the minimum.
        
        Returns
        -------
        int or tuple of (int, numpy.ndarray)
            Numerically smallest parity bitmask under the
            supplied group action.
        
            If `return_perm=True`, also return a signed edge
            permutation attaining this minimum.
        
        Notes
        -----
        The minimum is calculated over the orbit of the
        original parity bitmask.
        
        If no strictly smaller bitmask is found, the
        identity permutation is returned when requested.
        
        No graph ID is included.
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
    r"""
        Find the minimum orientation bitmask under a symmetry group.
        
        Parameters
        ----------
        orient_bits : int
            Orientation bitmask containing $|E|$ bits.
        
        sym_grp : numpy.ndarray
            Non-empty array of shape $(N_{\mathrm{sym}}, |E|)$
            containing signed edge permutations.
        
        num_edges : int
            Number of graph edges $|E|$, between 1 and 64.
        
        return_perm : bool, default=False
            If `True`, also return a permutation attaining
            the minimum.
        
        Returns
        -------
        int or tuple of (int, numpy.ndarray)
            Numerically smallest orientation bitmask under
            the supplied group action.
        
            If `return_perm=True`, also return a signed edge
            permutation attaining this minimum.
        
        Notes
        -----
        The signs of the permutations are used to account
        for reversals of edge reference directions.
        
        If no strictly smaller bitmask is found, the
        identity permutation is returned when requested.
        
        Global reversal of all edge orientations is not
        included as an additional equivalence operation.
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
    r"""
        Find the minimum combined bitmask under a symmetry group.
        
        Both orientation and parity decorations are transformed
        using the same signed edge permutation.
        
        Parameters
        ----------
        combined_bits : int
            Combined decoration bitmask containing $2|E|$ bits.
        
        sym_grp : numpy.ndarray
            Non-empty array of shape $(N_{\mathrm{sym}}, |E|)$
            containing signed edge permutations.
        
        num_edges : int
            Number of graph edges $|E|$, between 1 and 32.
        
        return_perm : bool, default=False
            If `True`, also return a permutation attaining
            the minimum.
        
        Returns
        -------
        int or tuple of (int, numpy.ndarray)
            Numerically smallest combined bitmask under the
            supplied group action.
        
            If `return_perm=True`, also return a signed edge
            permutation attaining this minimum.
        
        Notes
        -----
        The combined bitmask is
        
        $$
        c = 2^{|E|}d + p,
        $$
        
        where $d$ and $p$ are the orientation and parity
        bitmasks.
        
        Numerical minimization is equivalent to lexicographic
        minimization of $(d,p)$, with orientation taking
        precedence over parity.
        
        The two decorations are not minimized independently.
        
        If no strictly smaller bitmask is found, the
        identity permutation is returned when requested.
    """
    
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
    r"""
        Find the minimum encoded parity state under a symmetry group.
        
        Parameters
        ----------
        parity_state : int
            Encoded parity state containing a graph ID and
            $|E|$ parity bits.
        
        sym_grp : numpy.ndarray
            Non-empty array of shape $(N_{\mathrm{sym}}, |E|)$
            containing signed edge permutations.
        
        num_edges : int
            Number of graph edges $|E|$, between 1 and 64.
        
        return_perm : bool, default=False
            If `True`, also return a permutation attaining
            the minimum.
        
        Returns
        -------
        int or tuple of (int, numpy.ndarray)
            Minimum encoded parity state, optionally accompanied
            by a minimizing signed edge permutation.
        
        Notes
        -----
        Only the parity bitmask is minimized. The encoded
        graph ID is preserved.
        
        If no strictly smaller state is found, the identity
        permutation is returned when requested.
    """
    
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
    r"""
        Find the minimum encoded orientation state under a symmetry group.
        
        Parameters
        ----------
        orient_state : int
            Encoded orientation state containing a graph ID
            and $|E|$ orientation bits.
        
        sym_grp : numpy.ndarray
            Non-empty array of shape $(N_{\mathrm{sym}}, |E|)$
            containing signed edge permutations.
        
        num_edges : int
            Number of graph edges $|E|$, between 1 and 64.
        
        return_perm : bool, default=False
            If `True`, also return a permutation attaining
            the minimum.
        
        Returns
        -------
        int or tuple of (int, numpy.ndarray)
            Minimum encoded orientation state, optionally
            accompanied by a minimizing signed edge permutation.
        
        Notes
        -----
        Only the orientation bitmask is minimized. The encoded
        graph ID is preserved.
        
        Signed permutation entries account for changes in
        edge reference directions.
        
        Global orientation reversal is not included as an
        additional equivalence operation.
    """
    
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
    r"""
        Find the minimum encoded combined state under a symmetry group.
        
        Parameters
        ----------
        combined_state : int
            Encoded state containing a graph ID, orientation
            bits, and parity bits.
        
        sym_grp : numpy.ndarray
            Non-empty array of shape $(N_{\mathrm{sym}}, |E|)$
            containing signed edge permutations.
        
        num_edges : int
            Number of graph edges $|E|$, between 1 and 32.
        
        return_perm : bool, default=False
            If `True`, also return a permutation attaining
            the minimum.
        
        Returns
        -------
        int or tuple of (int, numpy.ndarray)
            Minimum encoded combined state, optionally
            accompanied by a minimizing signed edge permutation.
        
        Notes
        -----
        The graph ID is preserved.
        
        Both decorations are transformed using the same
        signed edge permutation.
        
        Numerical minimization gives priority to the orientation
        bitmask, followed by the parity bitmask among equivalent
        minimum orientations.
        
        The decorations are not minimized independently.
    """
    
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
    r"""
        Reverse all edge orientations in an encoded orientation state.
        
        The operation complements every orientation bit while
        preserving the encoded graph ID.
        
        Optionally, the reversed state is canonicalized under
        a supplied group of signed edge permutations.
        
        Parameters
        ----------
        orient_state : int
            Encoded orientation state containing a graph ID
            and $|E|$ orientation bits.
        
        num_edges : int
            Number of graph edges $|E|$, between 1 and 64.
        
        canonical : bool, default=True
            If `True`, find the minimum orientation bitmask
            after reversing all edge orientations.
        
        sym_grp : numpy.ndarray, optional
            Non-empty array of signed edge permutations of
            shape $(N_{\mathrm{sym}}, |E|)$.
        
            Required when `canonical=True`.
        
        Returns
        -------
        int
            Encoded state with reversed orientation bits,
            optionally canonicalized.
        
        Raises
        ------
        ValueError
            If `canonical=True` and `sym_grp` is not supplied,
            or if the state or symmetry group is invalid.
        
        Notes
        -----
        Global orientation reversal acts on the bitmask $d$ as
        
        $$
        R(d) = d \oplus (2^{|E|}-1).
        $$
        
        The encoded graph ID is preserved.
        
        If `canonical=True`, the returned orientation bitmask is
        
        $$
        d' = \min_{h\in H} h\cdot R(d),
        $$
        
        where $H$ is the supplied symmetry group.
        
        Global orientation reversal is an involution before
        canonicalization.
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

        reversed_bits, _ = _find_min_orient_bits(reversed_bits, sym_grp, num_edges)

    # Reassemble the packed state using Python integers
    
    return (int(graph_num) << num_edges) | int(reversed_bits)