# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 06:47:05 2026

@author: cartin
"""

import numba as nb
import numpy as np

from collections import Counter, deque
from trivalent.states import (
        find_min_parity_state,
        find_min_orient_state,
        find_min_combined_state,
        permute_parity_bits,
        permute_orient_bits,
        permute_combined_bits,
        permute_parity_state,
        permute_orient_state,
        permute_combined_state
    )

#=============================================================================#
# Complied Numba functions
#=============================================================================#

@nb.njit(inline = 'always')
def _init_assign(idx, shift, v_A, v_B, edge_list_A, edge_list_B, cyc_A, cyc_B,
                edge_map, edge_visit, edge_queue, vert_map, vert_visit, vert_queue, q_tail):
    
    edge_A_idx = cyc_A[v_A, idx]
    edge_B_idx = cyc_B[v_B, (idx - shift) % 3]
    
    edge_A = edge_list_A[edge_A_idx]
    edge_B = edge_list_B[edge_B_idx]
    
    v_A_is_start = (v_A == edge_A[0])
    v_B_is_start = (v_B == edge_B[0])
    
    sign = 1 if (v_A_is_start == v_B_is_start) else -1
    
    edge_map[edge_A_idx] = sign * (edge_B_idx + 1)
    edge_visit[edge_B_idx] = True
    
    other_v_A = edge_A[1] if v_A_is_start else edge_A[0]
    other_v_B = edge_B[1] if v_B_is_start else edge_B[0]
    
    vert_map[other_v_A] = other_v_B
    vert_visit[other_v_B] = True
    
    # Matching graph A edge, vertex in queue, so we know which edge used to
    # arrive at which vertex in BFS
    
    edge_queue[q_tail] = edge_A_idx
    vert_queue[q_tail] = other_v_A
    q_tail += 1
    
    return q_tail

#-----------------------------------------------------------------------------#

@nb.njit(inline = 'always')
def _continue_assign(num_vert, slot_A, slot_B, offset, curr_v_A, curr_v_B,
                    edge_list_A, edge_list_B, cyc_A, cyc_B,
                    edge_map, edge_visit, edge_queue, vert_map, vert_visit, vert_queue, q_tail):
    
    edge_A_idx = cyc_A[curr_v_A, (slot_A + offset) % 3]
    edge_B_idx = cyc_B[curr_v_B, (slot_B + offset) % 3]
    
    edge_A = edge_list_A[edge_A_idx]
    edge_B = edge_list_B[edge_B_idx]
    
    curr_v_A_is_start = (curr_v_A == edge_A[0])
    curr_v_B_is_start = (curr_v_B == edge_B[0])
    
    sign = 1 if (curr_v_A_is_start == curr_v_B_is_start) else -1
    tgt_idx = sign * (edge_B_idx + 1)
    
    if edge_map[edge_A_idx] == 0:
        
        # Ensure that we have not already tried to map edge_B to another edge
        
        if edge_visit[edge_B_idx]:
            return False, q_tail
        
        edge_map[edge_A_idx] = tgt_idx
        edge_visit[edge_B_idx] = True
        
    elif edge_map[edge_A_idx] != tgt_idx:         # Inconsistent edge map
        return False, q_tail
    
    other_v_A = edge_A[1] if curr_v_A_is_start else edge_A[0]
    other_v_B = edge_B[1] if curr_v_B_is_start else edge_B[0]
    
    if vert_map[other_v_A] == num_vert:
        
        # Ensure we have not already tried to map other_v_B to another vertex
        
        if vert_visit[other_v_B]:
            return False, q_tail
        
        vert_map[other_v_A] = other_v_B
        vert_visit[other_v_B] = True
        
        # Put new vertex, edge into queue at tail position
        
        edge_queue[q_tail] = edge_A_idx
        vert_queue[q_tail] = other_v_A
        q_tail += 1
        
    elif vert_map[other_v_A] != other_v_B:    # Inconsistent vertex map
        return False, q_tail

    return True, q_tail

#-----------------------------------------------------------------------------#

@nb.njit(cache = True)
def _graph_isomorphism(start_vert_A, vert_choices_B, edge_list_A, cyc_A, edge_list_B, cyc_B,
                      find_all, symmetry_buffer):
    
    # Assumption is that graph edge numbers for A, B have been checked to be equal
    
    num_edges = len(edge_list_A)
    num_vert = 2 * num_edges // 3
    count_perm = 0
    
    # Pre-allocate tracking buffer to maximize performance. Note that arrays
    # for edges use 1-based entries, since we determine signed permutations;
    # arrays for verts use num_vert as sentinel for unfilled entry
    #
    #   edge_queue, vert_queue : place objects *in A* here when reached by BFS
    #   edge_map, vert_map : mapping[idx_A] = idx_B gives edge mapping A -> B
    #   edge_visit, vert_visit : T/F for whether BFS has reached object *in B*
    #
    # Also, an edge is placed in edge_queue only as a way of knowing the
    # incoming edge into a new vertex. Thus, the number of edges placed there
    # is one less than the total number of vertices (since the start vertex is
    # never encountered by traveling along an edge). This is why we can set
    # the lengths of the two queues to be |V| - 1 < |E|.
    
    max_queue_size = num_vert - 1
    edge_queue = np.zeros(max_queue_size, dtype = np.int8)
    vert_queue = np.full(max_queue_size, num_vert, dtype = np.uint8)
    
    edge_map = np.zeros(num_edges, dtype = np.int8)
    vert_map = np.full(num_vert, num_vert, dtype = np.uint8)
    
    edge_visit = np.zeros(num_edges, dtype = nb.boolean)
    vert_visit = np.zeros(num_vert, dtype = nb.boolean)
    
    # Initialize BFS from starting vertex in A, and consider all possible
    # vertex maps to B, along with all three ways to map its incident edges to 
    # those in B
    
    for v_B in vert_choices_B:
        for shift in range(3):
            
            # Clean reset of all tracking buffers, then initialize for given
            # choices
            
            edge_queue[:] = 0
            vert_queue[:] = num_vert
            
            edge_map[:] = 0
            vert_map[:] = num_vert
            
            edge_visit[:] = False
            vert_visit[:] = False
            
            vert_map[start_vert_A] = v_B
            vert_visit[v_B] = True
            
            queue_head = 0
            queue_tail = 0
            
            # Assign initial edge maps using given shift in cyclic order, push
            # these edges, opposite side vertices in A onto queue
            
            for iii in range(3):
                queue_tail = _init_assign(iii, shift, start_vert_A, v_B, edge_list_A, edge_list_B, cyc_A, cyc_B,
                                         edge_map, edge_visit, edge_queue, vert_map, vert_visit, vert_queue, queue_tail)
            
            # Go through and process queue entries until either graph is
            # completely, or a contradiction is reached
            
            while queue_head < queue_tail:
                curr_v_A = vert_queue[queue_head]
                curr_v_B = vert_map[curr_v_A]
                
                in_edge_A_idx = edge_queue[queue_head]
                in_edge_B_idx = abs(edge_map[in_edge_A_idx]) - 1
                
                queue_head += 1
                
                # Find where incoming edges are in respective cyclic orders
                
                slot_A, slot_B = -1, -1
                valid = True
                
                for iii in range(3):
                    if cyc_A[curr_v_A, iii] == in_edge_A_idx:
                        slot_A = iii
                        break
                    
                for iii in range(3):
                    if cyc_B[curr_v_B, iii] == in_edge_B_idx:
                        slot_B = iii
                        break
                    
                # If cannot find edges for any reason, stop while loop, so that
                # we pass on to next vertex choice
                
                if (slot_A == -1) or (slot_B == -1):
                    valid = False
                    break
                
                # Continue BFS search; the logic is the same here as with setting
                # the initial edges, vertices, but now in addition we need to
                # check if there are contradictions in the mapping
                
                for offset in (1, 2):
                    valid, queue_tail = _continue_assign(num_vert, slot_A, slot_B, offset, curr_v_A, curr_v_B,
                                                        edge_list_A, edge_list_B, cyc_A, cyc_B,
                                                        edge_map, edge_visit, edge_queue, vert_map, 
                                                        vert_visit, vert_queue, queue_tail)
                    
                    if not valid:       # Inconsistency in mapping (break out of offset loop)
                        break
                    
                if not valid:       # Inconsistency in mapping (break out of queue loop)
                    break
                
            # Test whether any issues arose and all edges have been mapped
            
            if valid and np.count_nonzero(edge_map) == num_edges:
            
                symmetry_buffer[count_perm, :] = edge_map
                count_perm += 1
                
                if not find_all:
                    return 1        # Fast exit when graphs A, B are iso
                
                # Is symmetry buffer overflowing?
                
                if count_perm >= len(symmetry_buffer):
                    return count_perm

    return count_perm

#-----------------------------------------------------------------------------#

@nb.njit(cache = True)
def _compose_signed_perm(perm_A, perm_B):
    """
        With 1-based signed permutations, finds total permutation, assuming
        they are applied in the order (perm_B o perm_A)(x) = perm_B(perm_A(x)).
    """
    
    num_elem = len(perm_A)
    final_perm = np.zeros(num_elem, dtype = np.int8)
    
    for idx in range(num_elem):
        aaa = abs(perm_A[idx]) - 1
        aaa_sign = 1 if perm_A[idx] > 0 else -1
        
        bbb = abs(perm_B[aaa]) - 1
        bbb_sign = 1 if perm_B[aaa] > 0 else -1
        
        final_perm[idx] = aaa_sign * bbb_sign * (bbb + 1)
        
    return final_perm  

#-----------------------------------------------------------------------------#
# Union-find data structure
#-----------------------------------------------------------------------------#

@nb.njit(cache=True)
def _uf_find(i, parent):
    # Find the root with path compression
    root = i
    while parent[root] != root:
        root = parent[root]
        
    # Path compression step
    curr = i
    while curr != root:
        nxt = parent[curr]
        parent[curr] = root
        curr = nxt
        
    return root

#-----------------------------------------------------------------------------#

@nb.njit(cache=True)
def _uf_union(i, j, parent, rank):
    root_i = _uf_find(i, parent)
    root_j = _uf_find(j, parent)
    
    if root_i != root_j:
        if rank[root_i] < rank[root_j]:
            parent[root_i] = root_j
        elif rank[root_i] > rank[root_j]:
            parent[root_j] = root_i
        else:
            parent[root_j] = root_i
            rank[root_i] += 1
            
#-----------------------------------------------------------------------------#

@nb.njit(cache=True)
def _get_non_iso_edges(num_edges, sym_grp):
    """
        sym_grp must be a 2D numpy array of shape (num_syms, num_edges)
    """
    parent = np.arange(num_edges, dtype=np.int32)
    rank = np.zeros(num_edges, dtype=np.int32)
    
    num_syms = sym_grp.shape[0]
    
    # Process symmetries
    for sym_idx in range(num_syms):
        for src in range(num_edges):
            # Using 1-based indexing, convert to 0-based
            tgt = abs(sym_grp[sym_idx, src]) - 1
            _uf_union(src, tgt, parent, rank)
            
    # Collect unique roots
    roots = np.zeros(num_edges, dtype=np.int32)
    count = 0
    for i in range(num_edges):
        if parent[i] == i:
            roots[count] = i
            count += 1
            
    return roots[:count].copy() # Return only the populated slice

#=============================================================================#
# Graph class
#=============================================================================#


class Graph:
    """
    Represent a trivalent graph with a specified cyclic ordering of edges.

    A Graph object represents a 3-regular graph together with the
    counterclockwise cyclic ordering of incident edges at each vertex.
    These cyclic orders determine a combinatorial embedding of the graph
    in an orientable surface.

    The graph is constructed from an implicitly ordered edge list:
    the order in which edges appear determines their cyclic ordering
    at each vertex.

    Notes
    -----
    - Vertex and edge indices are zero-based.
    - The endpoints of each edge determine its reference direction,
      used when constructing signed edge permutations.
    - The graph may be modified in place by Pachner moves.
    - Face information is calculated when needed.
    - Construction does not establish planarity or 3-connectivity.

    Examples
    --------
    Create a triangular prism:

    >>> G = Graph.create_prism(3)
    >>> G.num_vert
    6
    >>> G.num_edges
    9
    >>> G.num_faces
    5
    """
    
    DEFAULT_MAX_VERT = 50       # Maximum number of vertices for any possible graph
    DEFAULT_MAX_SYM = 60        # Maximum number of graph symmetries (for dodecahedron)
    
    #-------------------------------------------------------------------------#

    def __init__(self, ordered_edge_list = None, max_num_vert = None, validate = True):
        r"""
        Initialize a graph from an implicitly ordered edge list.
        
        Parameters
        ----------
        ordered_edge_list : array_like of int, optional
            Array of shape $(|E|, 2)$, containing pairs of vertex
            indices $(u,v)$ specifying the graph edges.
        
            Vertex indices run from $0$ to $|V|-1$. The order in
            which edges occur in the list determines the
            counterclockwise cyclic order of incident edges
            at each vertex.
        
            If `None`, an empty `Graph` object is created.
        
        max_num_vert : int, optional
            Maximum number of vertices accommodated by the graph's
            internal storage, including subsequent Pachner moves.
        
            Defaults to Graph.DEFAULT_MAX_VERT (currently 50).
            The permitted range is 1 to 84.
        
        validate : bool, default=True
            If True, check that the edge count is compatible with
            a trivalent graph, that vertex labels are consecutive
            starting at zero, and that each vertex occurs exactly
            three times in the edge list.
        
            This does not test graph connectivity, planarity,
            or 3-connectivity.
        
        Raises
        ------
        ValueError
            If max_num_vert is invalid or the edge list fails
            the implemented validation checks.
        
        Notes
        -----
        The graph's vertex and edge counts are determined from
        the edge list. Face information is computed separately
        when first requested.
        """
        
        # Default placeholders for quantities to be computed when needed
        
        self._num_faces = None
        
        self._face_idx_list = None
        self._face_size_list = None
        self._face_bndy_list = None
        self._face_adj_matrix = None
        
        # Set hard limit for maximum number of vertices, edges here
        
        if max_num_vert is not None:
            
            if not isinstance(max_num_vert, (int, np.integer)):
                raise ValueError("max_num_vert must be an integer")
                
            if max_num_vert <= 0:
                raise ValueError("max_num_vert must be positive integer")
            
            self._max_num_vert = int(max_num_vert)
        else:
            self._max_num_vert = int(self.DEFAULT_MAX_VERT)
            
        # Check to make sure max number of vertices is valid, even if
        # DEFAULT_MAX_VERT is changed
    
        if self._max_num_vert > 84:
            raise ValueError("max_num_vert <= 84 for signed edge permutations of int8")
            
        self._max_num_edges = 3 * self._max_num_vert // 2
        
        # If an edge list is passed when Graph is created, assume it has an
        # implicit order based on vertex cyclic orders; otherwise, initialize
        # as empty object
        
        if ordered_edge_list is not None:
            
            edge_array = np.asarray(ordered_edge_list, dtype = np.int8)
        
            self._num_edges = edge_array.shape[0]
            self._num_vert = 2 * self._num_edges // 3
            
            # Pre-allocate arrays to have room for max number of vertices, edges
                
            self._vert_cyc_order = np.full((self._max_num_vert, 3), self._max_num_edges, \
                                          dtype = np.uint8)
            self._edge_list = np.full((self._max_num_edges, 2), self._max_num_vert, \
                                     dtype = np.int8)
            
            # Validate that edge list is sensible
            
            if validate:
                self._validate_edge_list(edge_array)
                
            self._edge_list[:self._num_edges] = edge_array
            
            # Use implicit edge order to determine vertex cyclic orders
            
            self._parse_implicit_order()
            
        else:
            
            # Allowance for class methods to use alternate creation methods
            # (not coded at this moment, but could include external specification
            # of the following variables)
            
            self._num_vert = 0
            self._num_edges = 0
            self._edge_list = None
            self._vert_cyc_order = None
            
    #-------------------------------------------------------------------------#
    
    @property
    def num_vert(self):
        r"""
            int: Number of vertices $|V|$ in the graph.
        """
        
        return self._num_vert
                
    #-------------------------------------------------------------------------#
    
    @property
    def num_edges(self):
        r"""
            int: Number of edges $|E|$ in the graph.
        """
        
        return self._num_edges
            
    #-------------------------------------------------------------------------#
    
    @property
    def num_faces(self):
        r"""
            int: Number of faces $|F|$ in the graph's combinatorial embedding.
        
            Face information is calculated on first access if it has
            not already been determined.
        """
    
        if self._num_faces is None:
            self.find_faces()
            
        return self._num_faces
            
    #-------------------------------------------------------------------------#
            
    def _validate_edge_list(self, edge_list):
        """
            Confirm that user-provided edge list is sensible edge list
        """
        
        # For a trivalent graph, 3V = 2E, so a multiple of three
        
        array_len = edge_list.shape[0]
        
        if array_len % 3 != 0:
            raise ValueError(f'Edge list has {array_len} edges; trivalent graphs edge counts must be a multiple of 3')
            
        # Vertex labels must be between 0..(V - 1)
        
        vertex_count = np.bincount(edge_list.ravel())
        
        if len(vertex_count) != (2 * array_len // 3):
            raise ValueError(f'Vertex label mismatch. Expected {2 * array_len // 3} labels, but found {len(vertex_count) - 1}')
            
        # Each vertex label must appear exactly three times
        
        if not np.all(vertex_count == 3):
            bad_vert = np.where(vertex_count != 3)[0]
            raise ValueError(f'Graph is not 3-regular. Vertices {bad_vert} have wrong degree.')
            
    #-------------------------------------------------------------------------#
            
    def _parse_implicit_order(self):
        """
            Each edge is placed into the vertex cyclic order in the order that
            it appears in the implicitly ordered edge list
        """
        
        # Track where to put the next appearance of an edge for each vertex
        
        vertex_places = np.zeros(self._num_vert, dtype = np.uint8)
        
        # Go through edge list, place edges in proper vertex cyclic order
        
        for edge_idx in range(self._num_edges):
            start, end = self._edge_list[edge_idx]
            
            start_place = vertex_places[start]
            self._vert_cyc_order[start, start_place] = edge_idx
            vertex_places[start] += 1
            
            end_place = vertex_places[end]
            self._vert_cyc_order[end, end_place] = edge_idx
            vertex_places[end] += 1
            
    #-------------------------------------------------------------------------#
    
    def _restore_implicit_order(self):
        """
            Given a correct vertex cyclic order, reorders the edge list to be
            consistent, relabels edges in vertex cyclic order to match this order
        """
        
        # Helper functions, for finding correct vertex label inside edge pair,
        # find next edge in CCW vertex cyclic order
        
        def get_other_vert(edge_idx, curr_vert):
            """
                Return vertex on edge 'edge_idx' that is *not* curr_vert
            """
            u, v = self._edge_list[edge_idx]
            return v if u == curr_vert else u
        
        def get_ccw_edge(edge_idx, curr_vert):
            """
                Returns edge label for edge CCW from edge 'edge_idx'
            """
            cycle = self._vert_cyc_order[curr_vert]
            if cycle[0] == edge_idx:
                return cycle[1]
            elif cycle[1] == edge_idx:
                return cycle[2]
            else:
                return cycle[0]
        
        # Initialization
        
        visited = [False] * self._num_edges
        active = [self._num_edges] * self._num_vert
        queue = deque()
        
        new_edge_list = np.empty((self._num_edges, 2), dtype = np.int8)
        edge_perm = np.full(self._num_edges, self._num_edges, dtype = np.int32)
        new_idx = 0
        
        curr_edge_idx = 0
        start_vert, curr_vert = self._edge_list[curr_edge_idx]
        
        # Main loop
        
        while new_idx < self._num_edges:
            
            # Current edge not already in edge list, active at both sides
            
            if not visited[curr_edge_idx] and active[curr_vert] in (curr_edge_idx, self._num_edges):
                visited[curr_edge_idx] = True
                
                # Queue edge CCW to current edge at starting vertex
                
                CCW_start_idx = get_ccw_edge(curr_edge_idx, start_vert)
                if not visited[CCW_start_idx]:
                    queue.append((CCW_start_idx, start_vert, \
                                  get_other_vert(CCW_start_idx, start_vert)))
                    active[start_vert] = CCW_start_idx
                else:
                    active[start_vert] = self._num_edges
                    
                # Add current edge to reordered edge list, and update edge perm
                # (NOTE: edge_perm takes the old index and gives the new, which
                # is needed to update vertex cyclic order; we will return the
                # inverse of this permutation at end)
                
                new_edge_list[new_idx] = self._edge_list[curr_edge_idx]
                edge_perm[curr_edge_idx] = new_idx
                new_idx += 1
                
                # Make CCW edge to current edge at current vertex next in line
                
                next_edge_idx = get_ccw_edge(curr_edge_idx, curr_vert)
                start_vert = curr_vert
                curr_vert = get_other_vert(next_edge_idx, curr_vert)
                curr_edge_idx = next_edge_idx
                
            # Current edge cannot be added to edge list, either because already
            # there, or else not active at current vertex
            
            else:
                if not visited[curr_edge_idx]:
                    queue.append((curr_edge_idx, start_vert, curr_vert))
                    active[curr_vert] = curr_edge_idx
                else:
                    active[curr_vert] = self._num_edges
                    
                curr_edge_idx, start_vert, curr_vert = queue.popleft()
                        
        # Ordering complete -- put new edge list into self._edge_list, and
        # use edge_perm to update new edge labels in vertex cyclic order
        
        self._edge_list[:self._num_edges] = new_edge_list
        self._vert_cyc_order[:self._num_vert] = edge_perm[self._vert_cyc_order[:self._num_vert]]
        
        # Return edge permutation giving old edge list idx at each new idx
        
        inverse_edge_perm = np.empty(self._num_edges, dtype=np.int8)
        inverse_edge_perm[edge_perm] = np.arange(self._num_edges, dtype=np.int8)
        
        return inverse_edge_perm
            
    #-------------------------------------------------------------------------#
    
    def is_isomorphic(self, other, return_perm = True):
        r"""
            Determine whether two embedded trivalent graphs are isomorphic.
            
            An isomorphism is required to preserve the counterclockwise cyclic
            ordering of incident edges at every vertex. Thus, the test is for
            isomorphism of combinatorially embedded graphs, rather than
            abstract graph isomorphism.
            
            Parameters
            ----------
            other : Graph
                Graph to compare with the current graph.
            
            return_perm : bool, default = True
                If `True`, return a signed edge permutation realizing an
                isomorphism, if one exists.
            
                If `False`, return only whether the graphs are isomorphic.
            
            Returns
            -------
            numpy.ndarray or bool or None
                If `return_perm = True` and the graphs are isomorphic, return
                a signed integer array of shape $(|E|,)$ describing the
                mapping of edges from `self` to `other`.
            
                Otherwise, return `None` if no isomorphism exists.
            
                If `return_perm = False`, return `True` or `False`.
            
            Notes
            -----
            The returned signed permutation uses one-based edge labels.
            For each zero-based source edge index $e$,
            
            $$
            p[e] = s_e (j+1),
            $$
            
            where $j$ is the corresponding zero-based edge index in
            `other`, and $s_e = \pm 1$.
            
            The sign records whether the isomorphism preserves ($+1$)
            or reverses ($-1$) the reference direction of that edge,
            as specified by its ordered endpoints.
            
            The sign does not indicate an independently assigned
            orientation decoration.
            
            Mirror embeddings are distinguished because the isomorphism
            must preserve the vertex cyclic orders.
            
            If multiple isomorphisms exist, only one is returned.
            
            Examples
            --------
            Test isomorphism without constructing an edge permutation:
            
            >>> G = Graph.create_prism(3)
            >>> G.is_isomorphic(G, return_perm = False)
            True
            
            Obtain a signed edge permutation:
            
            >>> perm = G.is_isomorphic(G)
            >>> len(perm) == G.num_edges
            True
        """

        # Verify whether number of vertices, edges are equal
        
        if self._num_vert != other._num_vert or self._num_edges != other._num_edges:
            return None if return_perm else False
        
        # Load in vertex face triples to (1) compute profile dicts for each
        # graph, with triples as keys, values as counts of these triples, and
        # compare for the two graphs, then if this passes, (2) send triples to
        # graph_isomorphism, along with edge lists and vertex cyclic orders
        
        # Note to self: I tested three ways of doing the quick isomorphism
        # check: (i) put the triples together into a sorted tuple, and then
        # use the built-in Python has function:
        #
        #   hash(tuple(sorted(map(tuple, mock_triples))))
        #
        # (ii) construct dictionaries using np.unique(triples, return_counts = True)
        # and zip(triples, counts), and (iii) using Counter, as shown below.
        # The timing ordering was (iii) < (i) < (ii). The supposition is that
        # numpy is too big of a tool for the relatively small arrays considered
        # here.
        
        self_triples = self._find_vert_triples()
        other_triples = other._find_vert_triples()
        
        self_profile = Counter(map(tuple, self_triples))
        other_profile = Counter(map(tuple, other_triples))
        
        if self_profile != other_profile:
            return None if return_perm else False
        
        # If all quick tests have been passes, try full-blown isomorphism test.
        # The idea is to find, for both graphs, the set of vertices which have
        # the rarest vertex face triple. This triple is chosen from the self
        # graph, and if there is an isomorphism, it must be in other graph as
        # well; this must be true, since comparing profiles checked this
        # condition. Then choose *one* vertex from self, and try to map this
        # vertex to all of the vertices in the set from other.
        
        match_buffer = np.zeros((1, self._num_edges), dtype = np.int8)
        
        min_triple = min(self_profile, key = lambda k : self_profile[k])
        
        self_mask = (self_triples == min_triple).all(axis = 1)
        other_mask = (other_triples == min_triple).all(axis = 1)
        
        self_vert_choices = np.sort(np.where(self_mask)[0])
        other_vert_choices = np.sort(np.where(other_mask)[0])
        
        result = _graph_isomorphism(
            self_vert_choices[0],
            other_vert_choices,
            self._edge_list[:self._num_edges].astype(np.int8),
            self._vert_cyc_order[:self._num_vert],
            other.edge_list[:other._num_edges].astype(np.int8),
            other.vert_cyc_order[:other._num_vert],
            find_all = False,
            symmetry_buffer = match_buffer
            )
        
        # Return either True/False or None/map between graphs, depending on
        # value of return_perm
        
        is_match = (result > 0)
        
        if not is_match:
            return None if return_perm else False
        
        return match_buffer[0] if return_perm else True
    
    #-------------------------------------------------------------------------#
    
    def __eq__(self, other):
        r"""
            Test equality of two embedded trivalent graphs.
            
            Two `Graph` objects are equal if there exists an isomorphism
            preserving the counterclockwise cyclic order of incident
            edges at every vertex.
            
            Parameters
            ----------
            other : Graph
                Graph to compare with the current graph.
            
            Returns
            -------
            bool
                `True` if the embedded graphs are isomorphic, and
                `False` otherwise.
            
            Notes
            -----
            Implements the `==` operator by calling
            `is_isomorphic(other, return_perm = False)`.
            
            For objects that are not instances of `Graph`, the method
            returns `NotImplemented`, allowing Python's normal
            comparison machinery to handle the result.
        """

        if not isinstance(other, Graph):
            return NotImplemented
        
        return self.is_isomorphic(other, return_perm = False)

    #-------------------------------------------------------------------------#
    
    def find_noniso_edges(self, mask_type = None, bitmask = None, sym_grp = None):
        r"""
            Find representatives of edge orbits under graph automorphisms.
            
            The edges are partitioned into orbits under the action of the
            positive automorphism group $\operatorname{Aut}^{+}(G)$, or the
            subgroup preserving a specified edge decoration.
            
            One representative is returned from each orbit.
            
            Parameters
            ----------
            mask_type : {None, 'orient', 'parity', 'combined'}, optional
                Type of edge decoration to preserve.
            
                - `None`: use all positive graph automorphisms.
                - `'orient'`: preserve the edge orientation bitmask.
                - `'parity'`: preserve the edge parity bitmask.
                - `'combined'`: preserve both orientations and parities.
            
            bitmask : int, optional
                Integer encoding the edge decorations specified by
                `mask_type`.
            
                Required when `mask_type` is not `None`. Ignored otherwise.
            
            sym_grp : numpy.ndarray, optional
                Array of shape $(N_{\mathrm{sym}}, |E|)$ containing signed
                edge permutations.
            
                If `None`, the positive automorphism group is calculated
                using `find_sym()`.
            
                Otherwise, the supplied group is used.
            
            Returns
            -------
            numpy.ndarray
                One-dimensional integer array containing one zero-based
                edge index from each orbit.
            
                The number of entries equals the number of edge orbits
                under the selected group action.
            
            Notes
            -----
            For an undecorated graph, the relevant group is
            $\operatorname{Aut}^{+}(G)$.
            
            For an oriented graph state $(G, d)$, the relevant group is the stabilizer
            
            $$
            \operatorname{Aut}^{+}(G, d)
            =
            \{g \in \operatorname{Aut}^{+}(G) : g \cdot d = d\},
            $$
            
            with the analogous definitions for parity and combined
            decorations.
            
            The orbit of an edge $e$ under a group $H$ is
            
            $$
            H \cdot e = \{g(e) : g \in H\}.
            $$
            
            Two edges are equivalent if they belong to the same orbit.
            
            The returned indices are representatives of these orbits;
            they are not necessarily the smallest indices in their
            respective orbits.
            
            Only the absolute values of the signed edge permutations
            are used when determining edge equivalence.
            
            The supplied `sym_grp` is assumed to contain valid graph
            automorphisms. Its group properties are not verified.
            
            Examples
            --------
            Find representatives of the edge orbits of a triangular prism:
            
            >>> G = Graph.create_prism(3)
            >>> edges = G.find_noniso_edges()
            >>> len(edges)
            2
        """
        
        # Calculate full graph symmetry group if none is given
        
        if sym_grp is None:
            sym_grp = self.find_sym()
    
        else:
            # Validate externally supplied symmetry group
            sym_grp = np.asarray(sym_grp)
        
            if sym_grp.ndim != 2:
                raise ValueError("sym_grp must be a 2D array")
        
            if sym_grp.shape[0] == 0 or sym_grp.shape[1] != self._num_edges:
                raise ValueError(
                    f"sym_grp must have shape (N, {self._num_edges}), N > 0"
                )
        
            if sym_grp.dtype.kind not in "iu":
                raise TypeError("Symmetry labels must be integers")
        
            if (np.any(sym_grp > self._num_edges) or
                (sym_grp.dtype.kind == "i" and
                 np.any(sym_grp < -self._num_edges))):
                raise ValueError("Symmetry labels are out of range")
        
            sym_grp = np.ascontiguousarray(sym_grp, dtype=np.int8)
        
        if sym_grp.ndim !=2:
            raise ValueError("sym_grp must be 2d array")
            
        if sym_grp.shape[1] != self._num_edges:
            raise ValueError(f"group elements must be {self._num_edges} in size")
            
        # If no edge property type is given, return non-isomorphic graph edges
        # under all graph symmetries
          
        if mask_type is None:
            return _get_non_iso_edges(self._num_edges, sym_grp)
        
        # If graph property is given, return non-isomorphic edges that preserve
        # this property, given as edges bitmask. To do this, we find the subgroup
        # of symmetries preserving the given structure.
        
        if not isinstance(mask_type, str) or mask_type not in ['orient', 'parity', 'combined']:
            raise ValueError("mask_type must bestring from ['orient', 'parity', 'combined']")
            
        if not isinstance(bitmask, (int, np.integer)):
            raise ValueError("bitmask must be integer")
            
        sym_subgrp = np.empty((sym_grp.shape[0], self._num_edges), dtype = np.int8)
        row = 0
            
        if mask_type == 'parity':
            for sym in sym_grp:
                if permute_parity_bits(bitmask, sym, num_edges = self._num_edges) == bitmask:
                    sym_subgrp[row] = sym
                    row += 1
        
        if mask_type == 'orient':
            for sym in sym_grp:
                if permute_orient_bits(bitmask, sym, num_edges = self._num_edges) == bitmask:
                    sym_subgrp[row] = sym
                    row += 1
                    
        if mask_type == 'combined':
            for sym in sym_grp:
                if permute_combined_bits(bitmask, sym, num_edges = self._num_edges) == bitmask:
                    sym_subgrp[row] = sym
                    row += 1
                    
        return _get_non_iso_edges(self._num_edges, sym_subgrp[:row])
    
    #-------------------------------------------------------------------------#

    def find_sym(self):
        r"""
            Find the positive automorphism group $\operatorname{Aut}^{+}(G)$
            of the embedded graph $G$.

            A positive automorphism is an isomorphism from the graph to itself
            that preserves the counterclockwise cyclic ordering of
            incident edges at every vertex.
            
            These automorphisms are represented by signed permutations
            of the graph edges.
            
            Returns
            -------
            numpy.ndarray
                Integer array of shape $(N_{\mathrm{sym}}, |E|)$, where
                $N_{\mathrm{sym}}$ is the number of automorphisms found.
            
                Each row represents one automorphism as a signed
                permutation of the edges.
            
            Notes
            -----
            The returned permutations use one-based signed edge labels.
            
            For a permutation `p`, the entry `p[e]` specifies the
            image of the zero-based edge index $e$:
            
            - `abs(p[e]) - 1` is the corresponding edge index.
            - A positive sign means the reference direction is preserved.
            - A negative sign means the reference direction is reversed.
            
            The signs refer to the ordering of edge endpoints, not
            to independently assigned orientation decorations.
            
            Only automorphisms preserving the vertex cyclic orders
            are included. Automorphisms reversing the embedding
            orientation are excluded.
            
            The identity automorphism is included. The ordering of
            group elements is not guaranteed.
            
            The implementation currently limits the number of
            returned automorphisms to `DEFAULT_MAX_SYM` (60).
            Graphs with larger automorphism groups may therefore
            have incomplete results.
            
            Examples
            --------
            Find the automorphism group of a triangular prism:
            
            >>> G = Graph.create_prism(3)
            >>> sym_grp = G.find_sym()
            >>> sym_grp.shape[1] == G.num_edges
            True
            
            The order of the returned group is:
            
            >>> len(sym_grp)
            6
        """
    
        self_triples = self._find_vert_triples()
        
        # Partition vertices by triple signature to find the minimum size tier
        
        triple_tuples = [tuple(row) for row in self_triples]
        counts = Counter(triple_tuples)
        min_triple = min(counts, key = lambda triple_idx: counts[triple_idx])
        
        # 2. Extract vertex indices belonging to this partition
        
        vert_choices = np.sort(np.array([vert for vert, triple in enumerate(triple_tuples) \
                                         if triple == min_triple], dtype=np.uint8))
        
        # 3. Calculate bounded buffer allocation size: 3 * K
        
        max_bound = min(3 * len(vert_choices), self.DEFAULT_MAX_SYM)
        sym_list = np.zeros((max_bound, self._num_edges), dtype=np.int8)
        
        # Slice raw properties down to active graph limits
        
        el = self._edge_list[:self._num_edges].astype(np.int8)
        vco = self._vert_cyc_order[:self._num_vert].astype(np.int8)
        
        sym_count = _graph_isomorphism(
            vert_choices[0], vert_choices,
            el, vco,
            el, vco,
            find_all = True,
            symmetry_buffer = sym_list
        )
        
        return sym_list[:sym_count]
    
    #-------------------------------------------------------------------------#
    
    def find_min_parity_state(self, parity_state, sym_grp):
        r"""
            Find the canonical representative of a parity state.
            
            The state is minimized over its orbit under the supplied
            group of signed edge permutations.
            
            Parameters
            ----------
            parity_state : int
                Encoded parity state containing a graph ID and
                $|E|$ parity bits.
            
            sym_grp : numpy.ndarray
                Array of shape $(N_{\mathrm{sym}}, |E|)$ containing
                signed edge permutations, usually obtained from
                `find_sym()`.
            
            Returns
            -------
            int
                Numerically smallest encoded parity state under
                the supplied group action.
            
            Notes
            -----
            The graph ID is preserved. Only the parity bitmask
            is transformed.
            
            If `sym_grp` is the full positive automorphism group,
            the result is a canonical representative of the
            parity state under $\operatorname{Aut}^{+}(G)$.
        """

        return find_min_parity_state(parity_state, sym_grp, num_edges = self._num_edges)
    
    #-------------------------------------------------------------------------#
    
    def find_min_orient_state(self, orient_state, sym_grp):
        r"""
            Find the canonical representative of an orientation state.
            
            The state is minimized over its orbit under the supplied
            group of signed edge permutations.
            
            Parameters
            ----------
            orient_state : int
                Encoded orientation state containing a graph ID and
                $|E|$ orientation bits.
            
            sym_grp : numpy.ndarray
                Array of shape $(N_{\mathrm{sym}}, |E|)$ containing
                signed edge permutations, usually obtained from
                `find_sym()`.
            
            Returns
            -------
            int
                Numerically smallest encoded orientation state
                under the supplied group action.
            
            Notes
            -----
            The graph ID is preserved. Only the orientation
            bitmask is transformed.
            
            A negative entry in a signed permutation accounts
            for reversal of the reference direction of an edge
            when transforming its orientation bit.
            
            Global reversal of all edge orientations is not
            included as an additional equivalence operation.
        """

        return find_min_orient_state(orient_state, sym_grp, num_edges = self._num_edges)
    
    #-------------------------------------------------------------------------#
    
    def find_min_combined_state(self, combined_state, sym_grp):
        r"""
            Find the canonical representative of a combined state.
            
            The state contains both orientation and parity decorations.
            Both bitmasks are transformed simultaneously under the
            same signed edge permutation.
            
            Parameters
            ----------
            combined_state : int
                Encoded combined state containing a graph ID,
                $|E|$ orientation bits, and $|E|$ parity bits.
            
            sym_grp : numpy.ndarray
                Array of shape $(N_{\mathrm{sym}}, |E|)$ containing
                signed edge permutations, usually obtained from
                `find_sym()`.
            
            Returns
            -------
            int
                Numerically smallest encoded combined state
                under the supplied group action.
            
            Notes
            -----x`
            The graph ID is preserved.
            
            The encoding places the orientation bits above the
            parity bits. Consequently, numerical minimization
            first minimizes the orientation bitmask and then,
            among equivalent minimum orientations, minimizes
            the parity bitmask.
            
            The same permutation must act on both decorations;
            they are not minimized independently.
        """

        return find_min_combined_state(combined_state, sym_grp, num_edges = self._num_edges)
        
    #-------------------------------------------------------------------------#
    
    def map_parity_state(self, parity_state, other):
        r"""
            Map a parity state to a positively isomorphic graph.
            
            Find a positive isomorphism from the current graph `self`
            to `other`, then transform the parity state to match the
            edge ordering of `other`.
            
            Parameters
            ----------
            parity_state : int
                Encoded parity state whose edge indices correspond
                to the current graph.
            
            other : Graph
                Target graph, positively isomorphic to the current graph.
            
            Returns
            -------
            numpy.uint64
                Encoded parity state with edge decorations expressed
                in the edge ordering of `other`.
            
            Raises
            ------
            ValueError
                If the two embedded graphs are not positively isomorphic.
            
            Notes
            -----
            The signed edge permutation is obtained automatically
            using `is_isomorphic()`.
            
            Only edge positions are permuted; signs in the permutation
            do not change parity values.
            
            The encoded graph ID is preserved. Neither graph is modified.
        """
        
        signed_edge_perm = self.is_isomorphic(other)
        
        if signed_edge_perm is None:
            raise ValueError("Graphs are not isomorphic")
            
        signed_edge_perm = signed_edge_perm.astype(dtype = np.int64)
            
        return permute_parity_state(parity_state, signed_edge_perm, num_edges = self._num_edges)
    
    #-------------------------------------------------------------------------#
    
    def map_orient_state(self, orient_state, other):
        r"""
            Map an orientation state to a positively isomorphic graph.
            
            Find a positive isomorphism from the current graph `self`
            to `other`, then transform the orientation state to match
            the edge ordering and reference directions of `other`.
            
            Parameters
            ----------
            orient_state : int
                Encoded orientation state whose edge indices correspond
                to the current graph.
            
            other : Graph
                Target graph, positively isomorphic to the current graph.
            
            Returns
            -------
            numpy.uint64
                Encoded orientation state with edge decorations
                expressed in the edge ordering of `other`.
            
            Raises
            ------
            ValueError
                If the two embedded graphs are not positively isomorphic.
            
            Notes
            -----
            The signed edge permutation is obtained automatically
            using `is_isomorphic()`.
            
            A negative permutation entry reverses the corresponding
            orientation bit to account for the change in edge
            reference direction.
            
            The encoded graph ID is preserved. Neither graph is modified.
        """
        
        signed_edge_perm = self.is_isomorphic(other)
        
        if signed_edge_perm is None:
            raise ValueError("Graphs are not isomorphic")
            
        signed_edge_perm = signed_edge_perm.astype(dtype = np.int64)
            
        return permute_orient_state(orient_state, signed_edge_perm, num_edges = self._num_edges)
    
    #-------------------------------------------------------------------------#
    
    def map_combined_state(self, combined_state, other):
        r"""
            Map a combined state to a positively isomorphic graph.
            
            Find a positive isomorphism from the current graph `self`
            to `other`, then transform both orientation and parity
            decorations to match the edge ordering of `other`.
            
            Parameters
            ----------
            combined_state : int
                Encoded combined state whose edge indices correspond
                to the current graph.
            
            other : Graph
                Target graph, positively isomorphic to the current graph.
            
            Returns
            -------
            numpy.uint64
                Encoded combined state with both decorations expressed
                in the edge ordering of `other`.
            
            Raises
            ------
            ValueError
                If the two embedded graphs are not positively isomorphic.
            
            Notes
            -----
            Both decorations are transformed using the same signed
            edge permutation obtained from `is_isomorphic()`.
            
            Negative permutation entries reverse orientation bits,
            but do not affect parity values.
            
            The encoded graph ID is preserved. Neither graph is modified.
        """
        
        signed_edge_perm = self.is_isomorphic(other)
        
        if signed_edge_perm is None:
            raise ValueError("Graphs are not isomorphic")
            
        signed_edge_perm = signed_edge_perm.astype(dtype = np.int64)
            
        return permute_combined_state(combined_state, signed_edge_perm, num_edges = self._num_edges)
    
    #-------------------------------------------------------------------------#
    
    def permute_parity_state(self, parity_state, signed_edge_perm):
        r"""
            Apply a signed edge permutation to an encoded parity state.
            
            Parameters
            ----------
            parity_state : int
                Encoded parity state to transform.
            
            signed_edge_perm : array_like of int
                Signed permutation of shape $(|E|,)$ using one-based
                edge labels.
            
            Returns
            -------
            numpy.uint64
                Encoded state with permuted parity bits.
            
            Notes
            -----
            Only the absolute values of the permutation entries
            affect the parity bitmask.
            
            The graph ID is preserved.
            
            The permutation is checked for validity as a signed
            permutation, but is not verified to be a positive
            automorphism of the graph.
        """

        return permute_parity_state(parity_state, signed_edge_perm, num_edges = self._num_edges)
    
    #-------------------------------------------------------------------------#
    
    def permute_orient_state(self, orient_state, signed_edge_perm):
        r"""
            Apply a signed edge permutation to an encoded orientation state.
            
            Parameters
            ----------
            orient_state : int
                Encoded orientation state to transform.
            
            signed_edge_perm : array_like of int
                Signed permutation of shape $(|E|,)$ using one-based
                edge labels.
            
            Returns
            -------
            numpy.uint64
                Encoded state with permuted orientation bits.
            
            Notes
            -----
            A negative permutation entry reverses the corresponding
            orientation bit when assigning it to its target edge.
            
            The graph ID is preserved.
            
            The permutation is checked for validity as a signed
            permutation, but is not verified to be a positive
            automorphism of the graph.
        """

        return permute_orient_state(orient_state, signed_edge_perm, num_edges = self._num_edges)
        
    #-------------------------------------------------------------------------#
    
    def permute_combined_state(self, combo_state, signed_edge_perm):
        r"""
            Apply a signed edge permutation to an encoded combined state.
            
            Parameters
            ----------
            combo_state : int
                Encoded state containing orientation and parity bits.
            
            signed_edge_perm : array_like of int
                Signed permutation of shape $(|E|,)$ using one-based
                edge labels.
            
            Returns
            -------
            numpy.uint64
                Encoded state with both decorations permuted.
            
            Notes
            -----
            The same signed permutation acts on both decorations.
            
            Negative entries reverse the corresponding orientation
            bits but do not change parity values.
            
            The graph ID is preserved.
            
            The permutation is checked for validity as a signed
            permutation, but is not verified to be a positive
            automorphism of the graph.
        """

        return permute_combined_state(combo_state, signed_edge_perm, num_edges = self._num_edges)
    
    #-------------------------------------------------------------------------#
    
    @property
    def edge_list(self):
        r"""
            ndarray of shape $(|E|, 2)$: Edges $(u, v)$ of the graph.
            
            Each row contains the two zero-based vertex indices of an edge.
            The order of the endpoints defines the reference direction of
            the edge, used when constructing signed edge permutations.
            
            Edges are indexed by their row positions, starting at zero.
            
            A copy of the active edge list is returned, so modifying the
            returned array does not affect the Graph object.
        """
        
        return self._edge_list[:self._num_edges].copy()
        
    #-------------------------------------------------------------------------#
    
    @property
    def vert_cyc_order(self):
        r"""
            ndarray of shape $(|V|, 3)$: Cyclic ordering of incident edges.
            
            Each row contains the zero-based indices of the three edges
            incident to a vertex, in counterclockwise cyclic order.
            
            Row i corresponds to vertex i.
            
            A copy of the active vertex cyclic orders is returned, so
            modifying the returned array does not affect the Graph object.
        """
        
        return self._vert_cyc_order[:self._num_vert].copy()
        
    #-------------------------------------------------------------------------#
    
    @property
    def face_bndy_list(self):
        r"""
            Return the ordered boundary edges of each face.
            
            Returns
            -------
            list of list of int
                Entry i contains the signed edge labels forming the
                boundary of face i, traversed counterclockwise.
            
            Notes
            -----
            The signed label of edge $e$ is $+(e+1)$ when
            traversed in its reference direction, and
                $-(e+1)$ when traversed in the opposite direction.
            
            The reference direction is determined by the order of the
            edge endpoints in edge_list, not by any orientation decoration.
            
            Face information is calculated on first access if necessary.
            A copy of each face boundary list is returned.
        """
        
        if self._face_bndy_list is None:
            self.find_faces()
            
        return [boundary.copy() for boundary in self._face_bndy_list[:self._num_faces]]
    
    #-------------------------------------------------------------------------#
    
    @property
    def face_idx_list(self):
        r"""
            Return the face indices on either side of each edge.
            
            Returns
            -------
            ndarray of shape (E, 2)
                For each edge e:
            
                - [e, 0] gives the index of the face to its left.
                - [e, 1] gives the index of the face to its right.
            
            Notes
            -----
            Left and right are defined relative to the reference direction
            of each edge, as specified by its ordered endpoints.
            
            Face indices are zero-based and determined by find_faces().
            They are not intrinsic labels preserved under graph isomorphisms
            or Pachner moves.
            
            Face information is calculated on first access if necessary.
            A copy of the array is returned.
        """
        
        if self._face_idx_list is None:
            self.find_faces()
        
        return self._face_idx_list[:self._num_edges].copy()
        
    #-------------------------------------------------------------------------#
    
    @property
    def face_adj_matrix(self):
        r"""
            Return the symmetric adjacency matrix of the graph's faces.
            
            Returns
            -------
            numpy.ndarray
                Symmetric array of shape $(|F|, |F|)$.
            
                The entry $A_{ij}$ counts the number of graph edges
                shared by faces $i$ and $j$.
            
            Notes
            -----
            The matrix is the adjacency matrix of the dual graph,
            with edge multiplicities retained.
            
            If an edge has the same face on both sides, it
            contributes two to the diagonal entry $A_{ii}$.
            
            Face information is calculated on first access if
            necessary. A copy of the matrix is returned.
        """
        
        if self._face_adj_matrix is None:
            A = np.zeros((self.num_faces, self.num_faces), dtype = np.int32)
            
            if self._face_idx_list is None:
                self.find_faces()
            
            left = self._face_idx_list[:self._num_edges, 0]
            right = self._face_idx_list[:self._num_edges, 1]
            
            np.add.at(A, (left, right), 1)
            np.add.at(A, (right, left), 1)
            
            self._face_adj_matrix = A
            
        return self._face_adj_matrix.copy()
    
    #-------------------------------------------------------------------------#
    
    @property
    def face_size_list(self):
        r"""
            Return the number of boundary edges of each face.
            
            Returns
            -------
            numpy.ndarray
                One-dimensional array of shape $(|F|,)$, where
                $|F|$ is the number of faces.
            
                Entry $i$ gives the length of the boundary walk
                of face $i$, counting edge traversals.
            
            Notes
            -----
            Face information is calculated on first access if
            necessary. A copy of the array is returned.
        """
        
        if self._face_size_list is None:
            self.find_faces()
        
        return self._face_size_list[:self._num_faces].copy()
    
    #-------------------------------------------------------------------------#
    
    def _find_vert_triples(self):
        """
            For each vertex, find triple of sizes for its incident faces, order
            each triple cyclically so minimum size is first in list.
        """
        
        # See if face indices, signatures have been calculated; if not, do so
        
        if self._face_idx_list is None:
            self.find_faces()
            
        # Go through every vertex, find sizes of incident faces and sort so it
        # is smallest version; note we preserve the cyclic order of the faces
        # around each vertex to weed out mirror images
        
        vert_triples = np.zeros((self._num_vert, 3), dtype = np.uint8)
        
        for vert in range(self._num_vert):
            
            face_sizes = []
            
            for edge in self._vert_cyc_order[vert]:
                
                # Determine if vert is the start or end of the edge in order
                # to grab the correct face to maintain correct CCW direction
                
                if self._edge_list[edge, 0] == vert:
                    face = self._face_idx_list[edge, 1]  # Start -> right face
                else:
                    face = self._face_idx_list[edge, 0]  # End -> left face
                
                face_sizes.append(self._face_size_list[face])
                
            # Find the index of the minimum face size, and cyclically shift to
            # put that value in front of the triple; ensure that the minimum
            # order triple is put in by checking all three possibilities
            
            canonical_triple = min(face_sizes, \
                                   face_sizes[1:] + face_sizes[:1], \
                                   face_sizes[2:] + face_sizes[:2])
                
            vert_triples[vert] = canonical_triple
            
        return vert_triples
    
    #-------------------------------------------------------------------------#
    
    def find_faces(self):
        r"""
            Find the faces of the combinatorially embedded graph.
            
            The faces are determined by the counterclockwise cyclic
            ordering of incident edges at each vertex. They are
            enumerated by tracing the directed boundary walks.
            
            The resulting face information is stored in the graph.
            
            Returns
            -------
            None
                The method updates the internally stored face data.
            
            Notes
            -----
            The following face information is calculated:
            
            - `num_faces`: number of faces $|F|$.
            - `face_bndy_list`: signed edge labels for each face boundary.
            - `face_idx_list`: left- and right-hand face indices for each edge.
            - `face_size_list`: number of edge traversals around each face.
            
            Face indices are zero-based and assigned in traversal order.
            
            The boundary of each face is traversed counterclockwise,
            with the face on the left. Boundary edges use one-based
            signed labels, with positive sign when traversed in the
            edge's reference direction.
            
            For a zero-based edge index $e$, `face_idx_list[e, 0]`
            and `face_idx_list[e, 1]` give the faces on the left
            and right of its reference direction, respectively.
            
            Face labels depend on the edge ordering and are not
            invariant under graph isomorphisms.
            
            For a non-planar combinatorial embedding, the faces
            correspond to the boundary walks of the associated
            orientable surface embedding, not necessarily the sphere.
            
            Face information is also calculated automatically when
            first requested through the corresponding properties.
        """
        
        # The face indices are recorded in a master list for each edge, with
        # face_idx_list[edge_idx][0] the face index for the left-hand face,
        # and face_idx_list[edge_idx][1] the right-hand face
        
        self._face_idx_list = np.full((self._max_num_edges, 2), self._max_num_vert, \
                                     dtype = np.uint8)
            
        face_bndy_list = []
        
        # Track whether the edge has been traversed, in both directions, using
        # shape (numEdges, 2) array; visited[edge][0] is forward direction, i.e.
        # start -> end, while visited[edge_idx][1] is backwards direction
        
        visited = np.zeros((self._num_edges, 2), dtype = np.bool_)
        
        face_signature_list = []
        
        # Start at each edge, and travel CCW around each face; this means we
        # move in CW direction at each vertex to reach next edge; record face
        # index for each edge reached, which is why left-hand is 0 in
        # face_idx_list -- traveling in canonical direction along edge while
        # moving CCW around face means we are on the left-hand side of edge
        
        face_idx = -1
        
        for start_edge_idx in range(self._num_edges):
            for start_dir in [0, 1]:
                if visited[start_edge_idx][start_dir]:
                    continue  # Already visited edge in this direction
                    
                curr_edge_idx, curr_dir = start_edge_idx, start_dir
                face_size = 0
                face_idx += 1
                curr_face_list = []
                
                # Find starting vertex based on direction traveling along edge,
                # with curr_dir = 0 giving start -> end, and curr_dir = 1 end
                # -> start; thus, 'left' and 'right' will be consistent based
                # on the canonical direction start -> end
                
                curr_vert = self._edge_list[curr_edge_idx][1 if curr_dir == 0 else 0]
                
                while True:
                    
                    # Mark edge as visited in current direction, increment
                    # current face size by 1, and put current face index on
                    # appropriate side of current edge
                    
                    visited[curr_edge_idx, curr_dir] = True
                    face_size += 1
                    self._face_idx_list[curr_edge_idx, curr_dir] = face_idx
                    
                    # Record edges using 1-based signed index, so that direction
                    # of canonical edge orientation relative to CCW direction
                    # around face is given
                        
                    if curr_dir == 0:
                        curr_face_list.append(int(curr_edge_idx) + 1)
                    else:
                        curr_face_list.append(-(int(curr_edge_idx) + 1))
                    
                    # At current vertex, travel around in CW direction to next
                    # vertex in path CCW around the face
                    
                    next_idx = 0
                    if self._vert_cyc_order[curr_vert, 1] == curr_edge_idx:
                        next_idx = 1
                    elif self._vert_cyc_order[curr_vert, 2] == curr_edge_idx:
                        next_idx = 2
                        
                    next_edge_idx = self._vert_cyc_order[curr_vert, next_idx - 1]
                    
                    # Determine which direction relative to canonical edge
                    # orientation we travel as we move CCW around face
                    
                    if self._edge_list[next_edge_idx, 0] == curr_vert:
                        next_dir = 0
                        next_vert = self._edge_list[next_edge_idx, 1]
                    else:
                        next_dir = 1
                        next_vert = self._edge_list[next_edge_idx, 0]
                        
                    # Move on to next vertex, edge
                    
                    curr_edge_idx = next_edge_idx
                    curr_dir = next_dir
                    curr_vert = next_vert
                    
                    # Have we returned to starting point?
                    
                    if (curr_edge_idx == start_edge_idx) and (curr_dir == start_dir):
                        break
                    
                face_signature_list.append(face_size)
                face_bndy_list.append(curr_face_list)
                
        # Update number of faces, list of face sizes indices by face_idx
        
        self._num_faces = (face_idx + 1)
        self._face_size_list = np.array(face_signature_list, dtype = np.uint8)
        self._face_bndy_list = face_bndy_list
        
    #-------------------------------------------------------------------------#
    
    def _clear_face_data(self):
        """
            Clears all face information; used when graph face structure changes
        """
        
        self._face_bndy_list = None
        self._face_idx_list = None
        self._face_size_list = None
        self._face_adj_matrix = None
        
    #-------------------------------------------------------------------------#
    # Pachner graph moves
    #-------------------------------------------------------------------------#
    
    def pachner13(self, vert_idx):
        r"""
            Perform a Pachner 1-3 move at a specified vertex.
            
            Replace the vertex by a triangular face, reconnecting
            its three incident edges to the vertices of the triangle.
            
            The graph is modified in place.
            
            Parameters
            ----------
            vert_idx : int
                Zero-based index of the vertex to expand.
            
            Returns
            -------
            tuple of int
                Zero-based indices of the three newly created edges,
                in their order in the updated edge list.
            
            Raises
            ------
            ValueError
                If the move would exceed the graph's `max_num_vert`
                storage limit.
            
            Notes
            -----
            The move changes the graph sizes according to
            
            $$
            (|V|, |E|, |F|)
            \longmapsto
            (|V|+2, |E|+3, |F|+1).
            $$
            
            The original vertex is retained, and two new vertices
            are appended to the vertex list.
            
            The indices of all existing edges are preserved.
            The three new edges are appended at indices
            $|E|$, $|E|+1$, and $|E|+2$, where $|E|$
            is the original edge count.
            
            Face information is invalidated and recalculated when
            subsequently requested.
            
            The method does not assign orientation or parity
            decorations to the new edges.
            
            Consequently, existing edge decorations retain
            their edge indices, while the new edges require
            additional decoration values.
            
            Existing edge indices are preserved, but changes
            in reference direction may require updating their
            orientation bits.
            
            Examples
            --------
            Expand vertex 0 of a tetrahedral graph:
            
            >>> G = Graph.create_prism(3)
            >>> new_edges = G.pachner13(0)
            >>> new_edges
            (9, 10, 11)
            >>> G.num_vert
            8
        """
        
        if (self._num_vert + 2) > self._max_num_vert:
            raise ValueError('Pachner 1-3 move exceeds max_num_vert value')
        
        # Get incident edges to vertex, write new edge lists by including new
        # vertices, maintaining start < end format for each edge
        
        edge_aaa, edge_bbb, edge_ccc = self._vert_cyc_order[vert_idx]
        
        if self._edge_list[edge_bbb, 0] == vert_idx:
            self._edge_list[edge_bbb] = [self._edge_list[edge_bbb, 1], self._num_vert]
        else:
            self._edge_list[edge_bbb, 1] = self._num_vert
            
        if self._edge_list[edge_ccc, 0] == vert_idx:
            self._edge_list[edge_ccc] = [self._edge_list[edge_ccc, 1], self._num_vert + 1]
        else:
            self._edge_list[edge_ccc, 1] = self._num_vert + 1
        
        # Create two new vertices, add them to end of edge list and vertex
        # cyclic order list. Note this means the remaining vertex, edge labels
        # remain in place in the edge list and vertex cyclic orders.
        
        self._edge_list[self._num_edges] = [vert_idx, self._num_vert]
        self._edge_list[self._num_edges + 1] = [vert_idx, self._num_vert + 1]
        self._edge_list[self._num_edges + 2] = [self._num_vert, self._num_vert + 1]
        
        self._vert_cyc_order[vert_idx] = [edge_aaa, self._num_edges, self._num_edges + 1]
        self._vert_cyc_order[self._num_vert] = [self._num_edges, edge_bbb, self._num_edges + 2]
        self._vert_cyc_order[self._num_vert + 1] = [self._num_edges + 1, self._num_edges + 2, edge_ccc]
        
        # Change vertex, edge numbers
        
        self._num_vert += 2
        self._num_edges += 3
    
        # Face information is no longer valid

        self._clear_face_data()
        
        if self._num_faces is not None:
            self._num_faces += 1
            
        # This move always succeeds, and the original edges remain in place in
        # the edge list. The new edges are added to the end as the last three,
        # and the last two vertices in the vertex cyclic order are the new
        # vertices.
        
        return self._num_edges - 3, self._num_edges - 2, self._num_edges - 1
    
    #-------------------------------------------------------------------------#
    
    def pachner22(self, edge_idx):
        r"""
            Perform a Pachner 2-2 move on a specified edge.
            
            Replace the local connection between the two vertices
            incident to the edge, preserving the trivalent structure
            and cyclic ordering of incident edges.
            
            The graph is modified in place if the move succeeds.
            
            Parameters
            ----------
            edge_idx : int
                Zero-based index of the edge on which to perform
                the move.
            
            Returns
            -------
            tuple of (int, numpy.ndarray)
                A pair `(new_edge_idx, move_perm)`.
            
                `new_edge_idx` is the zero-based index of the
                transformed edge in the updated graph.
            
                `move_perm` is a signed integer array of shape
                $(|E|,)$ mapping old edge indices to new indices.
            
                If the move is prohibited, return `-1` and an
                empty integer array. The graph is not modified.
            
            Raises
            ------
            ValueError
                If `edge_idx` is outside the range of valid
                edge indices.
            
            Notes
            -----
            The move preserves the numbers of vertices, edges,
            and faces.
            
            The implementation rejects a move when the two
            opposite faces involved already share an edge,
            as this would violate the intended 3-connectivity
            restriction.
            
            The signed edge mapping uses one-based labels.
            For an original edge index $e$,
            
            $$
            p[e] = s_e(j+1),
            $$
            
            where $j$ is its index after the move and
            $s_e = \pm 1$ records any reversal of its
            reference direction.
            
            The edge list may be reordered to preserve the
            implicit cyclic-order convention.
            
            Face information is invalidated after a successful
            move and recalculated when subsequently requested.
            
            The method does not assign new edge decorations
            or determine admissible decorated Pachner moves.
        """
        
        if (edge_idx < 0) or (edge_idx >= self._num_edges):
            raise ValueError(f'Edge index needs to be between 0 and {self._num_edges - 1}')
        
        start_vert, end_vert = self._edge_list[edge_idx]
        
        # .item() extracts the desired index out of np.array([idx])
        
        start_row = self._vert_cyc_order[start_vert]
        start_idx = np.where(start_row == edge_idx)[0].item()
        
        end_row = self._vert_cyc_order[end_vert]
        end_idx = np.where(end_row == edge_idx)[0].item()
        
        # Find edge indices of neighbors for chosen edge vertices, with specific
        # labels given by vertex cyclic ordering; use fact that -1 == +2 mod 3,
        # and Python always can process negative indices in slice as "before"
        
        aaa, bbb = start_row[start_idx - 2], start_row[start_idx - 1]
        ccc, ddd = end_row[end_idx - 2], end_row[end_idx - 1]
        
        # To preserve 3-connectedness, faces *not* incident to chosen edge
        # cannot already share an edge; first, find faces if not already done
        
        if self._face_idx_list is None:
            self.find_faces()
            
        # Find the faces at each vertex that are not incident to chosen edge,
        # and see if they already have an edge in common; first, need to
        # determine which side of aaa, ccc to look for this face, dependent
        # on the canonical direction of the edges, either towards (1) or away
        # (0) from the start, end vertices
        
        aaa_side = np.where(self._edge_list[aaa] == start_vert)[0].item()
        ccc_side = np.where(self._edge_list[ccc] == end_vert)[0].item()
            
        start_face_idx = self._face_idx_list[aaa, aaa_side]
        end_face_idx = self._face_idx_list[ccc, ccc_side]
        
        appear_same = (self._face_idx_list[:, 0] == start_face_idx) & \
                      (self._face_idx_list[:, 1] == end_face_idx)
        appear_diff = (self._face_idx_list[:, 1] == start_face_idx) & \
                      (self._face_idx_list[:, 0] == end_face_idx)
        
        if np.any(appear_same | appear_diff):
            
            # Two faces already share an edge, so cannot use 2-2 move here;
            # return 'fail' along with 
            
            return -1, np.empty(0, dtype = np.int8)
        
        # Change vertex cyclic orders for start, end vertices of chosen edge
        
        self._vert_cyc_order[start_vert] = np.array([bbb, ccc, end_vert])
        self._vert_cyc_order[end_vert] = np.array([ddd, aaa, start_vert])
        
        # Ensure that edge list still has implicit ordering matching the new
        # vertex cyclic order; shift the 2-2 move edge to its new slot in place;
        # At same time, create edge permutation of the move, including any
        # orientation flips of the affected edges, and confirm that edges
        # maintain start < end for vertex labels; these are ax -> ay and
        # cy -> cx.
        
        move_perm = np.arange(1, self._num_edges + 1, dtype = np.int8)
        
        if start_vert == self._edge_list[aaa, 0]:
            aaa_vert = self._edge_list[aaa, 1]
            aaa_loc = 1
        else:
            aaa_vert = self._edge_list[aaa, 0]
            aaa_loc = 0
            
        if end_vert == self._edge_list[ccc, 0]:
            ccc_vert = self._edge_list[ccc, 1]
            ccc_loc = 1
        else:
            ccc_vert = self._edge_list[ccc, 0]
            ccc_loc = 0
        
        self._edge_list[aaa, 0] = min(aaa_vert, end_vert)
        self._edge_list[aaa, 1] = max(aaa_vert, end_vert)
        
        self._edge_list[ccc, 0] = min(ccc_vert, start_vert)
        self._edge_list[ccc, 1] = max(ccc_vert, start_vert)
        
        if self._edge_list[aaa, aaa_loc] != aaa_vert:
            move_perm[aaa] = -move_perm[aaa]
            
        if self._edge_list[ccc, ccc_loc] != ccc_vert:
            move_perm[ccc] = -move_perm[ccc]
        
        # Find lowest index position that 2-2 move edge can be moved to ensure
        # implicit edge order. The logic behind the shift is the following. To
        # be consistent with vertex cyclic orders, we need to permute the order
        # of the edges in the edge list so that (up to cyclic permutations)
        #
        #   ax bx xy -> bx cx xy
        #   cy dy xy -> dy ay xy
        #
        # However, we also have to worry about how the ordering of the
        # vertices a, b, c, d are changed by this permutation. So to keep
        # the cyclic orders for these vertices the same, we only move the
        # edge xy in the graph edge list; this edge is placed in the
        # earliest possible spot in the graph edge list, because why not.
        #
        # Thus, there are two conditions:
        #
        #   (1) xy can be placed either after cx (if bx comes before cx), or
        #   between bx, cx (if cx comes before bx)
        #   (2) xy can be placed either after ay (if dy comes before ay), or
        #   between ay, dy (if ay comes before dy)
        #
        # In other words, for each interval, xy can go into the index *after*
        # where cx (or ay) currently are, or at the current index of cx (or dy),
        # so add one to start choices
    
        start_bc = ccc + 1
        end_bc = bbb if (ccc < bbb) else (bbb + self._num_edges)
        
        start_ad = aaa + 1
        end_ad = ddd if (aaa < ddd) else (ddd + self._num_edges)
        
        start_slot = max(start_bc, start_ad)
        end_slot = min(end_bc, end_ad)
        
        # After the 2-2 move and cyclic ordering, the four external edges can
        # come in the following orders in the edge list:
        #
        #   (1) ay bx cx dy -- place xy between ay, bx
        #   (2) ay bx dy cx -- same as (1)
        #   (3) ay cx bx dy -- place xy between cx, bx
        #   (4) ay cx dy bx -- same as (3)
        #   (5) ay dy bx cx -- place xy between ay, dy
        #
        # The ordering ay dy cx bx does not have consistent starting ordering,
        # so cannot appear as final orders. 
        
        if start_slot <= self._num_edges < end_slot:
            slot = 0
        elif end_slot < start_slot:
            
            # Have either dy < ay or bc < cx (but not both)
            
            slot = min(start_bc, start_ad)
        else:
            slot = start_slot % self._num_edges
            
        # Move edges according to chosen new position, and preserve sign of
        # edge index originally in that location

        if slot != edge_idx:
            
            moved_edge = self._edge_list[edge_idx].copy()
            
            if slot < edge_idx:
                
                # Indices move to right between slot, edge_idx
                
                for iii in range(slot, edge_idx):
                    if move_perm[iii] > 0:
                        move_perm[iii] = abs(move_perm[iii + 1])
                    else:
                        move_perm[iii] = -abs(move_perm[iii + 1])
                
                for iii in range(edge_idx, slot, -1):
                    self._edge_list[iii] = self._edge_list[iii - 1]
                    
                self._edge_list[slot] = moved_edge
                final_edge_idx = slot
                move_perm[edge_idx] = slot + 1
                
            else:
                
                # Because edge_idx < slot, need to adjust target index for
                # 2-2 move edge; indices move to left between edge_idx, slot
                
                target_slot = slot - 1
                
                for iii in range(target_slot, edge_idx, -1):
                    if move_perm[iii] > 0:
                        move_perm[iii] = abs(move_perm[iii - 1])
                    else:
                        move_perm[iii] = -abs(move_perm[iii - 1])
                
                for iii in range(edge_idx, target_slot):
                    self._edge_list[iii] = self._edge_list[iii + 1]
                    
                self._edge_list[target_slot] = moved_edge
                final_edge_idx = target_slot
                move_perm[edge_idx] = slot
                
        # Update vertex cyclic order
        
        self._parse_implicit_order()
            
        # Face information is no longer accurate, since faces have been permuted

        self._clear_face_data()
            
        # 2-2 move was successful, so return new location of edge_idx as
        # confirmation; if the move cannot be done because it violates
        # 3-connectedness, return -1 (done previously)
        
        return final_edge_idx, move_perm
        
    #-------------------------------------------------------------------------#
    
    def pachner31(self, cycle_list):
        r"""
            Perform a Pachner 3-1 move on a triangular cycle.
            
            Contract a 3-cycle to a single vertex, reconnecting
            the three external edges incident to the cycle.
            
            The graph is modified in place.
            
            Parameters
            ----------
            cycle_list : array_like of int
                Three zero-based edge indices forming a 3-cycle
                in the graph.
            
                The indices may be supplied in any order.
            
            Returns
            -------
            numpy.ndarray
                Signed integer array of shape $(|E|-3,)$,
                where $|E|$ denotes the original edge count.
            
                Entry `perm[j]` gives the signed one-based index
                of the original edge corresponding to new
                zero-based edge index $j$.
            
            Raises
            ------
            ValueError
                If the number of supplied edges is not three,
                an edge index is out of range, or the given
                edges do not form a 3-cycle.
            
            Notes
            -----
            The move changes the graph sizes according to
            
            $$
            (|V|, |E|, |F|)
            \longmapsto
            (|V|-2, |E|-3, |F|-1).
            $$
            
            The vertex with the smallest index in the 3-cycle
            is retained. The other two vertices and the three
            cycle edges are removed.
            
            Remaining vertices and edges are relabeled to
            maintain consecutive indices and the implicit
            cyclic-order convention.
            
            The returned mapping is from NEW edges to OLD edges:
            
            $$
            p[j] = s_j(i+1),
            $$
            
            where $i$ is the original index of the surviving
            edge corresponding to new edge $j$, and
            $s_j = \pm 1$ records any reversal of the
            reference direction.
            
            Face information is invalidated and recalculated
            when subsequently requested.
            
            The method does not update edge decorations or
            verify that the resulting graph is 3-connected.
        """

        cycle_list = np.asarray(cycle_list, dtype = np.int8)

        # Check that the given edges actually form a 3-cycle
    
        if len(cycle_list) != 3:
            raise ValueError('A list of three edge indices must be given.')

        if np.any((cycle_list < 0) | (cycle_list >= self._num_edges)):
            raise ValueError(f'Edge indices must be between 0 and {self._num_edges - 1}')

        # Identify vertices on 3-cycle, and external edges incident to those
        # edges. The external edges are used for the cyclic order of the one
        # vertex that is kept, so the ordering must be in correct CCW order
        # around the 3-cycle. This process could fail if there is an error in
        # how the 3-cycle edges are provided, so this is checked at end.

        # This is implemented akin to the computation of the boundary edges for
        # a face. Explicitly checking the ordering of the edges in the vertex
        # cyclic orders ensures the resulting final vertex has the correct
        # edge ordering in its vertex cyclic order as well.
        
        # NOTE: If the original graph is 3-connected, then the vertices external
        # to the 3-cycle *must* already be distinct. There is no need to check.

        vert_list = []
        ext_edges = []

        start_edge_idx = cycle_list[0]
        
        # Start process by seeing which way is CCW around cycle; find vertex
        # in this direction.

        for vert_idx in [0, 1]:
            start_vert = self._edge_list[start_edge_idx, vert_idx]
            cyc_idx = np.argmax(self._vert_cyc_order[start_vert] == start_edge_idx)
            
            if self._vert_cyc_order[start_vert, cyc_idx - 1] in cycle_list:
                curr_edge_idx = self._vert_cyc_order[start_vert, cyc_idx - 1]
                curr_vert = start_vert

                vert_list.append(curr_vert)
                ext_edges.append(self._vert_cyc_order[start_vert, cyc_idx - 2])

                break

        while True:

            # Explicitly write out both logic checks, so if there is a problem,
            # if is caused by break/if sequence that it would follow next.

            if curr_vert == self._edge_list[curr_edge_idx, 0]:
                curr_vert = self._edge_list[curr_edge_idx, 1]
            elif curr_vert == self._edge_list[curr_edge_idx, 1]:
                curr_vert = self._edge_list[curr_edge_idx, 0]

            if curr_vert == start_vert:
                break
            else:
                vert_list.append(curr_vert)

            cyc_idx = np.argmax(self._vert_cyc_order[curr_vert] == curr_edge_idx)
            if self._vert_cyc_order[curr_vert, cyc_idx - 1] in cycle_list:
                curr_edge_idx = self._vert_cyc_order[curr_vert, cyc_idx - 1]
                ext_edges.append(self._vert_cyc_order[curr_vert, cyc_idx - 2])

        if len(vert_list) != 3:
            raise ValueError('Given edges do not form a 3-cycle')

        # We are keeping the smallest label vertex v of the 3-cycle, and deleting
        # the other two. The external edges are now given by ext_edges in CCW
        # order, so rather than coming up with detailed logic to put them back
        # into the vertex cyclic order of v, we just cyclically permute them so
        # the smallest edge label is first, and will map them into the cyclic
        # order for v later.

        min_ext_idx = np.argmin(ext_edges)
        ext_edges = ext_edges[min_ext_idx:] + ext_edges[:min_ext_idx]

        # Sort other two lists; the slicing below assumes that both lists are
        # given in increasing order             

        vert_list.sort()
        [aaa, bbb, ccc] = vert_list

        cycle_list = np.sort(cycle_list)
        [xxx, yyy, zzz] = cycle_list

        # Choose lowest index vertex to keep, remove other two vertices; the
        # remaining vertex is connected to the external edges incident to the
        # 3-cycle; relabel vertices to match lower vertex number; delete all
        # three edges in cycle, and relabel others.
        
        # Added permutation tracker, to find map old label -> new label. Note
        # this is 1-based signed permutation, since some of the labels may be
        # flipped below.

        new_cyc_order = np.full((self._max_num_vert, 3), self._max_num_edges, \
                                dtype = np.uint8)
        new_edge_list = np.full((self._max_num_edges, 2), self._max_num_vert, \
                                dtype = np.uint8)
        old_edge_perm = np.full(self._num_edges - 3, self._max_num_edges, \
                                dtype = np.int8)

        new_vert = [iii for iii in range(bbb)] + [aaa] + \
            [(iii - 1) for iii in range(bbb + 1, ccc)] + [aaa] + \
            [(iii - 2) for iii in range(ccc + 1, self._num_vert)]

        new_edges = [iii for iii in range(xxx)] + [self._num_edges] + \
            [(iii - 1) for iii in range(xxx + 1, yyy)] + [self._num_edges] + \
            [(iii - 2) for iii in range(yyy + 1, zzz)] + [self._num_edges] + \
            [(iii - 3) for iii in range(zzz + 1, self._num_edges)]

        new_cyc_order[:bbb] = self._vert_cyc_order[:bbb]
        new_cyc_order[bbb : (ccc - 1)] = self._vert_cyc_order[(bbb + 1) : ccc]
        new_cyc_order[(ccc - 1) : (self._num_vert - 2)] = self._vert_cyc_order[(ccc + 1) : self._num_vert]

        new_edge_list[:xxx] = self._edge_list[:xxx]
        new_edge_list[xxx : (yyy - 1)] = self._edge_list[(xxx + 1) : yyy]
        new_edge_list[(yyy - 1) : (zzz - 2)] = self._edge_list[(yyy + 1) : zzz]
        new_edge_list[(zzz - 2) : (self._num_edges - 3)] = self._edge_list[(zzz + 1) : self._num_edges]
        
        old_edge_perm[:xxx] = np.arange(1, xxx + 1, dtype = np.int8)
        old_edge_perm[xxx : (yyy - 1)] = np.arange(xxx + 2, yyy + 1, dtype = np.int8)
        old_edge_perm[(yyy - 1) : (zzz - 2)] = np.arange(yyy + 2, zzz + 1, dtype = np.int8)
        old_edge_perm[(zzz - 2) : (self._num_edges - 3)] = np.arange(zzz + 2, self._num_edges + 1, \
                                                                     dtype = np.int8)

        # Put new cyclic order for kept vertex v, so old edge labels are 
        # remapped by new_edges permutation

        new_cyc_order[aaa] = ext_edges

        # Map old vertex, edge labels to new ones after deletions

        for row in range(self._num_vert - 2):
            for col in range(3):
                new_cyc_order[row, col] = new_edges[new_cyc_order[row, col]]

        for row in range(self._num_edges - 3):
            for col in range(2):
                new_edge_list[row, col] = new_vert[new_edge_list[row, col]]

            # Maintain start < end vertex labels;

            if new_edge_list[row, 0] > new_edge_list[row, 1]:
                new_edge_list[row, 0], new_edge_list[row, 1] = new_edge_list[row, 1], new_edge_list[row, 0]
                old_edge_perm[row] = -old_edge_perm[row]

        self._vert_cyc_order = new_cyc_order
        self._edge_list = new_edge_list

        self._num_vert -= 2
        self._num_edges -= 3
        
        # The new vertex cyclic order is correct, since it is easy to fix after
        # the 3-1 move (adjust labels downward as necessary), but the edge list
        # may be incompatible with this new order. In particular, the edge list
        # may imply a twist in the incident edges to the remaining vertex that
        # should not be there. We repair the incompatibility between the edge
        # list and vertex cyclic order at this point; note that restore_edge_perm
        # is 0-based, so no need to change it later.
        
        restore_edge_perm = self._restore_implicit_order()

        # Face information is no longer accurate, since faces have been permuted

        self._clear_face_data()
        
        if self._num_faces is not None:
            self._num_faces -= 1
            
        # Three edges and two vertices are deleted by this move; return the
        # permutation of the signed (old) edge indices, to update information
        # on edge orientations and/or parities; here, the perm is of the form
        # perm[new_edge_list_idx] = signed_old_edge_list_idx
        
        return old_edge_perm[restore_edge_perm]
        
    #-------------------------------------------------------------------------#
    # Library of standard graph families
    #-------------------------------------------------------------------------#
    
    @classmethod
    def create_prism(cls, N):
        r"""
            Construct an $N$-gonal prism as an embedded trivalent graph.
            
            The prism consists of two $N$-cycles connected by
            $N$ edges joining corresponding vertices.
            
            Parameters
            ----------
            N : int
                Number of vertices in each of the two polygonal
                faces. Must satisfy $N \geq 3$.
            
            Returns
            -------
            Graph
                A new `Graph` instance representing the
                $N$-gonal prism with its standard planar embedding.
            
            Raises
            ------
            ValueError
                If `N` is less than 3.
            
            Notes
            -----
            The resulting graph is planar, 3-regular, and
            3-connected, with
            
            $$
            |V| = 2N, \qquad |E| = 3N, \qquad |F| = N+2.
            $$
            
            The vertex cyclic orders are determined by the
            implicitly ordered edge list.
            
            The graph is created using the default vertex
            storage capacity.
            
            Examples
            --------
            Construct the square prism (cube graph):
            
            >>> G = Graph.create_prism(4)
            >>> G.num_vert, G.num_edges, G.num_faces
            (8, 12, 6)
        """
        
        if N < 3:
            raise ValueError('For n-prism creation, n >= 3')

        # Add three boundary edges of the first face
        
        edge_list = [[0, 1], [1, 2 * N - 1], [2 * N - 2, 2 * N - 1]]

        # Add the remaing three boundary edges of an adjacent face. When the
        # first edge is added, this allows the remaining edge of the first face
        # to be added as well.
        
        edge_list += [[0, 2], [0, 2 * N - 2], [2, 3], [1, 3]]

        # Work around the n-prism, completing all but last face

        for iii in range(1, N - 2):
            edge_list += [[2 * iii, 2 * (iii + 1)], [2 * (iii + 1), 2 * (iii + 1) + 1], \
                         [2 * iii + 1, 2 * (iii + 1) + 1]]
                
        # Add last two edges

        edge_list += [[2 * N - 4, 2 * N - 2], [2 * N - 3, 2 * N - 1]]
        
        # Return Graph object with given edge list
        
        return cls(edge_list)
    
    #-------------------------------------------------------------------------#