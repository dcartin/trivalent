# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 06:47:05 2026

@author: cartin
"""

import numba as nb
import numpy as np

from collections import Counter, deque

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

@nb.njit(cache = True)
def _permute_orient_bits(orient_bits, signed_perm, num_edges = 20):
    
    # The signed permutation is 1-based and records changes in edge orientation
    # due to the symmetry element; thus, if the orientation flips, the resulting
    # orientation direction must flip as well, done by XOR on orientation value
    
    perm_orient_bits = 0
    
    for idx in range(num_edges):
        target = signed_perm[idx]
        target_idx = abs(target) - 1
        current_bit = (orient_bits >> idx) & 1
        
        if target < 0:
            current_bit ^= 1
            
        perm_orient_bits |= (current_bit << target_idx)
        
    return perm_orient_bits

#-----------------------------------------------------------------------------#

@nb.njit(cache = True)
def _permute_parity_bits(parity_bits, signed_perm, num_edges = 20):
    
    # There is no check of edge orientation flip necessary, so no XOR needed,
    # simply move old parity value to its new location
    
    perm_parity_bits = 0
    
    for idx in range(num_edges):
        target_idx = abs(signed_perm[idx]) - 1
        current_bit = (parity_bits >> idx) & 1
        
        perm_parity_bits |= (current_bit << target_idx)
        
    return perm_parity_bits

#-----------------------------------------------------------------------------#

@nb.njit(cache = True)
def _find_min_orient_bits(orient_bits, sym_grp, num_edges = 20):
    
    min_orient_bits = orient_bits
    
    for idx in range(sym_grp.shape[0]):
        temp_bits = _permute_orient_bits(orient_bits, sym_grp[idx], num_edges)
        if temp_bits < min_orient_bits:
            min_orient_bits = temp_bits
            
    return min_orient_bits

#-----------------------------------------------------------------------------#

@nb.njit(cache = True)
def _find_min_parity_bits(parity_bits, sym_grp, num_edges = 20):
    
    min_parity_bits = parity_bits
    
    for idx in range(sym_grp.shape[0]):
        temp_bits = _permute_parity_bits(parity_bits, sym_grp[idx], num_edges)
        if temp_bits < min_parity_bits:
            min_parity_bits = temp_bits
            
    return min_parity_bits

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
    roots = []
    for i in range(num_edges):
        if parent[i] == i:
            roots.append(i)
            
    # Convert list to array for Numba return types
    return np.array(roots, dtype=np.int32)

#=============================================================================#
# Graph class
#=============================================================================#

class Graph:
    
    DEFAULT_MAX_VERT = 50       # Maximum number of vertices for any possible graph
    DEFAULT_MAX_SYM = 60        # Maximum number of graph symmetries (for dodecahedron)
    
    #-------------------------------------------------------------------------#
    
    def __init__(self, ordered_edge_list = None, max_num_vert = None, validate = True):
        """
            Construct an instance using only an implicitly ordered edge list
        """
        
        # Default placeholders for quantities to be computed when needed
        
        self._num_faces = None
        self._face_idx_list = None
        self._face_size_list = None
        
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
    
        if self._max_num_vert > 127:
            raise ValueError("max_num_vert exceeds int8 capacity of 127")
            
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
        return self._num_vert
                
    #-------------------------------------------------------------------------#
    
    @property
    def num_edges(self):
        return self._num_edges
            
    #-------------------------------------------------------------------------#
    
    @property
    def num_faces(self):
    
        if self._face_idx_list is None:
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
            consistent
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
    
    def __eq__(self, other):
        
        # Verify whether number of vertices, edges are equal
        
        if self._num_vert != other._num_vert or self._num_edges != other._num_edges:
            return False
        
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
            return False
        
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
        
        return result > 0
    
    #-------------------------------------------------------------------------#
    
    def find_noniso_edges(self):
        pass
    
    #-------------------------------------------------------------------------#

    def find_sym(self):
        """
            Find all signed symmetries of the graph, where sign indicates
            reversal of edge direction.
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
    
    @property
    def edge_list(self):
        """
            Return only those edges that are currently active
        """
        
        return self._edge_list[:self._num_edges]
        
    #-------------------------------------------------------------------------#
    
    @property
    def vert_cyc_order(self):
        """
            Return only cyclic orders of currently active vertices
        """
        
        return self._vert_cyc_order[:self._num_vert]
        
    #-------------------------------------------------------------------------#
    
    @property
    def face_idx_list(self):
        """
            Return left-, right-hand face indices for all active edges
        """
        
        if self._face_idx_list is None:
            self.find_faces()
        
        return self._face_idx_list[:self._num_edges]
        
    #-------------------------------------------------------------------------#
    
    @property
    def face_size_list(self):
        """
            Return face sizes
        """
        
        if self._face_idx_list is None:
            self.find_faces()
        
        return self._face_size_list[:self._num_edges]
        
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
        """
            Given edge list, calculate the labels of the faces on the left,
            right sides of each edge; record the face indices, so that we have
            a constant size array for all information.
        """
        
        # The face indices are recorded in a master list for each edge, with
        # face_idx_list[edge_idx][0] the face index for the left-hand face,
        # and face_idx_list[edge_idx][1] the right-hand face
        
        self._face_idx_list = np.full((self._max_num_edges, 2), self._max_num_vert, \
                                     dtype = np.uint8)
        
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
                
        # Update number of faces, list of face sizes indices by face_idx
        
        self._num_faces = (face_idx + 1)
        self._face_size_list = np.array(face_signature_list, dtype = np.uint8)
        
    #-------------------------------------------------------------------------#
    
    def pachner13(self, vert_idx):
        """
            Use the Pachner 1-3 move to expand the given vertex into a 3-cycle
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
        
        self._face_idx_list = None
        self._face_size_list = None
        
        if self._num_faces is not None:
            self._num_faces += 2
            
        # This move always succeeds, and the original edges remain in place in
        # the edge list. The new edges are added to the end as the last three,
        # and the last two vertices in the vertex cyclic order are the new
        # vertices.
        
        return self._num_edges - 3, self._num_edges - 2, self._num_edges - 1
    
    #-------------------------------------------------------------------------#
    
    def pachner22(self, edge_idx):
        """
            Use a Pachner 2-2 move for given edge, flipping the connections
            with its four neighboring vertices
        """
        
        if (edge_idx < 0) or (edge_idx >= self._num_edges):
            raise ValueError(f'Edge index needs to be between 0 and {self._num_vert}')
        
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
        
        # Find lowest index position that 2-2 move edge cna be moved to ensure
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
        #   (1) xy must be placed either after cx (if bx comes before cx), or
        #   between bx, cx (if cx comes before bx)
        #   (2) xy must be placed either after ay (if dy comes before ay), or
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
        # The orderings ay dy cx bx does not have consistent starting orderings,
        # so cannot appear as final orders. 
        
        if start_slot <= self._num_edges < end_slot:
            slot = 0
        elif end_slot < start_slot:
            
            # Have either dy < ay or bc < cx (but not both)
            
            slot = min(start_bc, start_ad)
        else:
            slot = start_slot % self._num_edges
            
        # Move edges according to chosen new position

        if slot != edge_idx:
            
            moved_edge = self._edge_list[edge_idx].copy()
            
            if slot < edge_idx:
                
                # Indices move to right between slot, edge_idx
                
                for iii in range(slot, edge_idx):
                    move_perm[iii] = move_perm[iii + 1]
                
                for iii in range(edge_idx, slot, -1):
                    self._edge_list[iii] = self._edge_list[iii - 1]
                    
                self._edge_list[slot] = moved_edge
                move_perm[edge_idx] = slot + 1
                
            else:
                
                # Because edge_idx < slot, need to adjust target index for
                # 2-2 move edge; indices move to left between edge_idx, slot
                
                target_slot = slot - 1
                
                for iii in range(target_slot, edge_idx, -1):
                    move_perm[iii] = move_perm[iii - 1]
                
                for iii in range(edge_idx, target_slot):
                    self._edge_list[iii] = self._edge_list[iii + 1]
                    
                self._edge_list[target_slot] = moved_edge
                move_perm[edge_idx] = slot
                
        # Update vertex cyclic order
        
        self._parse_implicit_order()
            
        # Face information is no longer accurate, since faces have been permuted
        
        self._face_idx_list = None
        self._face_size_list = None
            
        # 2-2 move was successful, so return edge_idx as confirmation; if the
        # move cannot be done because it violates 3-connectedness, return -1
        
        return edge_idx, move_perm
        
    #-------------------------------------------------------------------------#
    
    def pachner31(self, cycle_list):
        """
            Given a list of edge indices that form a 3-cycle, reduce the face
            to a single vertex.
        """

        cycle_list = np.asarray(cycle_list, dtype = np.int8)

        # Check that the given edges actually form a 3-cycle
    
        if len(cycle_list) != 3:
            raise ValueError('A list of three edge indices must be given.')

        if np.any((cycle_list < 0) | (cycle_list > self._num_edges)):
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

        self._face_idx_list = None
        self._face_size_list = None
        
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
        """
        Construct an n-prism Graph object with 2N vertices.

        Parameters
        ----------
        N : int
            The order of the prism base polygons (number of vertices in the 
            top/bottom faces). Must be at least 3.

        Returns
        -------
        Graph
            An initialized Graph instance representing the n-prism topology.

        Raises
        ------
        ValueError
            If `N < 3`, as a valid 3-connected trivalent prism requires at 
            least a triangular base.
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