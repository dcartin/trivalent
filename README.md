# Exploring 3-regular graphs

## Purpose

This package allows one to create a Graph object representing a 3-regular graph embedded in a 2d surface.
This graph is thus dual to the simplicial triangulation of the surface. One can study the properties of
the graph, and use Pachner moves (dual to those for the triangulation) to obtain other graphs.

Below is a brief description of the package. For more information, see the
`marimo` [notebook](https://molab.marimo.io/github/dcartin/trivalent/blob/master/docs/trivalent-guide.py)
for an overview of using `Graph` objects.

## Dependencies

- `numba >= 0.60`
- `numpy >= 2.0`
- `python >= 3.11`

## Main features

- Define a Graph object using an implicitly ordered edge list of vertex pairs
- Return the properties of the graph:
   - number of vertices $|V|$
   - number of edges $|E|$
   - number of faces $|F|$
   - implicitly ordered edge list
   - cyclic ordering of edges CCW around each vertex
   - boundary edges for each face
   - number of boundary edges for each face
- Test isomorphism between two Graphs
- Find the symmetry group of a Graph, written as a signed permutation of the edges
- Apply the graph Pachner moves, dual to those for a simplicial triangulation of a manifold
- Represent edge orientations and $Z_2$ parities as bitmasks or packed graph states
- Transform and canonicalize edge decorations under graph automorphisms, including global orientation reversal

## Example

```python
from trivalent import (
    Graph,
    encode_orient_state,
    find_min_orient_bits,
    reverse_orientation,
)

# Construct a triangular prism

G = Graph.create_prism(3)

# Obtain its symmetry group

symmetry_grp = G.find_sym()
num_edges = G.num_edges

# Example edge orientation bitmask

orient_bits = 0b101001011

# Find its canonical representative

min_bits = find_min_orient_bits(orient_bits, symmetry_grp, num_edges)

# Pack graph ID and orientation bits into a state

state = encode_orient_state(0, orient_bits, num_edges)

# Reverse every orientation and canonicalize

reversed_state = reverse_orientation(
    state, num_edges, canonical = True, sym_grp = symmetry_grp
)
```

## Principal functions and properties

Given a Graph object `G`, below are a list of all functions and properties defined for the graph.

- `G.num_vert`, `G.num_edges`, `G.num_faces`: number of vertices $|V|$, edges $|E|$, and faces $|F|$ for the graph
- `G.edge_list`: a list of all graph edges, as pairs of vertex labels
- `G.vert_cyc_order`: a list of the edge CCW cyclic orders around each vertex
- `G.face_adj_matrix`: a matrix giving adjacency of graph faces
- `G.face_bndy_list`: a list of the boundary edges of each face, ordered counterclockwise
- `G.face_idx_list`: a list of the face indices on either side of each edge
- `G.face_size_list`: a list of the number of boundary edges for each graph face
- `G == H`: test for isomorphism between `G` and second Graph object `H` (gives True/False)
- `G.is_isomorphic(H)`: tests for isomorphism, returns 1-based signed permutation between edges;
isomorphism preserves cyclic order of edges at each vertex, so mirror graphs are distinguished
- `G.find_min_*_state(state, sym_grp)`: for all elements of a symmetry group, find lexicographic minimum state
- `G.find_noniso_edges()`: find list of non-isomorphic edges under graph symmetry group, or subgroup preserving edge
properties (orientation, parity)
- `G.find_sym()`: find all signed permutations of the graph, as 1-based lists; the edge sign represents an orientation
flip of the edge
- `G.map_*_state(state, H)`: Map current graph state to match edge list of isomorphic graph `H`
- `G.permute_*_state(state, perm)`: For given graph state and signed edge permutation, find permuted state
- `G.pachner13(vert_label)`: perform the Pachner 1-3 move on the graph, given a vertex label as argument
- `G.pachner22(edge_label)`: perform the Pachner 2-2 move on the graph, given an edge label as argument
- `G.pachner31(cycle_list)`: perform the Pachner 3-1 move on the graph, given the edge labels of a 3-cycle

There are also functions independent of Graph objects, which allow useful operations on pre-computed information. 
This information may be edge decorations encoded into a bitmask, or a state where the bitmask is
combined with a standard graph ID into an integer. Edge decorations are represented as integer bitmasks, using $|E|$
bits for orientations or parities separately, or $2|E|$ bits for combined decorations. A packed state additionally
contains a graph ID. The graph ID is supplied by the user or an external enumeration; it is not calculated by the
encoding functions. For example, a edge orientation bitmask `orient_bits` for graph 10 (binary `0b1010`)
with `num_edges` edges, the corresponding state would be `(0b1010 << num_edges) | orient_bits`.

- `decode_*_state(state)`: given state, return graph ID and edge decoration bitmask(s)
- `encode_*_state(state)`: given graph ID and edge decoration bitmask(s), return state
- `permute_*_bits(bitmask)`: permute bitmask using graph symmetry element
- `permute_*_state(state)`: permute states using graph symmetry element
- `find_min_*_bits(bitmask)`: for all elements of a symmetry group, find lexicographic minimum bitmask
- `find_min_*_state(state)`: for all elements of a symmetry group, find lexicographic minimum state
- `reverse_orientation(state)`: reverse edge orientations of a given state, with option to give minimal canonical
version

For `permute_*_bits(bitmask)`, the bitmasks represent either edge orientations or parities, while
`reverse_orientation` takes a state with graph ID and edge orientation bitmask. For all other functions,
the bitmask or state can represent only orientations or parities, or a combined bitmask in the form
`(orient_bits << num_edges) | parity_bits`.

## Standard graphs

- `Graph.create_prism(n)`: $n$-prism graph with $2n$ vertices

## Data files

The data files included in `/data` give information about the possible graph states, including the possibility of edge orientation and parities. For smaller vertex numbers, the adjacency matrix of the Pachner graph under the 2-2 move is also stored. Larger files are not included here, but can be found on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22986927.svg)](https://doi.org/10.5281/zenodo.22986927)
