# Exploring 3-regular graphs

## Purpose

This package allows one to create a Graph object representing a 3-regular graph embedded in a 2d surface. This graph is thus dual to the simplicial triangulation of the surface. One can study the properties of the graph, and use Pachner moves (dual to those for the triangulation) to obtain other graphs.

Below is a brief description of the package. For more information, see the `marimo` [notebook](https://molab.marimo.io/github/dcartin/trivalent/blob/master/docs/trivalent-guide.py) for an overview of using `Graph` objects.

## Dependencies

- `numba`
- `numpy`

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

## List of functions and properties

Given a Graph object `G`, below are a list of all functions and properties defined for the graph.

- `G = H`: test for isomorphism between `G` and second Graph object `H`
- `G.num_vert`, `G.num_edges`, `G.num_faces`: number of vertices $|V|$, edges $|E|$, and faces $|F|$ for the graph
- `G.edge_list`: a list of all graph edges, as pairs of vertex labels
- `G.vert_cyc_order`: a list of the edge CCW cyclic orders around each vertex
- `G.face_idx_list`: a list of the face indices on either side of each edge
- `G.face_size_list`: a list of the number of boundary edges for each graph face
- `G.find_noniso_edges()`: find list of non-isomorphic edges under graph symmetry group, or subgroup preserving edge properties (orientation, parity)
- `G.find_sym()`: find all signed permutations of the graph, as 1-based lists; the edge sign represents an orientation flip of the edge
- `G.pachner13(vert_label)`: perform the Pachner 1-3 move on the graph, given a vertex label as argument
- `G.pachner22(edge_label)`: perform the Pachner 2-2 move on the graph, given an edge label as argument
- `G.pachner31(cycle_list)`: perform the Pachner 3-1 move on the graph, given the edge labels of a 3-cycle

## Standard graphs available

- `G.create_prism(n)`: $n$-prism graph with $2n$ vertices