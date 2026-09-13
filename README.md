# Exploring 3-regular graphs dual to 2d surfaces

## Purpose

This package allows one to create a Graph object representing a 3-regular graph embedded in a 2d surface. This graph is thus dual to the simplicial triangulation of the surface. One can study the properties of the graph, and use Pachner moves (dual to those for the triangulation) to obtain other graphs.

Below is a brief description of the package. For more information, see the `marimo` notebooks located in the folder `docs/`.

# Dependencies

- `numba`
- `numpy`

## Main features

- Define a Graph object using an implicitly ordered list of vertex pairs
- Return the properties of the graph:
   - number of vertices $|V|$
   - number of edges $|E|$
   - cyclic ordering of edges CCW around each vertex
   - number of faces $|F|$
   - boundary edges for each face
- Test isomorphism between two Graphs
- Find the symmetry group of a Graph, written as a signed permutation of the edges
- Apply the graph Pachner moves, dual to those for a simplicial triangulation of a manifold

## List of functions and properties

Given a Graph object `G`, below are a list of all functions and properties defined for the graph.

- `G = H`: test for isomorphism of `G` with a second Graph object `H`
- `G.find_sym()`: find all signed permutations of the graph; the edge sign represents an orientation flip of the edge
- `G.active_edges()`: a list of all graph edges
- `G.active_vert()`: a list of the edge cyclic orders around each vertex
- `G.active_face_indices()`: a list of the face indices on either side of each edge
- `G.active_face_sizes()`: a list of the number of boundary edges for each graph face
- `G.find_faces()`: find the faces of the graph
- `G.pachner13()`, `G.pachner22()`, `G.pachner31()`: perform the Pachner 1-3, 2-2, and 3-1 moves on the graph