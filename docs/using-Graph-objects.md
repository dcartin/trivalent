# Working with Graph objects

This package allows one to create a Graph object representing a 3-regular
graph embedded in a 2d surface. This graph is thus dual to the simplicial
triangulation of the surface. One can study the properties of the graph, and
use Pachner moves (dual to those for the triangulation) to obtain other graphs.
In this guide, we will show the basic workflow for using trivalent, including
the creation of a graph, understanding its properties, comparing it to other
graphs, and using the graph Pachner moves to create other graphs. First, import
the Graph class from trivalent. This includes various class functions to add in
understanding the structure of the graph, along with the ability to apply the
Pachner moves to the graph.

```python
from trivalent import Graph
```

## Graph creation

To create a graph, use the `Graph` command. Usually, the argument will be a
list of edges for the graph; as described below, this list is ordered
implicitly to preserve the cyclic orders of the edges around the graph
vertices. If no parameters are given, the graph will be created with zero
vertices or edges.

For this guide, we will create a planar tetrahedral graph. To do this, we need
to provide an appropriate edge list for the `Graph` object. There are a few
guidelines:

* Vertices are numbered from 0 to $|V| - 1$. Note that a trivalent 3-regular
graph with $|V|$ vertices and $|E|$ edges must have $3|V| = 2|E|$.
* An edges is given by an ordered list of vertex labels. This will be a list of
$|E|$ two-element lists.
* The edges incident to each vertex are attached to that vertex in
counterclockwise order as given in the provided edge list. For example, if the
edge list contains `[[0, 1] ... [0, 2] ... [0, 3]]`, then the vertices 1, 2, 3
appear in CCW order around vertex 0.

An acceptable edge list for the planar tetrahedral graph is given below.

```python
# Define edge list for later use

tet_edge_list = [[0, 1], [0, 2], [1, 3], [0, 3], [1, 2], [2, 3]]

# Define tetrahedral Graph object

G = Graph(tet_edge_list)
```

## Graph properties

Now that we have a `Graph` object, we can look at a few of its properties. Note
that, unlike the number of vertices $|V|$ and edges $|E|$, the number of faces
$|F|$ requires a computation of the faces from the edge list and cyclic orders
of the graph. However, this is done automatically anytime properties of the
faces are requested.

```python
print(f'Number of vertices   : {G.num_vert}')
print(f'Number of edges      : {G.num_edges}')
print(f'Number of faces      : {G.num_faces}')

print(f'\nEuler characteristic : {G.num_vert - G.num_edges + G.num_faces}')
```

    Number of vertices   : 4
    Number of edges      : 6
    Number of faces      : 4
    
    Euler characteristic : 2

We can also get the edge list, and the cyclic orders of the incident edges
around each vertex. Each of these is given as a `numpy` array.

The edge list is identical to that used to create the `Graph` object in the
first place. It is a list of ordered vertex label pairs.

```python
for edge_label, edge in enumerate(G.edge_list):
    print(f'edge {edge_label} : {edge}')
```

    edge 0 : [0 1]
    edge 1 : [0 2]
    edge 2 : [1 3]
    edge 3 : [0 3]
    edge 4 : [1 2]
    edge 5 : [2 3]

The cyclic orders use these edge labels (i.e. their indices in the original
edge list) to give the CCW ordering of the incident edges around each vertex.

```python
for vert_label, cyc_order in enumerate(G.vert_cyc_order):
    print(f'vert {vert_label} : {cyc_order}')
```

    vert 0 : [0 1 3]
    vert 1 : [0 2 4]
    vert 2 : [1 4 5]
    vert 3 : [2 3 5]

To see how these two lists match up, as an example, take vertex 2. In the list from `Graph.edge_list`, this vertex appears in the pairs describing edges 1, 4, and 5. This is exactly the cyclic order given in `Graph.vert_cyc_order` for this vertex. Note that the edge labels must be in increasing order, due to the implicit ordering of the edges given in the edge list.

The faces of the graph are created when `Graph.find_faces()` is invoked. One way to see these faces is to show their boundary edges using `Graph.face_bndy_list`. This gives a list of 1-based signed lists, each giving the boundary edges when traveling around the face in a counterclockwise direction (so the face is always on the left in the direction of travel). The sign for each edge label gives whether this travel around the face is with, or against, the canonical direction of the edge $(u, v)$, from $u$ to $v$.

```python
for face_label, face_edge_list in enumerate(G.face_bndy_list):
    print(f'Face label {face_label} : {face_edge_list}')
```

    Face label 0 : [1, 5, -2]
    Face label 1 : [-1, 4, -3]
    Face label 2 : [2, 6, -4]
    Face label 3 : [3, -6, -5]

Each edge appears twice in the list above, for its two incident faces, once with each sign choice. The same information can be given from an edge-oriented view by using `Graph.face_idx_list`. We can determine the face "to the left" or "right" of the edge from this direction. The faces incident to each edge are given as the pair `[left_face_idx, right_face_idx]` in the list below.

```python
for edge_label, face_pair in enumerate(G.face_idx_list):
    print(f'edge {edge_label} : {face_pair}')
```

    edge 0 : [0 1]
    edge 1 : [2 0]
    edge 2 : [3 1]
    edge 3 : [1 2]
    edge 4 : [0 3]
    edge 5 : [2 3]

As one can see, each face appears three times in the list. Thus, each of the
faces have three boundary edges, as expected for the tetrahedron.

```python
print(f'Face sizes : {G.face_size_list}')
```

    Face sizes : [3 3 3 3]

## Graph isomorphisms and symmetry groups

Next we look at determining whether two graphs are identical or not. The same
algorithm, when comparing the graph to itself, can give the symmetry group of
the graph.

The implicit ordering of the edge list is invariant under permutations, since
all that matters is the incident edges are given in the correct CCW ordering
around each vertex. Thus, we can take the same edge list as above, but use a
cyclic permutation of the edges in the list to define a new graph `H`. For
example, suppose we take the first three edges in the list, and move them to
the end of the list.

```python
H = Graph([[0, 3], [1, 2], [2, 3], [0, 1], [0, 2], [1, 3]])
```

This graph should be equivalent to the original `Graph` object `G`. The first
time this is run may take a while, since it relies on a `numba` procedure,
which must be complied.

```python
print(f'G == H : {G == H}')
```

    G == H : True

The same thing is true if we permute the vertex labels. Suppose we use the
original labels from `G`, but increase them by 1 mod 4.

```python
K = Graph([[1, 2], [1, 3], [2, 0], [1, 0], [2, 3], [3, 0]])

print(f'G == K : {G == K}')
```

    G == K : True

The same algorithm is used to find all the symmetries of a graph. These are
reported as signed edge labels, where a label is positive if its original
canonical orientation is preserved, and negative if it is reversed. Again, a
`numba` procedure is used to do this, so the first use may take a longer time.

For the tetrahedron, there are a total of 12 symmetries, as given below. The
identity transformation should be the first listed.

```python
for sym in G.find_sym():
    print(sym)
```

    [1 2 3 4 5 6]
    [ 4  1 -6  2 -3  5]
    [ 2  4 -5  1  6 -3]
    [-1  3  2  5  4 -6]
    [ 5 -1  6  3 -2  4]
    [ 3  5 -4 -1 -6 -2]
    [-2 -5  4  6  1  3]
    [ 6 -2 -3 -5 -4  1]
    [-5  6 -1 -2  3 -4]
    [-3 -4  5 -6 -1  2]
    [-6 -3 -2 -4 -5 -1]
    [-4 -6  1 -3  2 -5]

## Pachner moves

Finally, we cover the graph Pachner moves on the graphs, starting with the 2-2
move `Graph.pachner22()`. This move preserves the number of vertices, so the
initial and final graphs have the same $|V|$. However, in general they will not
be isomorphic graphs. Thus, given an edge label `label`,
`Graph.pachner22(label)` performs a 2-2 move on that edge, and returns
`[new_label, perm]`. Here `new_label` is the new edge index for the edge
used in the move, and `perm` is a `numpy` array giving the permutation from the
original edge labels to the those of the final graph. This permutation is a
signed permutation, indicating if any edge orientations are reversed during the
mapping from the old to new graphs; thus, it is 1-based, instead of 0-based, so
that every edge index can have a sign as appropriate.

These graph moves are required to preserve the 3-connectedness of the graph, so
if this property would be violated, then the move does not go through. In
particular, a 2-2 move cannot be done on any edge of the tetrahedron without
giving multiple edges between two vertices. In this case, instead of returning
the edge label, it returns `[-1, []]`.

```python
print(G.pachner22(0))
```

    (-1, array([], dtype=int8))

All graphs with $|V| \ge 6$ have at least one possible 2-2 move on them, so we
use the unique $|V| = 6$ graph as an example. Obviously, with only one graph
possible, the 2-2 move takes this graph to itself, but the edges are permuted.

```python
# Define edge list for later use

six_vert_edge_list = [[0, 1], [0, 4], [0, 5], [1, 2], [3, 4], \
                      [1, 3], [4, 5], [2, 5], [2, 3]]

# Define Graph object, act with 2-2 move

P = Graph(six_vert_edge_list)
new_edge, perm = P.pachner22(0)

print(f'New edge idx : {new_edge}')
print(f'Permutation  : {perm}')
```

    New edge idx: 3
    Permutation : [4 1 2 3 5 6 7 8 9]

Note that `Graph.pachner22()` modifies the graph, so that it is not isomorphic
to the original graph. This is easier to see if we compare two $|V| = 8$ graphs.
Let `C` be the 4-prism graph (i.e. the projection of the cube onto the plane),
and `D` be the only other graph with eight vertices; the latter is known as the
standard graph $\Delta_8$.

```python
# Define edge lists separately for later use

cube_edge_list = [[0, 1], [0, 6], [0, 7], [1, 2], [4, 6], [6, 7], \
                  [3, 4], [2, 5], [5, 7], [1, 3], [2, 3], [4, 5]]
std_edge_list = [[0, 1], [2, 6], [1, 6], [0, 4], [3, 4], [2, 5], \
                 [0, 7], [5, 7], [1, 3], [2, 3], [4, 5], [6, 7]]

# Define Graph objects, show they are not isomorphic graphs

C = Graph(cube_edge_list)
D = Graph(std_edge_list)

print(f'C == D : {C == D}')
```

    C == D : False

The only 2-2 move that takes the graph `D` to `C` is the one on its first listed edge. When this is done, then `D` is now the 4-prism graph, rather than its original form. This allows a check for what the resulting final graph is, from the known list of all graphs with a certain number of vertices.

```python
# Perform 2-2 move on edge 0

D.pachner22(0)

# Identify which edge list D now matches with

print(f'D is standard graph : {D == Graph(std_edge_list)}')
print(f'D is cubic graph    : {D == Graph(cube_edge_list)}')
```

    D is standard graph : False
    D is cubic graph    : True

`Graph.pachner13()` takes a vertex label, and modifies the graph by changing this vertex into a 3-cycle. It returns the labels for the newly created edges, which will all be at the end of the graph edge list.

We can show that acting on a vertex of the tetrahedron gives the unique six-vertex graph.

```python
# Perform 1-3 move on vertex 0

T = Graph(tet_edge_list)
print(f'New edge labels: {T.pachner13(0)}')

# Identify new form of T

print(f'T is tetrahedron      : {T == Graph(tet_edge_list)}')
print(f'T is six-vertex graph : {T == Graph(six_vert_edge_list)}')
```

    New edge labels: (6, 7, 8)
    T is tetrahedron      : False
    T is six-vertex graph : True

Similarly, `Graph.pachner31()` takes a list of three edge labels (which must form a 3-cycle), and replaces this cycle with a single vertex. This is the inverse operation of the 1-3 move above. The function returns the 1-based signed permutation from the *new* edge labels to the *old*. This facilitates mapping over any edge decorations on the graph. Note that this permutation cannot be given the opposite way, since two of the edges are removed.

```python
# Perform 3-1 move on edges 1, 2, 6

U = Graph(six_vert_edge_list)
print(f'Signed permutation: {U.pachner31([1, 2, 6])}')

# Identify new form of U

print(f'U is tetrahedron      : {U == Graph(tet_edge_list)}')
print(f'U is six-vertex graph : {U == Graph(six_vert_edge_list)}')
```

    Signed permutation: [ 1  4 -5  6 -8  9]
    U is tetrahedron      : True
    U is six-vertex graph : False