import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <h1> Using trivalent</h1>

    This package allows one to create a Graph object representing a 3-regular graph embedded in a 2d surface. This graph is thus dual to the simplicial triangulation of the surface. One can study the properties of the graph, and use Pachner moves (dual to those for the triangulation) to obtain other graphs.

    In this guide, we will show the basic workflow for using `trivalent`, including the creation of a graph, understanding its properties, comparing it to other graphs, and using the graph Pachner moves to create other graphs.

    First, import the `Graph` class from trivalent. This includes various class functions to add in understanding the structure of the graph, along with the ability to apply the Pachner moves to the graph.
    """)
    return


@app.cell
def _():
    from trivalent import Graph

    return (Graph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Graph creation

    To create a graph, use the `Graph` command. Usually, the argument will be a list of edges for the graph; as described below, this list is ordered implicitly to preserve the cyclic orders of the edges around the graph vertices. If no parameters are given, the graph will be created with zero vertices or edges.

    For this guide, we will create a planar tetrahedral graph. To do this, we need to provide an appropriate edge list for the `Graph` object. There are a few guidelines:

    * Vertices are numbered from 1 to $|V|$. Note that a trivalent 3-regular graph with $|V|$ vertices and $|E|$ edges must have $3|V| = 2|E|$.
    * An edges is given by an ordered list of vertex labels. This will be a list of $|E|$ two-element lists.
    * The edges incident to each vertex are attached to that vertex in counterclockwise order as given in the provided edge list. For example, if the edge list contains `[[1, 2] ... [1, 3] ... [1, 4]]`, then the vertices 2, 3, 4 appear in CCW order around vertex 1.

    An acceptable edge list for the planar tetrahedral graph is given below.
    """)
    return


@app.cell
def _(Graph):
    # Define edge list for later use

    tet_edge_list = [[0, 1], [0, 2], [1, 3], [0, 3], [1, 2], [2, 3]]

    # Define tetrahedral Graph object

    G = Graph(tet_edge_list)
    return G, tet_edge_list


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Graph properties

    Now that we have a `Graph` object, we can look at a few of its properties. Note that, unlike the number of vertices $|V|$ and edges $|E|$, the number of faces $|F|$ requires a computation of the faces from the edge list and cyclic orders of the graph. However, this is done automatically anytime properties of the faces are requested.
    """)
    return


@app.cell
def _(G):
    print(f'Number of vertices   : {G.num_vert}')
    print(f'Number of edges      : {G.num_edges}')
    print(f'Number of faces      : {G.num_faces}')

    print(f'\nEuler characteristic : {G.num_vert - G.num_edges + G.num_faces}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also get the edge list, and the cyclic orders of the incident edges around each vertex. Each of these is given as a `numpy` array.

    The edge list is identical to that used to create the `Graph` object in the first place. It is a list of ordered vertex label pairs.
    """)
    return


@app.cell
def _(G):
    for edge_label, edge in enumerate(G.active_edges):
        print(f'edge {edge_label} : {edge}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The cyclic orders use these edge labels (i.e. their indices in the original edge list) to give the CCW ordering of the incident edges around each vertex.
    """)
    return


@app.cell
def _(G):
    for vert_label, cyc_order in enumerate(G.active_vert):
        print(f'vert {vert_label} : {cyc_order}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To see how these two lists match up, as an example, take vertex 2. In the list from `G.active_edges`, this vertex appears in the pairs describing edges 1, 4, and 5. This is exactly the cyclic order given in `G.active_vert` for this vertex. Note that the edge labels must be in increasing order, due to the implicit ordering of the edges given in the edge list.

    When the faces are created, they are numbered by their own labels; these labels can be obtained by using `G.active_face_indices`.
    """)
    return


@app.cell
def _(G):
    for arrow_label, face_pair in enumerate(G.active_face_indices):
        print(f'edge {arrow_label} : {face_pair}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As one can see, each face appears three times in the list. Thus, each of the faces have three boundary edges, as expected for the tetrahedron.
    """)
    return


@app.cell
def _(G):
    print(f'Face sizes : {G.active_face_sizes}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Graph isomorphisms and symmetry groups

    Next we look at determining whether two graphs are identical or not. The same algorithm, when comparing the graph to itself, can give the symmetry group of the graph.

    The implicit ordering of the edge list is invariant under permutations, since all that matters is the incident edges are given in the correct CCW ordering around each vertex. Thus, we can take the same edge list as above, but use a cyclic permutation of the edges in the list to define a new graph `H`. For example, suppose we take the first three edges in the list, and move them to the end of the list.
    """)
    return


@app.cell
def _(Graph):
    H = Graph([[0, 3], [1, 2], [2, 3], [0, 1], [0, 2], [1, 3]])
    return (H,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This graph should be equivalent to the original `Graph` object `G`. The first time this is run may take a while, since it relies on a `numba` procedure, which must be complied.
    """)
    return


@app.cell
def _(G, H):
    print(f'G = H : {G == H}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same thing is true if we permute the vertex labels. Suppose we use the original labels from `G`, but increase them by 1 mod 4.
    """)
    return


@app.cell
def _(G, Graph):
    K = Graph([[1, 2], [1, 3], [2, 0], [1, 0], [2, 3], [3, 0]])

    print(f'G = K : {G == K}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same algorithm is used to find all the symmetries of a graph. These are reported as signed edge labels, where a label is positive if its original orientation is preserved, and negative if it is reversed. Again, a `numba` procedure is used to do this, so the first use may take a longer time.

    For the tetrahedron, there are a total of 12 symmetries, as given below. The identity transformation should be the first listed.
    """)
    return


@app.cell
def _(G):
    for sym in G.find_sym():
        print(sym)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pachner moves

    Finally, we cover the graph Pachner moves on the graphs; at the moment, only the 2-2 move `Graph.pachner22()` is fully implemented. This move preserves the number of vertices, so the initial and final graphs have the same $|V|$. However, in general they will not be isomorphic graphs. Thus, given an edge label `label`, `Graph.pachner22(label)` performs a 2-2 move on that edge, and returns `[label, perm]`. Here `perm` is a `numpy` array giving the permutation from the original edge labels to the those of the final graph; this is a signed permutation, if any edge orientations are reversed.

    These graph moves are required to preserve the 3-connectedness of the graph, so if this property would be violated, then the move does not go through. In particular, a 2-2 move cannot be done on any edge of the tetrahedron without giving multiple edges between two vertices. In this case, instead of returning the edge label, it returns `[-1, []]`.
    """)
    return


@app.cell
def _(G):
    print(G.pachner22(0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    All graphs with $|V| \ge 6$ have at least one possible 2-2 move on them, so we use the unique $|V| = 6$ graph as an example. Obviously, with only one graph possible, the 2-2 move takes this graph to itself, but the edges are permuted.
    """)
    return


@app.cell
def _(Graph):
    # Define edge list for later use

    six_vert_edge_list = [[0, 1], [0, 4], [0, 5], [1, 2], [3, 4], [1, 3], [4, 5], [2, 5], [2, 3]]

    # Define Graph object, act with 2-2 move

    P = Graph(six_vert_edge_list)
    P.pachner22(0)
    return (six_vert_edge_list,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that `Graph.pachner22()` modifies the graph, so that it is not isomorphic to the original graph. This is easier to see if we compare two $|V| = 8$ graphs. Let `C` be the 4-prism graph (i.e. the projection of the cube onto the plane), and `D` be the only other graph with eight vertices; the latter is known as the standard graph $\Delta_8$.
    """)
    return


@app.cell
def _(Graph):
    # Define edge lists separately for later use

    cube_edge_list = [[0, 1], [0, 6], [0, 7], [1, 2], [4, 6], [6, 7], [3, 4], [2, 5], [5, 7], [1, 3], [2, 3], [4, 5]]
    std_edge_list = [[0, 1], [2, 6], [1, 6], [0, 4], [3, 4], [2, 5], [0, 7], [5, 7], [1, 3], [2, 3], [4, 5], [6, 7]]

    # Define Graph objects, show they are not isomorphic graphs

    C = Graph(cube_edge_list)
    D = Graph(std_edge_list)

    print(f'C = D : {C == D}')
    return D, cube_edge_list, std_edge_list


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The only 2-2 move that takes the graph `D` to `C` is the one on its first listed edge. When this is done, then `D` is now the 4-prism graph, rather than its original form. This allows a check for what the resulting final graph is, from the known list of all graphs with a certain number of vertices.
    """)
    return


@app.cell
def _(D, Graph, cube_edge_list, std_edge_list):
    # Perform 2-2 move on edge 0

    D.pachner22(0)

    # Identify which edge list D now matches with

    print(f'D is standard graph : {D == Graph(std_edge_list)}')
    print(f'D is cubic graph    : {D == Graph(cube_edge_list)}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Currently, `Graph.pachner13()` takes a vertex label, and modifies the graph by changing this vertex into a 3-cycle. However, it does not return any information, such as the mapping of the original edges into the final graph, or the labels of the new vertices and edges.

    We can show that acting on a vertex of the tetrahedron gives the unique six-vertex graph.
    """)
    return


@app.cell
def _(Graph, six_vert_edge_list, tet_edge_list):
    # Perform 1-3 move on vertex 0

    T = Graph(tet_edge_list)
    T.pachner13(0)

    # Identify new form of T

    print(f'T is tetrahedron      : {T == Graph(tet_edge_list)}')
    print(f'T is six-vertex graph : {T == Graph(six_vert_edge_list)}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Similarly, `Graph.pachner31()` takes a list of three edge labels (which must form a 3-cycle), and replaces this cycle with a single vertex. This is the inverse operation of the 1-3 move above.
    """)
    return


@app.cell
def _(Graph, six_vert_edge_list, tet_edge_list):
    # Perform 3-1 move on edges 1, 2, 6

    U = Graph(six_vert_edge_list)
    U.pachner31([1, 2, 6])

    # Identify new form of U

    print(f'U is tetrahedron      : {U == Graph(tet_edge_list)}')
    print(f'U is six-vertex graph : {U == Graph(six_vert_edge_list)}')
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
