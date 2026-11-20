import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data file format

    For each set of 3-regular graphs $G_i$ with a given vertex number $|V|$ and dual to the 2-sphere $S^2$, the descriptions of these graphs are stored in `.npz` files for later use. This document explains how the files `V=vvv-Pachner-graph-base.npz` are formatted, where `vvv` is an even number giving $|V|$. The files contain the following:

    * `edge_lists`
    * `group_elements`, `group_ptr`
    * `matrix_col`, `matrix_data`, `matrix_indptr`
    * `move_edges`, `move_perms`

    The details of these various entries are provided below.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Graph information

    The first few entries give the data necessary to reconstruct the graphs themselves, as well as their symmetry groups.

    `edge_lists`: An `np.array` of size (`N_graph`, `N_edge`, 2), where `N_graph` is the number of non-isomorphic graphs with $|V|$ vertices, `N_edges` = $3|V| / 2$ is the number of edges for all these graphs, and each edge is of the form (`start`, `end`) for the two incident vertex labels, with `start` < `end`. Thus, `edge_lists[iii]` is the list of edges for graph `iii` in the canonical list. The entries are in `np.uint8` format, which is sufficient for $|V| <= 255$. The edges are given in implicit order, meaning the edges appear in the same order as they do in the cyclic order around each vertex, in CCW arrangement.

    `group_elements`, `group_ptr`: These two arrays provide all information to reconstruct the automorphism group of the graph structures. The group elements are given as a list of permutations of the edges; thus, each element is an `np.array` of shape (`N_edges`) of *signed* edge indices. A negative edge label records a flip in the orientation of the edge under the group element. Recall that the number of edges $|E|$ (or `N_edges`) in a graph is $|E| = 3|V| / 2$. In general, the groups will be of arbitrary size, so this information is recorded in `group_ptr`, which is an `np.array` recording where each group starts and stops in group_elements. Specifically, if we want the symmetry group for graph `iii`, then they are the elements in the list `group_elements[group_ptr[iii], group_ptr[iii + 1]]`. This means that `group_ptr` is an array of length (`N_graph` + 1) with `np.uint16` entries, and `group_elements` is an `np.array` of shape (`N_group`, `N_edge`), with `np.int8` elements. Since the permutation elements are *signed*, then they run from $-|E|$ to $+|E|$ (without 0), but `np.uint8` is still sufficient for $|V| <= 127$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Pachner graph

    Given the set of $G_i$, the Pachner graph ${\cal P}$ is the graph obtained by using the 2-2 moves between the various $G_i$. The nodes of the Pachner graph are the graphs $G_i$, and the edges of ${\cal P}$ are the valid 2-2 moves between the $G_i$. The remaining entries in the file pertain to the graph ${\cal P}$.

    `matrix_col`, `matrix_data`, `matrix_indptr`: This is the information needed to construct the CSR adjacency matrix of shape (`numGraph`, `numGraph`) for the Pachner graph. This is stored in this manner so that the individual 2-2 moves are synchronized with the edges and permutations associated with the move, since some of the moves are multiple edges in the Pachner graph. Since `matrix_col` is giving the target graph labels, it is given as `np.uint16`, to accomodate the 2274 graphs of $|V| = 18$; `matrix_indptr` gives the number of valid moves, so is set at `np.int32` for the |V| = 18 case of XXX graph; `matrix_data` is only a series of 1's, and is chosen as `np.uint8`.

    `move_edges`: This records the edge label for the source edge of the 2-2 move as described in the *source* graph. This is an `np.array` of size `(N_moves`), with `np.uint8` entries, since these edge labels are unsigned.The location of the moves for a given graph are given by `indptr`.

    `move_perms`: For every 2-2 move $G \to G'$, the permutation between the source and target graphs under the isomorphism is computed. This is a *signed* permutation of the edge labels, so the entries run from $-|E|$ to $+|E|$. This `np.array` has shape (`N_moves`, `N_edges`), with entries `np.int8`, since the signs are needed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example data file

    Here we use the data for the $|V| = 12$ Pachner graph to demonstrate the specifics of this information.
    """)
    return


@app.cell
def _(np):
    numVert = 12
    pachnerData = np.load(f'../data/V={numVert}-Pachner-graph-base.npz')

    # Read out graph data into each entry

    edgeLists = pachnerData['edge_lists']
    groupElements, groupPtr = pachnerData['group_elements'], pachnerData['group_ptr']
    numGraph = len(edgeLists)

    col, data, indptr = pachnerData['matrix_col'], pachnerData['matrix_data'], pachnerData['matrix_indptr']
    moveEdges = pachnerData['move_edges']
    movePerms = pachnerData['move_perms']

    print(f'Read in {numGraph} graphs with V = {numVert} vertices')
    return groupPtr, indptr, numGraph


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One thing we can investigate is the sizes of the various symmetry groups. We print these out now.
    """)
    return


@app.cell
def _(groupPtr, numGraph):
    for graph_id in range(numGraph):
        print(f'graph {graph_id:2d} : num of symmetries = {groupPtr[graph_id + 1] - groupPtr[graph_id]}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice that graphs 3 and 14 have a high degree of symmetry. This means that most of their edges are isomorphic under the automorphism group. Thus, these edges are treated as equivalent when computing the Pachner graph. In other words, if two edges are isomorphic, then a valid 2-2 move on one of these edges is *not* counted separately from a 2-2 move on the other. Thus, we expect that the graphs $G_3$ and $G_{14}$ have a low number of edges in the Pachner graph (i.e. a low number of 2-2 moves from them to other graphs).

    Conversely, there are other graphs with trivial automorphism groups (size 1). In these graphs, none of the edges are isomorphic, so a 2-2 move on any edge is considered distinct. If these moves do not eliminate the 3-connectedness of the resulting graph, they will appear as valid links in the Pachner graph.

    We can test this intuition by comparing the size of the graph symmetry groups to the number of links for each graph $G_i$ in the Pachner graph ${\cal P}$. As we expect the graph of link number vs. group size follows a descending curve.
    """)
    return


@app.cell
def _(groupPtr, indptr, numGraph, plt):
    numSym = [groupPtr[iii + 1] - groupPtr[iii] for iii in range(numGraph)]
    numLinks = [indptr[iii + 1] - indptr[iii] for iii in range(numGraph)]

    plt.xlabel('num of symmetries')
    plt.ylabel('num of Pachner graph edges')

    plt.scatter(numSym, numLinks)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By testing different values of $|V|$, one sees that the number of symmetries tends to "pile up" at the lower end, as an increasing number of graphs have trivial automorphism groups for larger $|V|$. However, there is still a spread in the number of Pachner graph links, so the data points accumulate more on the left-hand side of the graph.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


if __name__ == "__main__":
    app.run()
