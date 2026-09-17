# -*- coding: utf-8 -*-
"""
Created on Sun May 11 09:26:45 2025

@author: cartin
"""

# Go to trivalent folder and run
#
#   conda activate notebooks
#   python -m unittest tests.test_trivalent

import unittest
import numpy as np

from collections import Counter
from hypothesis import given, strategies as st
from trivalent import Graph

#=============================================================================#

@st.composite
def prism_w_edge(draw):
    """
        Strategy to provide n-prism with chosen target edge
    """
    
    # Setting min at 6 avoids situation for 5-prism where taxicab delta can be
    # 4 (b/c face sizes exchange, reducing total number of changes); setting
    # max n at 25 avoids exceeding DEFAULT_MAX_VERT = 50 limit in trivalent
    
    N = draw(st.integers(min_value = 6, max_value = 25))
    G = Graph.create_prism(N)
    tgt_edge = draw(st.integers(min_value = 0, max_value = 3 * N - 1))
    
    return N, G, tgt_edge

#-------------------------------------------------------------------------#

@st.composite
def prism_w_vert(draw):
    """
        Strategy to provide n-prism with chosen target edge
    """
    
    # Setting min at 6 avoids situation for 5-prism where taxicab delta can be
    # 4 (b/c face sizes exchange, reducing total number of changes); setting
    # max n at 24 avoids exceeding DEFAULT_MAX_VERT = 50 limit in trivalent
    
    N = draw(st.integers(min_value = 6, max_value = 24))
    G = Graph.create_prism(N)
    tgt_vert = draw(st.integers(min_value = 0, max_value = 2 * N - 1))
    
    return N, G, tgt_vert

#=============================================================================#

class TestTrivalent(unittest.TestCase):

    #-------------------------------------------------------------------------#
    
    def test_noHyperedges(self):
        """
            Each element of edge list must have only two vertices
        """
        
        self.assertRaises(ValueError, Graph, [[0, 1, 2], [0, 2], [1, 3], [1, 2], [0, 3], [2, 3]])

    #-------------------------------------------------------------------------#
    
    def test_correctEdgeNum(self):
        """
            Edge list must meet requirement 2E = 3V
        """
        
        self.assertRaises(ValueError, Graph, [[0, 1], [0, 2], [1, 3], [1, 2], [0, 3]])
        self.assertRaises(ValueError, Graph, [[0, 1], [0, 4], [0, 5], [1, 2], [3, 4], \
                                              [0, 3], [4, 5], [2, 5], [1, 3], [2, 3]])

    #-------------------------------------------------------------------------#
    
    def test_trivalentGraph(self):
        """
            Every vertex must appear exactly three times in edge list
        """
        
        self.assertRaises(ValueError, Graph, [[0, 1], [0, 2], [1, 3], [1, 2], [0, 3], [0, 3]])
        
    #-------------------------------------------------------------------------#
    
    def test_sameTet(self):
        """
            Two planar tetrahedra with different edge lists
        """
        
        G = Graph([[0, 1], [1, 3], [0, 2], [0, 3], [1, 2], [2, 3]])
        H = Graph([[0, 1], [0, 2], [1, 3], [1, 2], [0, 3], [2, 3]])
        
        self.assertEqual(G, H)

    #-------------------------------------------------------------------------#
    
    def test_mirrorGraphs(self):
        """
            Mirror image graphs with equal number of vertices, edges
        """
        
        G = Graph([[0, 1], [0, 6], [0, 7], [4, 6], [6, 7], [5, 7], [1, 2], [2, 8], \
                    [2, 9], [3, 4], [4, 5], [5, 8], [8, 9], [1, 3], [3, 9]])
        
        H = Graph([[0, 1], [0, 6], [0, 7], [4, 6], [6, 7], [5, 7], [1, 2], [3, 4], \
                    [3, 8], [3, 9], [4, 5], [2, 5], [1, 8], [8, 9], [2, 9]])
        
        self.assertNotEqual(G, H)

    #-------------------------------------------------------------------------#
    
    def test_pachner13Valid(self):
        
        G = Graph([[0, 1], [1, 2], [0, 3], [0, 2], [1, 3], [2, 3]])
        G.pachner13(0)
        
        H = Graph([[0, 1], [1, 2], [0, 3], [3, 4], [3, 5], [0, 2], [1, 4], [4, 5], [2, 5]])
        
        self.assertEqual(G, H)

    #-------------------------------------------------------------------------#
    
    def test_pachner22Triangle(self):
        """
            Choose edge of 3-cycle for 2-2 move
        """
        
        # For this V = 10 graph, the vertices 067, 123, and 489 all form
        # 3-cycles, so any of these edges should give an invalid move
        
        expected = [0, -1, -1, -1, 4, -1, -1, -1, 8, -1, 10, 11, -1, -1, 14]
        
        for tgt in range(15):
            G = Graph([[0, 1], [0, 6], [0, 7], [1, 2], [4, 6], [4, 8], [4, 9], [6, 7], \
                       [3, 8], [8, 9], [2, 5], [5, 7], [1, 3], [2, 3], [5, 9]])
                
            result, perm = G.pachner22(tgt)
            
            self.assertEqual(result, expected[tgt], f"Incorrect validity at edge {tgt}")

    #-------------------------------------------------------------------------#
    
    def test_pachner22Not3Connected(self):
        """
            Choose edge for 2-2 move that would violate 3-connectedness
        """
        
        G = Graph([[0, 1], [0, 6], [0, 7], [4, 6], [6, 7], [5, 7], [1, 2], [2, 8], \
                   [2, 9], [3, 4], [4, 5], [5, 8], [8, 9], [1, 3], [3, 9]])
        result = G.pachner22(10)
        
        self.assertEqual(result[0], -1)
        self.assertListEqual(result[1].tolist(), [])
                    
    #-------------------------------------------------------------------------#
    
    def test_pachner22Valid(self):
        """
            Choose edge for 2-2 move that is valid
        """
        
        G = Graph([[0, 1], [0, 6], [0, 7], [4, 6], [6, 7], [5, 7], [1, 2], [2, 8], \
                   [2, 9], [3, 4], [4, 5], [5, 8], [8, 9], [1, 3], [3, 9]])
        result = G.pachner22(0)
        movePerm = [7, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        
        H = Graph([[4, 6], [6, 9], [0, 5], [0, 8], [0, 9], [7, 8], [8, 9], [2, 7], \
                   [1, 7], [3, 4], [4, 5], [2, 5], [1, 3], [1, 6], [2, 3]])
        
        self.assertEqual(result[0], 0)
        self.assertListEqual(result[1].tolist(), movePerm)
        self.assertEqual(G, H)

    #-------------------------------------------------------------------------#
    
    def test_pachner22ValidFlip(self):
        """
            Choose edge for 2-2 move that is valid, with one edge flipping sign
        """
        
        G = Graph([[0, 1], [0, 6], [0, 7], [4, 6], [6, 7], [5, 7], [1, 2], [2, 8], \
                   [2, 9], [3, 4], [4, 5], [5, 8], [8, 9], [1, 3], [3, 9]])
        result = G.pachner22(14)
        movePerm = [  1,   2,   3,   4,   5,   6,   7,   8,   9,  -10, 12,  13,  14,  15, 11]
        
        H = Graph([[0, 1], [0, 8], [0, 9], [6, 8], [8, 9], [7, 9], [4, 6], [6, 7], \
                   [5, 7], [1, 2], [3, 4], [4, 5], [2, 5], [1, 3], [2, 3]])
        
        self.assertEqual(result[0], 14)
        self.assertListEqual(result[1].tolist(), movePerm)
        self.assertEqual(G, H)

    #-------------------------------------------------------------------------#
    
    @given(prism_w_edge())
    def test_pachner22Involution(self, data_tuple):
        """
            Applying the 2-2 move twice restores the original graph
        """
        
        n, G, tgtIdx = data_tuple
        
        # Apply 2-2 move on target edge

        newEdge, movePerm = G.pachner22(tgtIdx)
        
        # Find index of new edge using 1-based signed permutation
        
        newTgtIdx = abs(movePerm[tgtIdx]) - 1
        
        # Apply 2-2 move to created edge
        
        G.pachner22(newTgtIdx)
        
        # Compare to another n-prism graph
        
        H = Graph.create_prism(n)
        self.assertEqual(G, H)

    #-------------------------------------------------------------------------#
    
    @given(prism_w_edge())
    def test_pachner22Locality(self, data_tuple):
        """
            The 2-2 move only affects adjacency near target edge
        """
        
        n, G, tgtIdx = data_tuple
        
        # Find labels of vertices adjacent to all vertices
        
        adjBeforeList = {vvv : set() for vvv in range(2 * n)}
        for u, v in G.edge_list:
            adjBeforeList[u].add(v)
            adjBeforeList[v].add(u)
            
        # Apply 2-2 move to target edge
        
        G.pachner22(tgtIdx)
        
        # Find adjacency list after transformation
        
        adjAfterList = {vvv : set() for vvv in range(2 * n)}
        for u, v in G.edge_list:
            adjAfterList[u].add(v)
            adjAfterList[v].add(u)
        
        # Find all vertices where a change occurred in adjacency; at most, this
        # should be four vertices -- two on the edge used, and two of their
        # neighbors
        
        change = {vertex for vertex in adjBeforeList if adjBeforeList[vertex] != adjAfterList[vertex]}
        self.assertLessEqual(len(change), 4, f"Locality violation; {len(change)} vertices affected")

    #-------------------------------------------------------------------------#
    
    @given(prism_w_edge())
    def test_pachner22FaceConservation(self, data_tuple):
        """
            The total number of boundary edges for faces is conserved, and only
            the face involved should change size.
        """
        
        n, G, tgtIdx = data_tuple
        
        # Find number of boundary edges for all faces
        
        oldFaceSizeList = G.face_size_list.copy()
        oldFaceSizeDict = Counter(oldFaceSizeList)
        
        # Apply 2-2 move to target edge
        
        G.pachner22(tgtIdx)
        newFaceSizeList = G.face_size_list.copy()
        newFaceSizeDict = Counter(newFaceSizeList)
        
        # Find new face size list, and see if (1) number of boundary edges is
        # still 2|E|, and (2) the affected faces are altered in the pattern
        # (+1, -1, -1, +1)
        
        self.assertEqual(sum(oldFaceSizeList), sum(G.face_size_list), 
                         "Sums of face size list are not equal")
        
        self.assertEqual(sum(G.face_size_list), 6 * n,
                         "Sum of face size list not equal to 2|E|")
        
        # The expected pattern in the face size changes should give a total
        # taxicab distance of 8 between the old and new face size vectors
        
        allFaceSizes = set(oldFaceSizeDict.keys()).union(set(newFaceSizeDict.keys()))
        change = sum(abs(oldFaceSizeDict[iii] - newFaceSizeDict[iii]) for iii in allFaceSizes)
        
        self.assertEqual(change, 8, "Face size taxicab delta ({change}) is incorrect, expected 8")

    #-------------------------------------------------------------------------#
    
    def test_pachner31Contractible(self):
        """
            Choose triangle whose external edges are in correct order in edge
            list when internal edges removed
        """
        
        G = Graph([[0, 1], [0, 4], [0, 5], [1, 2], [3, 4], [1, 3], [4, 5], [2, 5], [2, 3]])
        perm = G.pachner31([1, 2, 6])
        
        self.assertListEqual(list(perm), [1, 4, -5, 6, -8, 9])
        
        H = Graph([[0, 1], [0, 2], [1, 3], [0, 3], [1, 2], [2, 3]])
        
        self.assertEqual(G, H)

    #-------------------------------------------------------------------------#
    
    def test_pachner31NonContractible(self):
        """
            Choose triangle whose external edges are not in correct order in
            edge list when internal edges removed
        """
        
        G = Graph([[0, 1], [0, 6], [0, 7], [4, 6], [4, 8], [4, 9], [6, 7], [5, 7], \
                   [1, 2], [3, 8], [8, 9], [5, 9], [2, 5], [1, 3], [2, 3]])
        perm = G.pachner31([8, 13, 14])
        
        self.assertListEqual(list(perm), [1, 13, 8, 2, 4, 5, 10, 12, 6, 3, 7, 11])
        
        H = Graph([[0, 1], [0, 6], [0, 7], [4, 6], [6, 7], [5, 7], \
                   [1, 2], [3, 4], [4, 5], [2, 5], [1, 3], [2, 3]])
        
        self.assertEqual(G, H)

    # #-------------------------------------------------------------------------#
    
    def test_pachner31EdgeIdxFlip(self):
        """
            Choose triangle with vertex labels flipped in edge by after-move mapping
        """
        
        G = Graph([[0, 1], [0, 6], [0, 7], [4, 6], [4, 8], [4, 9], [6, 7], [5, 7], \
                   [1, 2], [3, 8], [8, 9], [5, 9], [2, 5], [1, 3], [2, 3]])
        perm = G.pachner31([1, 2, 6])
        
        self.assertListEqual(list(perm), [1, 9, 13, -4, 5, 10, 14, 15, -8, 6, 11, 12])
        
        H = Graph([[0, 1], [0, 6], [0, 7], [4, 6], [6, 7], [5, 7], \
                   [1, 2], [3, 4], [4, 5], [2, 5], [1, 3], [2, 3]])
        
        self.assertEqual(G, H)

    # #-------------------------------------------------------------------------#
    
    def test_pachner31CycleOrder(self):
        """
            Use same triangle with different edge orderings as arguments
        """
        
        G = Graph([[0, 1], [0, 6], [0, 7], [4, 6], [4, 8], [4, 9], [6, 7], [5, 7], \
                   [1, 2], [3, 8], [8, 9], [5, 9], [2, 5], [1, 3], [2, 3]])
        perm = G.pachner31([1, 2, 6])
        
        self.assertListEqual(list(perm), [1, 9, 13, -4, 5, 10, 14, 15, -8, 6, 11, 12])
        
        H = Graph([[0, 1], [0, 6], [0, 7], [4, 6], [4, 8], [4, 9], [6, 7], [5, 7], \
                   [1, 2], [3, 8], [8, 9], [5, 9], [2, 5], [1, 3], [2, 3]])
        H.pachner31([2, 1, 6])
        
        self.assertEqual(G, H)

    #-------------------------------------------------------------------------#
    
    def test_pachner31NotTriangle(self):
        """
            List of elements does not form a 3-cycle in graph
        """
        
        G = Graph([[0, 1], [0, 6], [0, 7], [4, 6], [4, 8], [4, 9], [6, 7], [5, 7], \
                   [1, 2], [3, 8], [8, 9], [5, 9], [2, 5], [1, 3], [2, 3]])
        
        self.assertRaises(ValueError, G.pachner31, [0, 1, 2])

    #-------------------------------------------------------------------------#
    
    @given(prism_w_vert())
    def test_pachner31Invariant(self, data_tuple):
    
        
        G = Graph([[0, 1], [0, 6], [0, 7], [4, 6], [4, 8], [4, 9], [6, 7], [5, 7], \
                   [1, 2], [3, 8], [8, 9], [5, 9], [2, 5], [1, 3], [2, 3]])
        n, G, tgtIdx = data_tuple
        
        # Use 1-3 move on random vertex of n-prism
        
        G.pachner13(tgtIdx)
        
        # Check values of |V|, |E|, |F| after move
        
        self.assertEqual(G.num_vert, 2 * n + 2)
        self.assertEqual(G.num_edges, 3 * n + 3)
        self.assertEqual(G.num_faces, n + 3)
        
        # Check all vertices are 3-regular
        
        _, counts = np.unique(G.edge_list, return_counts = True)
        self.assertTrue(np.all(counts == 3))

    #-------------------------------------------------------------------------#
    
    @given(prism_w_vert())
    def test_pachner13_31_Involution(self, data_tuple):
    
        n, G, tgtIdx = data_tuple
        
        # Perform 1-3 move, then 3-1 move on resulting 3-cycle
        
        G.pachner31(G.pachner13(tgtIdx))
        
        # Check values of |V|, |E|, |F| after move, isomorphism with n-prism
        
        self.assertEqual(G.num_vert, 2 * n)
        self.assertEqual(G.num_edges, 3 * n)
        self.assertEqual(G.num_faces, n + 2)
        
        self.assertTrue(G == Graph.create_prism(n))

    #-------------------------------------------------------------------------#
    
#=============================================================================#

if __name__=='__main__':
    unittest.main()