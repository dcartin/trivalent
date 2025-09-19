# -*- coding: utf-8 -*-
"""
Created on Sun May 11 09:26:45 2025

@author: cartin
    
"""

# Go to trivalent folder and run "python -m unittest tests.test_trivalent"

import unittest
from trivalent import Vertex, Edge, Graph

class TestTrivalent(unittest.TestCase):

    #-------------------------------------------------------------------------#
    
    def test_numberOfVertices(self):
        """
            Number of vertices must be a positive even integer
        """
        
        self.assertRaises(ValueError, Graph, -1)
        self.assertRaises(ValueError, Graph, 1.5)
        self.assertRaises(ValueError, Graph, '2')
        self.assertRaises(ValueError, Graph, 2j)
        self.assertRaises(ValueError, Graph, [2])
        
        self.assertRaises(ValueError, Graph, 5)
        self.assertRaises(ValueError, Graph, 2.0)

    #-------------------------------------------------------------------------#

    def test_addEdges(self):
        """
            graph.addEdges input must be list of edges in form [start, end, (color), (twist)]
        """
        
        graph = Graph(4)
        
        # added_edge_list must be a list
        
        self.assertRaises(ValueError, graph.addEdges, 0)
        self.assertRaises(ValueError, graph.addEdges, '[0, 1]')
        
        # Each entry in added_edge_list must be a list of 2-4 integers
        
        self.assertRaises(ValueError, graph.addEdges, [0, 2])
        
        self.assertRaises(ValueError, graph.addEdges, [['0', 2]])
        self.assertRaises(ValueError, graph.addEdges, [[0, '2']])
        
        self.assertRaises(ValueError, graph.addEdges, [[0.0, 2]])
        self.assertRaises(ValueError, graph.addEdges, [[0.5, 2]])
        self.assertRaises(ValueError, graph.addEdges, [[0, 1.5]])
        self.assertRaises(ValueError, graph.addEdges, [[0, 2.0]])
        
        self.assertRaises(ValueError, graph.addEdges, [[0]])
        self.assertRaises(ValueError, graph.addEdges, [[0, 1, 2, 3, 4]])
        
        # Each label start, end must be within current number of vertices
        
        self.assertRaises(ValueError, graph.addEdges, [[-1, 3]])
        self.assertRaises(ValueError, graph.addEdges, [[3, -1]])
        self.assertRaises(ValueError, graph.addEdges, [[0, 4]])
        self.assertRaises(ValueError, graph.addEdges, [[4, 0]])

    #-------------------------------------------------------------------------#
    
    def test_addVertices(self):
        """
            Number of added vertices must be a positive integer
        """
        
        graph = Graph(4)
        
        self.assertRaises(ValueError, graph.addVertices, -1)
        self.assertRaises(ValueError, graph.addVertices, 0)
        self.assertRaises(ValueError, graph.addVertices, '2')
        self.assertRaises(ValueError, graph.addVertices, [2])
        self.assertRaises(ValueError, graph.addVertices, 1.0)

    #-------------------------------------------------------------------------#
    
    def test_trivalentVertex(self):
        """
            A maximum of three edges incident to a vertex
        """
        
        graph = Graph(6)
        
        self.assertRaises(ValueError, graph.addEdges, [[0, 1], [0, 2], [0, 3], [0, 3]])
        self.assertRaises(ValueError, graph.addEdges, [[0, 1], [0, 2], [0, 3], [0, 4]])

    #-------------------------------------------------------------------------#
    
    def test_unequalVertexNumber(self):
        """
            Equal graphs must have same number of vertices
        """
        
        # Graphs have same number of vertices
        
        self.assertNotEqual(Graph(4), Graph(6))

    #-------------------------------------------------------------------------#
    
    def test_unequalEdgeNumber(self):
        """
            Equal graphs must have same number of edges
        """
        
        G = Graph(4)
        H = Graph(4)
        H.addEdges([[0, 1]])
        
        self.assertNotEqual(G, H)

    #-------------------------------------------------------------------------#
    
    def test_unequalDegreeList(self):
        """
            Equal graphs must have same degree lists
        """
        
        G = Graph(4)
        G.addEdges([[0, 1], [0, 3], [1, 2], [2, 3]])
        
        H = Graph(4)
        H.addEdges([[0, 1], [0, 2], [0, 3], [1, 2]])
        
        self.assertNotEqual(G, H)

    #-------------------------------------------------------------------------#
    
    def test_sameEdgeDescription(self):
        """
            Graphs are created by equal edge lists
        """
        
        G = Graph(4)
        G.addEdges([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
        
        H = Graph(4)
        H.addEdges([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
        
        self.assertEqual(G, H)

    #-------------------------------------------------------------------------#
    
    def test_diffEdgeDescription(self):
        """
            Equal graphs with non-trivial vertex isomorphism
        """
        
        G = Graph(4)
        G.addEdges([[0, 1], [0, 1], [0, 2], [1, 3], [2, 3], [2, 3]])
        
        H = Graph(4)
        H.addEdges([[0, 1], [0, 2], [0, 2], [1, 3], [1, 3], [2, 3]])
        
        self.assertEqual(G, H)

    #-------------------------------------------------------------------------#
    
    def test_setColorNoOrientFlip(self):
        """
            Change edge color with no change in orientation
        """
        
        G = Graph(4)
        G.addEdges([[0, 1, 1], [1, 3, 1], [0, 2, -1], [0, 3, -1], [1, 2, -1], [2, 3, -1]])
        
        self.assertEqual(G.edge(0).setColor(3), True)

    #-------------------------------------------------------------------------#
    
    def test_setColorOrientFlip(self):
        """
            Change edge color with change in orientation
        """
        
        G = Graph(4)
        G.addEdges([[0, 1, 1], [1, 3, 1], [0, 2, -1], [0, 3, -1], [1, 2, -1], [2, 3, -1]])
        
        self.assertEqual(G.edge(0).setColor(-3), False)

    #-------------------------------------------------------------------------#
    
    def test_connectEdgeNotIncident(self):
        """
            Connecting edge does not have vertex as start or end
        """
        
        G = Graph(4)
        G.addEdges([[0, 1, 1], [1, 3, 1], [0, 2, -1], [0, 3, -1], [1, 2, -1]])
                    
        e = Edge(start = G.vertex(2), end = G.vertex(3), color = -1)
        
        self.assertRaises(ValueError, G.vertex(1).connectEdge, e)

    #-------------------------------------------------------------------------#
    
    def test_connectEdgeWithSink(self):
        """
            Connecting edge with given orientation creates a sink at vertex
        """
        
        G = Graph(4)
        G.addEdges([[1, 3, 1], [0, 2, -1], [0, 3, -1], [1, 2, -1], [2, 3, -1]])
                    
        e = Edge(start = G.vertex(0), end = G.vertex(1), color = -1)
        
        with self.assertRaises(AttributeError):
            G.vertex(0).connectEdge(e)

    #-------------------------------------------------------------------------#
    
    def test_connectEdgeWithSource(self):
        """
            Connecting edge with given orientation creates a source at vertex
        """
        
        G = Graph(4)
        G.addEdges([[0, 1, 1], [1, 3, 1], [0, 2, -1], [0, 3, -1], [1, 2, -1]])
                    
        e = Edge(start = G.vertex(2), end = G.vertex(3), color = 1)
        
        with self.assertRaises(AttributeError):
            G.vertex(2).connectEdge(e)

    #-------------------------------------------------------------------------#
    
    def test_replaceEdgeNotIncident(self):
        """
            Replacement edge does not have vertex as start or end
        """
        
        G = Graph(4)
        G.addEdges([[0, 1, 1], [1, 3, 1], [0, 2, -1], [0, 3, -1], [1, 2, -1], [2, 3, -1]])
                    
        e = Edge(start = G.vertex(2), end = G.vertex(3), color = -1)
        
        self.assertRaises(ValueError, G.vertex(1).replaceEdge, G.edge(5), e)

    #-------------------------------------------------------------------------#
    
    def test_replaceEdgeWithSink(self):
        """
            Replacement edge with given orientation creates a sink at vertex
        """
        
        G = Graph(4)
        G.addEdges([[0, 1, 1], [1, 3, 1], [0, 2, -1], [0, 3, -1], [1, 2, -1], [2, 3, -1]])
                    
        oldEdge = G.edge(0)
        newEdge = Edge(start = G.vertex(0), end = G.vertex(1), color = -1)
        
        with self.assertRaises(AttributeError):
            G.vertex(0).replaceEdge(remove_edge = oldEdge, added_edge = newEdge)

    #-------------------------------------------------------------------------#
    
    def test_replaceEdgeWithSource(self):
        """
            Replacement edge with given orientation creates a source at vertex
        """
        
        G = Graph(4)
        G.addEdges([[0, 1, 1], [1, 3, 1], [0, 2, -1], [0, 3, -1], [1, 2, -1], [2, 3, -1]])
               
        oldEdge = G.edge(5)
        newEdge = Edge(start = G.vertex(2), end = G.vertex(3), color = 1)
        
        with self.assertRaises(AttributeError):
            G.vertex(2).replaceEdge(remove_edge = oldEdge, added_edge = newEdge)

    #-------------------------------------------------------------------------#
    
    def test_nonIsomorphicGraphs(self):
        """
            Non-isomorphic graphs with equal number of vertices, edges
        """
        
        G = Graph(4)
        G.addEdges([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
        
        H = Graph(4)
        H.addEdges([[0, 1], [0, 1], [0, 2], [1, 3], [2, 3], [2, 3]])
        
        self.assertNotEqual(G, H)

    #-------------------------------------------------------------------------#
    
    def test_mirrorGraphs(self):
        """
            Mirror image graphs with equal number of vertices, edges
        """
        
        G = Graph(10)
        G.addEdges([[0, 1], [0, 6], [0, 7], [4, 6], [6, 7], [5, 7], [1, 2], [2, 8], \
                    [2, 9], [3, 4], [4, 5], [5, 8], [8, 9], [1, 3], [3, 9]])
        
        H = Graph(10)
        H.addEdges([[0, 1], [0, 6], [0, 7], [4, 6], [6, 7], [5, 7], [1, 2], [3, 4], \
                    [3, 8], [3, 9], [4, 5], [2, 5], [1, 8], [8, 9], [2, 9]])
        
        self.assertNotEqual(G, H)

    #-------------------------------------------------------------------------#
    
    def test_oneMove(self):
        """
            Add self-loop to left side of edge, as moving from start to end
        """
        
        G = Graph(4)
        G.addEdges([[0, 1], [1, 2], [0, 3], [0, 2], [1, 3], [2, 3]])
        G.oneMove(0)
        
        H = Graph(6)
        H.addEdges([[0, 4], [1, 4], [1, 2], [0, 3], [0, 2], [1, 3], [2, 3], [4, 5], [5, 5]])
        
        self.assertEqual(G, H)

    #-------------------------------------------------------------------------#
    
    def test_twoMove(self):
        """
            Add double edge in middle of edge
        """
        
        G = Graph(4)
        G.addEdges([[0, 1], [1, 2], [0, 3], [0, 2], [1, 3], [2, 3]])
        G.twoMove(0)
        
        H = Graph(6)
        H.addEdges([[0, 4], [4, 5], [1, 5], [4, 5], [1, 2], [0, 3], [0, 2], [1, 3], [2, 3]])
        
        self.assertEqual(G, H)

    #-------------------------------------------------------------------------#
    
    def test_threeMoveTwoDistinctAdjVert(self):
        """
            Expand single vertex into 3-cycle, two adjacent vertices distinct
        """
        
        G = Graph(4)
        G.addEdges([[0, 2], [2, 3], [0, 2], [0, 3], [1, 1], [1, 3]])
        G.threeMove(0)
        
        H = Graph(6)
        H.addEdges([[0, 2], [0, 4], [0, 5], [2, 3], [2, 4], [4, 5], [3, 5], [1, 1], [1, 3]])
        
        self.assertEqual(G, H)

    #-------------------------------------------------------------------------#
    
    def test_threeMoveAllDistinctAdjVert(self):
        """
            Expand single vertex into 3-cycle, all adjacent vertices distinct
        """
        
        G = Graph(4)
        G.addEdges([[0, 1], [1, 2], [0, 3], [0, 2], [1, 3], [2, 3]])
        G.threeMove(0)
        
        H = Graph(6)
        H.addEdges([[0, 1], [1, 2], [0, 3], [3, 4], [3, 5], [0, 2], [1, 4], [4, 5], [2, 5]])
        
        self.assertEqual(G, H)

    #-------------------------------------------------------------------------#
    
    def test_fourMoveOneDistinctAdjVert(self):
        """
            Expand single edge (double edge + single neighbor) into 4-cycle
        """
        
        # Single neighbor "to left" of edge xy
        
        G = Graph(4)
        G.addEdges([[0, 0], [0, 1], [1, 3], [2, 3], [1, 2], [2, 3]])
        G.fourMove(5)
        
        H = Graph(6)
        H.addEdges([[0, 0], [0, 1], [1, 5], [1, 2], [3, 4], [3, 5], [3, 4], [2, 5], [2, 4]])
        
        self.assertEqual(G, H)
        
        # Single neighbor "to right" of edge xy
        
        G = Graph(4)
        G.addEdges([[0, 0], [0, 1], [1, 2], [2, 3], [1, 3], [2, 3]])
        G.fourMove(5)
        
        H = Graph(6)
        H.addEdges([[0, 0], [0, 1], [1, 4], [2, 4], [1, 3], [3, 4], [2, 5], [3, 5], [2, 5]])
        
        self.assertEqual(G, H)

    #-------------------------------------------------------------------------#
    
    def test_fourMoveAdjSelfloop(self):
        """
            Expand single edge (one vertex has self-loop) into 4-cycle
        """
        
        # Self-loop on x (lower label vertex)
        
        G = Graph(4)
        G.addEdges([[0, 0], [0, 1], [1, 3], [2, 3], [1, 2], [2, 3]])
        G.fourMove(1)
        
        H = Graph(6)
        H.addEdges([[0, 4], [2, 3], [3, 5], [1, 5], [2, 3], [1, 2], [1, 4], [0, 4], [0, 5]])
        
        self.assertEqual(G, H)
        
        # Self-loop on y (upper label vertex; switched 0 <-> 1 in starting edge
        # list, which results in permutation 0 <-> 1, 4 <-> in final list)
        
        G = Graph(4)
        G.addEdges([[1, 1], [0, 1], [0, 3], [2, 3], [0, 2], [2, 3]])
        G.fourMove(1)
        
        H = Graph(6)
        H.addEdges([[1, 5], [2, 3], [3, 4], [0, 4], [2, 3], [0, 2], [0, 5], [1, 5], [1, 4]])
        
        self.assertEqual(G, H)

    #-------------------------------------------------------------------------#
    
    def test_fourMoveThreeDistinctAdjVert(self):
        """
            Expand single edge (part of double edge) into 4-cycle
        """
        
        # Single neighbor "to left" of edge xy
        
        G = Graph(6)
        G.addEdges([[0, 1], [0, 3], [0, 2], [2, 4], [1, 2], [3, 4], [1, 4], [3, 5], [5, 5]])
        G.fourMove(0)
        
        H = Graph(8)
        H.addEdges([[0, 2], [1, 4], [2, 4], [3, 4], [3, 5], [0, 7], [0, 6], \
                    [1, 6], [2, 7], [1, 7], [3, 6], [5, 5]])
        
        self.assertEqual(G, H)
        
        # Single neighbor "to right" of edge xy
        
        G = Graph(6)
        G.addEdges([[0, 1], [1, 4], [3, 4], [1, 2], [2, 4], [0, 2], [0, 3], [3, 5], [5, 5]])
        G.fourMove(0)
        
        H = Graph(8)
        H.addEdges([[0, 3], [1, 2], [2, 4], [3, 5], [0, 7], [0, 6], [1, 6], \
                   [2, 6], [4, 7], [1, 7], [3, 4], [5, 5]])
        
        self.assertEqual(G, H)

    #-------------------------------------------------------------------------#
    
    def test_fourMoveAllDistinctAdjVert(self):
        """
            Expand single edge (all neighbors distinct) into 4-cycle
        """
        
        G = Graph(6)
        G.addEdges([[0, 1], [1, 2], [2, 5], [1, 3], [2, 3], [0, 4], [3, 4], [0, 5], [4, 5]])
        G.fourMove(0)
        
        H = Graph(8)
        H.addEdges([[2, 7], [1, 7], [2, 5], [1, 3], [1, 6], [2, 3], [4, 6], \
                    [0, 6], [3, 4], [0, 5], [0, 7], [4, 5]])
        
        self.assertEqual(G, H)

    #-------------------------------------------------------------------------#
    
    def test_sameTetOrient(self):
        """
            Two planar tetraheadra with same edge orientations
        """
        
        G = Graph(4)
        G.addEdges([[0, 1, 1], [1, 3, 1], [0, 2, -1], [0, 3, -1], [1, 2, -1], [2, 3, -1]])
        
        H = Graph(4)
        H.addEdges([[0, 1, 1], [0, 2, -1], [1, 3, 1], [1, 2, -1], [0, 3, -1], [2, 3, -1]])
        
        self.assertEqual(G, H)

    #-------------------------------------------------------------------------#
    
    def test_oppTetOrient(self):
        """
            Two planar tetrahedra with mirror edge orientations
        """
        
        G = Graph(4)
        G.addEdges([[0, 1, 1], [1, 3, 1], [0, 2, -1], [0, 3, -1], [1, 2, -1], [2, 3, -1]])
        
        H = Graph(4)
        H.addEdges([[0, 1, 1], [0, 2, -1], [1, 3, -1], [1, 2, 1], [0, 3, -1], [2, 3, 1]])
        
        self.assertNotEqual(G, H)
        
#=============================================================================#

if __name__=='__main__':
    unittest.main()
