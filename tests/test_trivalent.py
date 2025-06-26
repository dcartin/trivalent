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
    
    # def setUp(self):
    #     self.graph = Graph()

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
            graph.addEdges input must be list of edges in form [start, end]
        """
        
        graph = Graph(4)
        
        # added_edge_list must be a list
        
        self.assertRaises(ValueError, graph.addEdges, 0)
        self.assertRaises(ValueError, graph.addEdges, '[0, 1]')
        
        # Each entry in added_edge_list must be a list of two integers
        
        self.assertRaises(ValueError, graph.addEdges, [0, 2])
        
        self.assertRaises(ValueError, graph.addEdges, [['0', 2]])
        self.assertRaises(ValueError, graph.addEdges, [[0, '2']])
        
        self.assertRaises(ValueError, graph.addEdges, [[0.0, 2]])
        self.assertRaises(ValueError, graph.addEdges, [[0.5, 2]])
        self.assertRaises(ValueError, graph.addEdges, [[0, 1.5]])
        self.assertRaises(ValueError, graph.addEdges, [[0, 2.0]])
        
        self.assertRaises(ValueError, graph.addEdges, [[0]])
        self.assertRaises(ValueError, graph.addEdges, [[0, 2, 3]])
        
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
    
    def test_threeMove(self):
        """
            Expand single vertex into 3-cycle
        """
        
        G = Graph(4)
        G.addEdges([[0, 1], [1, 2], [0, 3], [0, 2], [1, 3], [2, 3]])
        G.threeMove(0)
        
        H = Graph(6)
        H.addEdges([[0, 1], [1, 2], [0, 3], [3, 4], [3, 5], [0, 2], [1, 4], [4, 5], [2, 5]])
        
        self.assertEqual(G, H)

    #-------------------------------------------------------------------------#
    
    def test_fourMove(self):
        """
            Expand single edge into 4-cycle
        """
        
        G = Graph(4)
        G.addEdges([[0, 1], [1, 2], [0, 3], [0, 2], [1, 3], [2, 3]])
        G.fourMove(0)
        
        H = Graph(6)
        H.addEdges([[0, 1], [2, 4], [1, 4], [0, 3], [0, 5], [2, 5], [1, 3], [2, 3], [4, 5]])
        
        self.assertEqual(G, H)
        
#=============================================================================#

if __name__=='__main__':
    unittest.main()
