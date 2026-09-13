# -*- coding: utf-8 -*-
"""
Created on Sun May 11 09:26:45 2025

@author: cartin
"""

# Go to trivalent folder and run "python -m unittest tests.test_trivalent"

import unittest
import numpy as np

from trivalent import Graph

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
        
        G = Graph([[4, 6], [6, 9], [0, 5], [0, 8], [0, 9], [7, 8], [8, 9], [2, 7], \
                   [1, 7], [3, 4], [4, 5], [2, 5], [1, 3], [1, 6], [2, 3]])
        result = G.pachner22(4)
        
        self.assertEqual(result[0], -1)
        self.assertListEqual(result[1].tolist(), [])

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
    
    def test_pachner31Contractible(self):
        """
            Choose triangle whose external edges are in correct order in edge
            list when internal edges removed
        """
        
        G = Graph([[0, 1], [0, 4], [0, 5], [1, 2], [3, 4], [1, 3], [4, 5], [2, 5], [2, 3]])
        G.pachner31([1, 2, 6])
        
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
        G.pachner31([8, 13, 14])
        
        H = Graph([[0, 1], [0, 6], [0, 7], [4, 6], [6, 7], [5, 7], \
                   [1, 2], [3, 4], [4, 5], [2, 5], [1, 3], [2, 3]])
        
        self.assertEqual(G, H)

    #-------------------------------------------------------------------------#
    
    def test_pachner31EdgeIdxFlip(self):
        """
            Choose triangle with vertex labels flipped in edge by after-move mapping
        """
        
        G = Graph([[0, 1], [0, 6], [0, 7], [4, 6], [4, 8], [4, 9], [6, 7], [5, 7], \
                   [1, 2], [3, 8], [8, 9], [5, 9], [2, 5], [1, 3], [2, 3]])
        G.pachner31([1, 2, 6])
        
        H = Graph([[0, 1], [0, 6], [0, 7], [4, 6], [6, 7], [5, 7], \
                   [1, 2], [3, 4], [4, 5], [2, 5], [1, 3], [2, 3]])
        
        self.assertEqual(G, H)

    #-------------------------------------------------------------------------#
    
    def test_pachner31CycleOrder(self):
        """
            Use same triangle with different edge orderings as arguments
        """
        
        G = Graph([[0, 1], [0, 6], [0, 7], [4, 6], [4, 8], [4, 9], [6, 7], [5, 7], \
                   [1, 2], [3, 8], [8, 9], [5, 9], [2, 5], [1, 3], [2, 3]])
        G.pachner31([1, 2, 6])
        
        H = Graph([[0, 1], [0, 6], [0, 7], [4, 6], [4, 8], [4, 9], [6, 7], [5, 7], \
                   [1, 2], [3, 8], [8, 9], [5, 9], [2, 5], [1, 3], [2, 3]])
        H.pachner31([2, 1, 6])
        
        self.assertEqual(G, H)

    #-------------------------------------------------------------------------#
    
#=============================================================================#

if __name__=='__main__':
    unittest.main()