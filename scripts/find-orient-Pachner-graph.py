# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:21:13 2026

@author: cartin
"""

import numpy as np

from collections import deque
from trivalent import Graph, decode_orient_state, encode_orient_state, \
    find_min_orient_state, permute_orient_state

#-----------------------------------------------------------------------------#

def compose_signed_perm(perm_A, perm_B):
    """
        With 1-based signed permutations, finds total permutation, assuming
        they are applied in the order (perm_B o perm_A)(x) = perm_B(perm_A(x)).
    """
    
    num_elem = len(perm_A)
    final_perm = np.zeros(num_elem, dtype = np.int8)
    
    for idx in range(num_elem):
        aaa = abs(perm_A[idx]) - 1
        aaa_sign = 1 if perm_A[idx] > 0 else -1
        
        bbb = abs(perm_B[aaa]) - 1
        bbb_sign = 1 if perm_B[aaa] > 0 else -1
        
        final_perm[idx] = aaa_sign * bbb_sign * (bbb + 1)
        
    return final_perm 
   
#-----------------------------------------------------------------------------#

# Initialization

numVert = 8
numEdges = 3 * numVert // 2

edgeMask = (1 << numEdges) - 1

# Create arrays for COO matrix, recording edge site used

src, tgt, movePerm, edges = [], [], [], []

# Import base matrix information for |V| value

data = np.load(f'../data/V={numVert}-Pachner-graph-base.npz')

graphList = data['edge_lists']

grpElem = data['grp_elem']
grpPtr = data['grp_ptr']

srcEdgeList = data['src_edge_idx']
srcEdgePtr = data['src_edge_ptr']
tgtGraphList = data['tgt_graphs']
movePermList = data['move_perm']

#-----------------------------------------------------------------------------#

# Read in information about all possible 2-2 moves

pachnerDict = {}
numGraph = len(graphList)

for graphNum in range(numGraph):
    pachnerDict[graphNum] = {}
    
    for iii in range(srcEdgePtr[graphNum], srcEdgePtr[graphNum + 1]):
        tgtGraphEdgeList = graphList[tgtGraphList[iii]]
        
        moveEdgeIdx = int(abs(movePermList[iii][srcEdgeList[iii]]) - 1)
        
        start, end = tgtGraphEdgeList[moveEdgeIdx]
        startEdgeInfo, endEdgeInfo = [], []
        
        # Read through final graph edges, find natural orientation directions
        # of other two edges incident on start, end vertices
        
        for edgeLabel, (u, v) in enumerate(tgtGraphEdgeList):
            if edgeLabel == moveEdgeIdx:
                continue
            
            if u == start:
                startEdgeInfo.append((edgeLabel, 0))
            elif v == start:
                startEdgeInfo.append((edgeLabel, 1))
                
            if u == end:
                endEdgeInfo.append((edgeLabel, 1))
            elif v == end:
                endEdgeInfo.append((edgeLabel, 0))

        # Put needed information into dict labeled with graph id, init edge idx
    
        pachnerDict[graphNum][srcEdgeList[iii]] = {
            'new_graph_num'   : int(tgtGraphList[iii]),
            'move_perm'       : movePermList[iii],
            'move_edge_idx'   : moveEdgeIdx,
            'start_edge_info' : startEdgeInfo,
            'end_edge_info'   : endEdgeInfo
            }

#-----------------------------------------------------------------------------#

# Create appropriate seed graph for |V|

G = Graph.create_prism(numVert // 2)

for graphNum, edgeList in enumerate(graphList):
    edgeMap = G.is_isomorphic(Graph(edgeList))
    
    if isinstance(edgeMap, np.ndarray):
        break

# Create list of edge orientations so they point as 2k -> 2k + 1, 2k -> 2k + 2,
# 2k + 1 -> 2k + 3. Unlike parity case, here we need to set *every* edge
# orientation appropriately; however, only two edges are set to 0, so use edge
# mask, then change these two.

seedState = edgeMask ^ (1 << 1) ^ (1 << 4)
seedState |= (graphNum << numEdges)    # Add graph ID at front of binary rep
        
seedState = permute_orient_state(seedState, edgeMap, numEdges)
seedState = find_min_orient_state(seedState, grpElem[grpPtr[graphNum] : grpPtr[graphNum + 1]], \
                                numEdges)

graphSet = set([seedState])
stateQueue = deque([seedState])

#-----------------------------------------------------------------------------#

# Create mock graph to access Graph class functions

G = Graph(graphList[0])

# Main loop: use BFS search to go through Pachner graph, finding all states,
# along with linking information

while stateQueue:
    
    currGraphState = stateQueue.popleft()
    currGraphNum, currOrientBits = decode_orient_state(currGraphState, numEdges)
    currSymGrp = grpElem[grpPtr[currGraphNum] : grpPtr[currGraphNum + 1]]

    # Find all non-isomorphic edges under graph symmmetries preserving edge
    # orientations, try 2-2 move on each
    
    nonIsoEdges = G.find_noniso_edges(mask_type = 'orient', bitmask = currOrientBits, \
                                      sym_grp = currSymGrp)

    for edgeLabel in nonIsoEdges:
        
        # Only edges giving valid 2-2 moves are in pachnerDict
        
        if edgeLabel not in pachnerDict[currGraphNum]:
            continue
        
        # Pre-calculated information for current graph, edge
            
        transition = pachnerDict[currGraphNum][edgeLabel]
        
        tgtGraphNum = transition['new_graph_num']
        nextGraphState = encode_orient_state(tgtGraphNum, currOrientBits, numEdges)
        moveOrientState = permute_orient_state(nextGraphState, transition['move_perm'], numEdges)
        
        # At each trivalent vertex, must have in-degree >= 1, out-degree >= 1
        
        moveEdgeIdx = transition['move_edge_idx']
        startOutDeg, endInDeg = 0, 0
        
        for adjLabel, idx in transition['start_edge_info']:
            startOutDeg += (idx ^ ((moveOrientState >> adjLabel) & 1))
            
        for adjLabel, idx in transition['end_edge_info']:
            endInDeg += (idx ^ ((moveOrientState >> adjLabel) & 1))
            
        # Try each possibility for the new edge orientation; since both could
        # be consistent, add to queue, final list within this check
        
        tgtSymGrp = grpElem[grpPtr[tgtGraphNum] : grpPtr[tgtGraphNum + 1]]
                
        if (startOutDeg != 2) and (endInDeg != 2):  # Orientation start -> end
            posOrientState = moveOrientState | (1 << moveEdgeIdx)
            
            minState, minPerm = find_min_orient_state(posOrientState, tgtSymGrp, \
                                                      numEdges, return_perm = True)
            
            if minState not in graphSet:
                stateQueue.append(minState)
                graphSet.add(minState)
                                          
            finalPerm = compose_signed_perm(transition['move_perm'], minPerm)
                
            src.append(currGraphState)
            tgt.append(minState)
            movePerm.append(finalPerm)
            edges.append(edgeLabel)
                
        if (startOutDeg != 0) and (endInDeg != 0):  # Orientation end -> start
            negOrientState = moveOrientState & ~(1 << moveEdgeIdx)
            
            minState, minPerm = find_min_orient_state(negOrientState, tgtSymGrp, \
                                                      numEdges, return_perm = True)
                                                      
            finalPerm = compose_signed_perm(transition['move_perm'], minPerm)
            
            if minState not in graphSet:
                stateQueue.append(minState)
                graphSet.add(minState)
                
            src.append(currGraphState)
            tgt.append(minState)
            movePerm.append(finalPerm)
            edges.append(edgeLabel)
                
# Current list of states is unsorted, so need to (1) sort src list, (2) match
# bitmasks to state indices in sorted list, then (3) match other results to
# this sorting

stateArray = np.array(sorted(graphSet), dtype = np.int64)

srcArray = np.asarray(src, dtype = np.uint64)
tgtArray = np.asarray(tgt, dtype = np.uint64)
edgeArray = np.asarray(edges, dtype = np.int8)

row = np.searchsorted(stateArray, srcArray)
col = np.searchsorted(stateArray, tgtArray)

# Verify that every state was found

assert np.array_equal(stateArray[row], srcArray)
assert np.array_equal(stateArray[col], tgtArray)

# Sort primarily by row, secondarily by edge

order = np.lexsort((edgeArray, row))

row = row[order]
col = col[order]

# Apply the same permutation to transition information

movePermArray = np.asarray(movePerm, dtype = np.int8)[order]
edgeArray = edgeArray[order]

# Save all information needed to reconstruct the graph

np.savez_compressed(
    f'../data/V={numVert}-Pachner-graph-orient.npz',
        states = stateArray,
        matrix_row = row,
        matrix_col = col,
        move_perm = movePermArray,
        move_edge = edgeArray
)