# -*- coding: utf-8 -*-
"""
Created on Sun May 11 08:43:37 2025

@author: cartin

In order to enumerate all possible cubic graphs, we need to do the following
operations:
    
    (1) identify edges, in order to create self-loops (valence 1 in dual) and
    multiedges (valence 2)
    (2) identify vertices (valence 3 in dual)
    (3) identify vertices at ends of single edge (valence 4 in dual)
    (4) identify vertices of two edges incident at the same third vertex
    (valence 5 in dual)
    

"""

from itertools import permutations

#=============================================================================#

class Vertex:
    
    def __init__(self, label = 0):
        
        self.label = label
        
        # Edge order is list of edges incident to vertex, given in CCW order
        
        self.edge_order = []
        
    #-------------------------------------------------------------------------#
        
    def __repr__(self):
        return repr(self.label)
        
    #-------------------------------------------------------------------------#
    
    def connectEdge(self, added_edge = None):
        """
            Connect given edge to vertex, placing next in CCW order
        """
        
        if len(self.edge_order) == 3:
            raise ValueError('Vertex {} already trivalent'.format(self.label))
            
        self.edge_order += [added_edge]
        
    #-------------------------------------------------------------------------#
    
    def replaceEdge(self, remove_edge = None, added_edge = None):
        """
            Remove edge from vertex cyclic order, place new edge in same location
        """
        
        if remove_edge not in self.edge_order:
            raise ValueError('Edge to remove not incident to vertex {}'.format(self.label))
            
        place = self.edge_order.index(remove_edge)
        self.edge_order[place] = added_edge
    
    #-------------------------------------------------------------------------#
    
    def removeEdge(self, remove_edge = None):
        
        if len(self.edge_order) == 0:
            raise ValueError('Vertex {} already disconnected'.format(self.label))
            
        if remove_edge in self.edge_order:
            self.edge_order.remove(remove_edge)
        
    #-------------------------------------------------------------------------#
    
    def edgeOrder(self):
        return [edge for edge in self.edge_order]

#=============================================================================#

class Edge:
    
    def __init__(self, start = None, end = None, color = None, twist = None):
        
        self.start = start
        self.end = end
        
        # Sign of twist gives whether twist is right-handed (+) as one travels
        # in direction of edge orientation, or left-handed (-). The orientation
        # is given by the sign of color, whether from start to end (+), or end
        # to start (-). Color is an integer, which is twice the representation
        # value j.
        
        self.color = color
        self.twist = twist
        
    #-------------------------------------------------------------------------#
        
    def __repr__(self):
        
        edge_str = "[" + repr(self.start) + ", " + repr(self.end)
        if self.color != None:
            edge_str += ", color = {}".format(self.color)
        if self.twist != None:
            edge_str += ", twist = {}".format(self.twist)
        edge_str += "]"
        
        return edge_str
        
    #-------------------------------------------------------------------------#
    
    def __eq__(self, other):
        """
            Test equality of start, end vertices *only* at the moment
        """
        
        if ((self.start == other.start) and (self.end == other.end)) or \
            ((self.start == other.end) and (self.end == other.start)):
                return True
        else:
            return False
        
        # # Twist is in same direction when traveling either direction along edge,
        # # so values must be same
        
        # if self.twist != other.twist:
        #     return False
        
        # # Must have same vertices and color size. If both edges have the same
        # # start and end vertices, the color is positive; if the vertices are
        # # flipped, then the color must be negative.
        
        # # Sign of color is orientation-dependent; if sizes are same, return
        # # relative sign, otherwise return False
        
        # if self.color == other.color:
        #     return 1
        # elif self.color == -other.color:
        #     return -1
        # else:
        #     return False
        
    #-------------------------------------------------------------------------#
    
    def __hash__(self):
        """
            Returns memory location of the edge as unique identifer
        """
        
        return id(self)
        
    #-------------------------------------------------------------------------#
    
    def setColor(self, color):
        
        if type(color) != int or color < 1:
            raise ValueError('color must be a positive integer')
        
        self.color = color
        
    #-------------------------------------------------------------------------#
    
    def changeStart(self, new_start = None):
        
        if type(new_start) != Vertex:
            raise ValueError('New edge start must be Vertex')
    
        if self in self.start.edgeOrder():
            self.start.removeEdge(self)
            
        self.start = new_start
        
    #-------------------------------------------------------------------------#
    
    def changeEnd(self, new_end = None):
        
        if type(new_end) != Vertex:
            raise ValueError('New edge end must be Vertex')
            
        if self in self.end.edgeOrder():
            self.end.removeEdge(self)
        
        self.end = new_end
        
    #-------------------------------------------------------------------------#
    
    def setEnd(self, end):
        """
            Adds end to edge, either by integer label or as a Vertex object.
            Does not check whether integer is within correct range for end.
        """
        
        if type(end) not in [int, Vertex]:
            raise ValueError('end must be either an integer label or a Vertex')
        elif type(end) == Vertex:
            self.end = end
        else:
            self.end = Vertex(label = end)
        
    #-------------------------------------------------------------------------#
    
    def setStart(self, start):
        """
            Adds start to edge, either by integer label or as a Vertex object.
            Does not check whether integer is within correct range for start.
        """
        
        if type(start) not in [int, Vertex]:
            raise ValueError('start must be either an integer label or a Vertex')
        elif type(start) == Vertex:
            self.start = start
        else:
            self.start = Vertex(label = start)
        
    #-------------------------------------------------------------------------#
    
    def setTwist(self, twist):
        
        if type(twist) != int:
            raise ValueError('twist must be an integer')
        
        self.twist = twist
        
    #-------------------------------------------------------------------------#
    
    def color(self):
        """
            Return color (twice the representation value) without orientation.
        """
        
        return abs(self.color)
        
    #-------------------------------------------------------------------------#
    
    def twist(self):
        """
            Return twist.
        """
        
        return self.twist
        
    #-------------------------------------------------------------------------#
    
    def start(self):
        """
            Returns start vertex for edge
        """
        
        return self.start
        
    #-------------------------------------------------------------------------#
    
    def end(self):
        """
            Returns end vertex for edge
        """
        
        return self.end
        
#=============================================================================#

class Graph:
    
    def __init__(self, num_vert = 0):
        
        if type(num_vert) != int or num_vert <= 0:
            raise ValueError('Number of vertices must be a positive integer')
        
        if (num_vert % 2) == 1:
            raise ValueError('Trivalent graphs must have an even number of vertices')
    
        self.num_vert = num_vert
        self.vert_list = []
        for iii in range(num_vert):
            self.vert_list += [Vertex(label = iii)]
        
        self.edge_list = []
        
    #-------------------------------------------------------------------------#
    
    def __repr__(self):
        return 'Graph of {} vertices and {} edges'.format(self.num_vert, len(self.edge_list))
        
    #-------------------------------------------------------------------------#
    
    def __eq__(self, other):
        
        # First, we check that the number of vertices and edges are equal; after
        # that, we check that the vertex degree lists are also the same.
        
        if self.num_vert != other.num_vert or len(self.edge_list) != len(other.edge_list):
            return False
        
        if sorted([len(vert.edge_order) for vert in self.vert_list]) != \
            sorted([len(vert.edge_order) for vert in other.vert_list]):
                return False
        
        # Use an edge-based solution for the multigraph isomorphism problem,
        # by matching only edges in the two graphs. We start with the first
        # edge in self, and try all possible starting matches in the edge list
        # for the other graph.
        
        for iii in range(len(other.edge_list)):
            
            # print('===\niii = {}'.format(iii))
            
            edge_match_dict = {}
            queue = [(self.edge_list[0], self.edge_list[0].start, \
                      other.edge_list[iii], other.edge_list[iii].start)]
            
            while len(queue) > 0:
                (current_self_edge, current_self_vert, \
                 current_other_edge, current_other_vert) = queue.pop()
                
                # Test whether matching the two edges is consistent, already
                # visited, or whether we need to add all other edges incident
                # to same vertex
                
                if edge_match_dict.get(current_self_edge, None):
                    if edge_match_dict[current_self_edge] != current_other_edge:
                        break
                    else:
                        continue    # Edge already visited, do not repeat
                else:
                    edge_match_dict[current_self_edge] = current_other_edge
                    
                # Use the cyclic orders of the two vertices to add to queue.
                # NOTE: we are assuming here that all vertices are trivalent
                
                self_shift = current_self_vert.edge_order.index(current_self_edge)
                other_shift = current_other_vert.edge_order.index(current_other_edge)
                
                for jjj in [1, 2]:
                    next_self_edge = current_self_vert.edge_order[(jjj + self_shift) % 3]
                    if current_self_vert != next_self_edge.start:
                        next_self_vert = next_self_edge.start
                    else:
                        next_self_vert = next_self_edge.end
                        
                    next_other_edge = current_other_vert.edge_order[(jjj + other_shift) % 3]
                    if current_other_vert != next_other_edge.start:
                        next_other_vert = next_other_edge.start
                    else:
                        next_other_vert = next_other_edge.end
                        
                    queue += [(next_self_edge, next_self_vert, \
                               next_other_edge, next_other_vert)]
                        
            # We have matched all edges in the two graphs, or else queue was
            # terminated early because of an inconsistency. If the former,
            # return True
            
            if len(edge_match_dict) == 3 * self.num_vert // 2 == len(set(edge_match_dict.values())):
                return True
            
        # No consistent solutions
        
        return False
    
    #-------------------------------------------------------------------------#
    
    def addEdgeList(self, added_edge_list):
        """
            Add edges to graph, given as list of Edge objects
        """
        
        if type(added_edge_list) != list:
            raise ValueError('Added edges must be given in list')
            
        if any([type(edge != Edge) for edge in added_edge_list]):
            raise ValueError('Entries in added edge list must be Edge object')
            
        # for 

    #-------------------------------------------------------------------------#
        
    def addEdges(self, added_edge_list):
        """
            Add edges to graph, given as list of vertex labels [start, end]
        """
        
        # Added edge list must be (1) a list of (2) lists in the form [start,
        # end], where (3) start, end are integers from 0 to num_vert - 1, and
        # (4) start < end.
        
        if type(added_edge_list) != list:
            raise ValueError('Added edges must be given in list')
            
        if any([(type(edge) != list or len(edge) != 2 or type(edge[0]) != int or type(edge[1]) != int) \
                for edge in added_edge_list]):
            raise ValueError('Entries in added edge list must be in form [start, end]')
            
        if any([(min(edge) < 0 or max(edge) >= self.num_vert) for edge in added_edge_list]):
            raise ValueError('Vertex labels must be between 0 and {}'.format(self.num_vert - 1))
            
        # Edges are added to vertex CCW in order given in added edge list;
        # Vertex.connectEdge tests whether too many edges added to vertex, which
        # also ensures total number of edges does not exceed 3/2 number of vertices.

        # Also, vertices are given as integers in added_edge_list, so we use
        # self.vert_list to change them into the appropriate Vertex objects.
        
        for edge in added_edge_list:
            start_vert = self.vert_list[edge[0]]
            end_vert = self.vert_list[edge[1]]
            
            new_edge = Edge(start = start_vert, end = end_vert)
            
            start_vert.connectEdge(new_edge)
            end_vert.connectEdge(new_edge)
            
            self.edge_list += [new_edge]
            
    #-------------------------------------------------------------------------#
        
    def addVertices(self, added_num_vert):
        """
            Add given number of vertices to graph.
        """
        
        if type(added_num_vert) != int or added_num_vert <= 0:
            raise ValueError('Number of added vertices must be positive integer')
        
        for iii in range(added_num_vert):
            self.vert_list += [Vertex(label = self.num_vert + iii)]
        
        self.num_vert += added_num_vert
        
    #-------------------------------------------------------------------------#
    
    def threeMove(self, vert_label = None):
        if vert_label != None:
            
            if type(vert_label) != int or vert_label < 0:
                raise ValueError('vertex label must be a positive integer')
                
            if vert_label >= self.num_vert:
                raise ValueError('vertex label must be between 0 and {}'.format(self.num_vert - 1))
            
            current_vert = self.vert_list[vert_label]
            adj_edges = current_vert.edgeOrder()
                
            # First, we create two new vertices. The two new vertices will have
            # labels larger than any other current vertices in the graph, so we
            # put them as the end of the vertex list.
            
            self.vert_list += [Vertex(label = self.num_vert), Vertex(label = self.num_vert + 1)]
            self.num_vert += 2
            
            # We have the current vertex x, and the three vertices a, b, c in
            # CCW order around it. So, before this move, the edges ax, bx, cx
            # appear in that order in the edge list for the graph. We create
            # two new vertices y, z, and do the following:
            #
            #   (1) add edges xy, xz immediately after ax
            #   (2) change edge bx to by, add edge yz immediately after
            #   (3) change edge cx to cz
            #
            # This gives the ordering ax xy xz ... by yz ... cz, so that the
            # cyclic orders around each vertex are given by
            #
            #   x: ayz
            #   y: xbz
            #   z: xyc
            #
            # The orderings around a, b, c are preserved, since we do not
            # change their relative ordering, but just alter what they are
            # connected to. Note that we could not move all of the existing
            # vertices to the beginning of each cyclic order, since this would
            # have given inconsistent orderings for x, y, z -- a would give yz,
            # b would give zx, and c would give xy.
            #
            # Remember that A[-2] comes before A[-1] in a Python list, so the
            # former corresponds to y, the latter to z.
            
            # (1) add edges xy, xz immediately after ax
            
            first_index = self.edge_list.index(adj_edges[0])
            
            other_vert_list = []
            
            for iii in [1, 2]:
                if adj_edges[iii].start != current_vert:
                    other_vert_list += [adj_edges[iii].start]
                    adj_edges[iii].end = None
                else:
                    other_vert_list += [adj_edges[iii].end]
                    adj_edges[iii].start = None
                    
                current_vert.removeEdge(adj_edges[iii])
                    
            edge_01 = Edge(start = current_vert, end = self.vert_list[-2])
            edge_02 = Edge(start = current_vert, end = self.vert_list[-1])
            
            self.edge_list = self.edge_list[:(first_index + 1)] + [edge_01, edge_02] + \
                self.edge_list[(first_index + 1):]
                
            current_vert.connectEdge(edge_01)
            current_vert.connectEdge(edge_02)
            self.vert_list[-2].connectEdge(edge_01)
            self.vert_list[-1].connectEdge(edge_02)
                
            # (2) change edge bx to by, add edge yz immediately after
            
            second_index = self.edge_list.index(adj_edges[1])
            
            if adj_edges[1].start == None:              # To ensure ordered vertex labels
                adj_edges[1].start = adj_edges[1].end
            adj_edges[1].end = self.vert_list[-2]
                
            edge_12 = Edge(start = self.vert_list[-2], end = self.vert_list[-1])
            
            self.edge_list = self.edge_list[:(second_index + 1)] + [edge_12] + \
                self.edge_list[(second_index + 1):]
                
            self.vert_list[-2].connectEdge(adj_edges[1])
            self.vert_list[-2].connectEdge(edge_12)
            self.vert_list[-1].connectEdge(edge_12)
            
            # (3) change edge cx to cz
            
            if adj_edges[2].start == None:              # To ensure ordered vertex labels
                adj_edges[2].start = adj_edges[2].end
            adj_edges[2].end = self.vert_list[-1]
                
            self.vert_list[-1].connectEdge(adj_edges[2])
            
    #-------------------------------------------------------------------------#
    
    def fourMove(self, edge_label = None):
        
        if edge_label != None:
            
            if type(edge_label) != int or edge_label < 0:
                raise ValueError('edge label must be positive integer')
                
            if edge_label >= 3 * self.num_vert // 2:
                raise ValueError('edge label must be between 0 and {}'.format(3 * self.num_vert // 2))
            
            current_edge = self.edge_list[edge_label]
            current_start = current_edge.start
            current_end = current_edge.end
                
            # First, we create two new vertices. The two new vertices will have
            # labels larger than any other current vertices in the graph, so we
            # put them as the end of the vertex list.
            
            self.vert_list += [Vertex(label = self.num_vert), Vertex(label = self.num_vert + 1)]
            self.num_vert += 2
            
            # We have the current edge with vertices x < y. The cyclic order of
            # x is of the form ax, bx, xy, while that for y is cy, dy, xy;
            # however, the ordering of these in the graph edge list is not
            # fixed, so there are 3 possibilities for each vertex x, y. We want
            # to create two new vertices z, w, and do the following:
            #
            #   (1) change edge bx to bz, add xz immediately *before* bz
            #   (2) change edge cy to cw, add wy immediately *after* cw
            #   (3) add wz at end of edge list
            #
            # Since we do not move bx -> bz, cy -> cw, then the b, c cyclic
            # orders remain the same. For the z order, we have rotated the
            # insertion of xz so that we get the correct order xz bz ... wz,
            # without affecting the w order cw wy ... wz. In particular, note
            # that bx -> xz bz preserves both the b, x orders, while cy ->
            # cw wy keeps both the c, y orders in place. This gives the cyclic
            # orders
            #
            #   x: azy
            #   y: dxw
            #   z: xbw
            #   w: cyz
            #
            # Since it is a little ambiguous in the description above, let
            # w be self.vert_list[-2] and z be self.vert_list[-1].
            
            # (1) change edge bx to bz, add xz immediately *before* bz
            
            x_index = current_start.edge_order.index(current_edge)
            edge_bx = current_start.edge_order[x_index - 1]
            
            if edge_bx.start == current_start:
                edge_bx.start = edge_bx.end
            edge_bx.end = self.vert_list[-1]
            
            edge_xz = Edge(start = current_start, end = self.vert_list[-1])
            
            bx_index = self.edge_list.index(edge_bx)
            self.edge_list = self.edge_list[:bx_index] + [edge_xz] + self.edge_list[bx_index:]
            
            self.vert_list[-1].connectEdge(edge_xz)
            self.vert_list[-1].connectEdge(edge_bx)     # Changed bx -> bz
            
            current_start.replaceEdge(remove_edge = edge_bx, added_edge = edge_xz)
            
            # (2) change edge cy to cw, add wy immediately *after* cw
            
            y_index = current_end.edge_order.index(current_edge)
            edge_cy = current_end.edge_order[(y_index + 1) % 3]     # Assume trivalent vertex
            
            if edge_cy.start == current_end:
                edge_cy.start = edge_cy.end
            edge_cy.end = self.vert_list[-2]
            
            edge_wy = Edge(start = current_end, end = self.vert_list[-2])
            
            cy_index = self.edge_list.index(edge_cy)    # Must do here, since index changes above
            self.edge_list = self.edge_list[:(cy_index + 1)] + [edge_wy] + \
                self.edge_list[(cy_index + 1):]
            
            self.vert_list[-2].connectEdge(edge_cy)
            self.vert_list[-2].connectEdge(edge_wy)
            
            current_end.replaceEdge(remove_edge = edge_cy, added_edge = edge_wy)
            
            # (3) add wz at end of edge list
            
            edge_wz = Edge(start = self.vert_list[-2], end = self.vert_list[-1])
            self.edge_list += [edge_wz]
            
            self.vert_list[-2].connectEdge(edge_wz)
            self.vert_list[-1].connectEdge(edge_wz)
        
    #-------------------------------------------------------------------------#
    
    def fiveMove(self):
        pass
        
    #-------------------------------------------------------------------------#
    
    def vertex(self, label = 0):
        """
            Returns Vertex object from vert_list at given index
        """
        
        if type(label) != int:
            raise ValueError('Vertex label must be integer')
            
        if label < 0 or label >= self.num_vert:
            raise ValueError('Vertex label must be between 0 and {}'.format(self.num_vert - 1))
            
        return self.vert_list[label]
        
    #-------------------------------------------------------------------------#
    
    def edge(self, label = 0):
        """
            Returns Edge object from edge_list at given index
        """
        
        if type(label) != int:
            raise ValueError('Edge label must be integer')
            
        if label < 0 or label >= 3 * self.num_vert // 2:
            raise ValueError('Vertex label must be between 0 and {}'.format(3 * self.num_vert // 2 - 1))
            
        return self.edge_list[label]
        
    #-------------------------------------------------------------------------#
    
    def numVert(self):
        return self.num_vert
        
    #-------------------------------------------------------------------------#
    
    def edgeList(self):
        return [edge for edge in self.edge_list]
    
    #-------------------------------------------------------------------------#