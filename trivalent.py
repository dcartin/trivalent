# -*- coding: utf-8 -*-
"""
Created on Sun May 11 08:43:37 2025

@author: cartin

To-do list:
    
* To add non-trivial colors, need to modify code in:
    OK Vertex.__init__
    OK Vertex.connectEdge
    OK Vertex.replaceEdge -- possibly gooned up, since self-loop code removed,
        or do we need to add such code to all vertex edge order methods?
    OK Vertex.removeEdge
    OK Edge.setColor
    Graph.__eq__ (check magnitude of color as well)
    Graph.pachner22
    
* What is the point of...
    Vertex.permuteOrder
    Vertex.addEdgeOrient
    
* Create Graph color list, then pass this along to every vertex; this only
creates one version, I think. Is there a reason to change the color list after
the Graph object is created?

* pachner22() changes edges, graph edge list in ways different than oneMove(),
etc? Refactor code so that methods are consistent.

* Need to change Graph.addEdges to reflect possibility of failed Vertex.connectEdge?

* Vertex.removeEdge:
	-- what if self-loop?
	-- is it dealing with color correctly?
    
* In Edge.setOrient, I seem to be checking the source/sink condition twice at
the incident vertices?

* __eq__ does not check that every edge compared has a color; this becomes an
issue with pachner22(), which potentially removes the color of one 
    
* Why are the following graphs being given as the same?
    [[0, 1, 1], [1, 2, 1], [0, 4, 1], [3, 4, -1], [2, 5, 1], [0, 5, -1], [1, 3, -1], [2, 3, -1], [4, 5, -1]]
    [[0, 1, 1], [1, 4, 1], [1, 5, -1], [2, 4, 1], [4, 5, 1], [0, 3, -1], [0, 2, -1], [3, 5, -1], [2, 3, -1]]
SOLUTION: There were some edge flips in the fourMove() algorithm that were not
reversing the edge color at the same time. This should be fixed; is this tested
in test_trivalent?

* I think the following graphs should not be equivalent, since the cyclic orders
for vertices 1, 4 of the second graph do not match the first?
    [[0, 1], [0, 4], [0, 5], [1, 2], [3, 4], [1, 3], [4, 5], [2, 5], [2, 3]]
    [[0, 1], [0, 4], [4, 5], [0, 2], [3, 4], [1, 3], [3, 5], [2, 5], [1, 2]]
SOLUTION: These are not, if you create graphs using these edge lists; the
second graph is not planar. However, the second was created by an erroneous
algorthm for the Pachner 2-2 move, so that the vertex cyclic orders were not
consistent with the overall graph edge list. This should be fixed.

* HOW TO CHECK that the cyclic orders and edge list are consistent for a graph?
"""

from itertools import product

#=== Helper functions ========================================================#

def colorList(r):
    """
    Find all possible colorings of stubs incident on a trivalent vertex when
    using representations of SU(2)_q, with q = exp(i pi / r) and r >= 3. The
    conditions for the three colors i, j, k are
    
    (1) i, j, k from [0, 1, ..., r - 1]
    (2) i + j - k is positive and even, as are j + k - i, k + i - j
    (3) i + j + k <= 2r - 4
    
    Parameters
    ----------
    r : int
        Parameter defining root of unity q for representations
        of SU(2)_q used to color edges.

    Returns
    -------
    list
        Sorted list of all possible colorings for stubs incident
        to a trivalent vertex, for the chosen representations of SU(2)_q.

    """
    
    # Create list to keep results in
    
    edge_color_list = []
    
    # Create lists to increment colors
    # along two of three edges
    
    add12 = [1, 1, 0]
    add13 = [1, 0, 1]
    add23 = [0, 1, 1]
    
    # Start queue with all three colors
    # equal to zero, then build up other
    # possibilities by adding one to
    # two of the edges
    
    edge_color_queue = [[0, 0, 0]]
    
    while len(edge_color_queue) > 0:
        current_vertex = edge_color_queue.pop()
        
        # Add to results lists, if not already there,
        # then process; otherwise, skip this list
        
        if (0 not in current_vertex) and (current_vertex not in edge_color_list):
            edge_color_list += [current_vertex]
        
        # Create next possible entries for queue,
        # if adding two to color sum does not
        # exceed limit on sum
        
        if sum(current_vertex) <= (2 * r - 6):
            
            # Increment colors on 1st, 2nd edges, then
            # see if this is already in queue
            
            next_vertex = sorted([sum(iii) for iii in zip(current_vertex, add12)])
            
            if next_vertex not in edge_color_queue:
                edge_color_queue += [next_vertex]
            
            # Increment colors on 1st, 3rd edges, then
            # see if this is already in queue
            
            next_vertex = sorted([sum(iii) for iii in zip(current_vertex, add13)])
            
            if next_vertex not in edge_color_queue:
                edge_color_queue += [next_vertex]
                
            # Increment colors on 2nd, 3rd edges, then
            # see if this is already in queue
            
            next_vertex = sorted([sum(iii) for iii in zip(current_vertex, add23)])
            
            if next_vertex not in edge_color_queue:
                edge_color_queue += [next_vertex]
                
    # Return resulting list
    
    return sorted(edge_color_list)

#-----------------------------------------------------------------------------#

class UnionFind:
    
    def __init__(self, num_edges):
        self.num_edges = num_edges
        self.root_list = list(range(num_edges))
        
    def findRoot(self, label):
        if self.root_list[label] == label:
            return label
        
        return self.findRoot(self.root_list[label])
    
    def join(self, iii, jjj):
        iii_root = self.findRoot(iii)
        jjj_root = self.findRoot(jjj)
        
        self.root_list[max(iii_root, jjj_root)] = min(iii_root, jjj_root)
        
    def printRoots(self):
        return [self.root_list[iii] for iii in range(self.num_edges) \
                if iii == self.root_list[iii]] 

#=============================================================================#

class Vertex:
    
    def __init__(self, label = None, COLOR_LIST = None):
        
        if label and (type(label) != int or label < 0):
            raise ValueError('vertex label must be non-negative integer')
            
        if COLOR_LIST and (type(COLOR_LIST) != list or not all([len(triad) == 3 for triad in COLOR_LIST]) or \
            not all([all([type(label) == int for label in triad]) for triad in COLOR_LIST])):
                
            raise ValueError('color list must be list of integer triples')
        
        self.label = label
        
        # Edge order is list of edges incident to vertex, given in CCW order
        
        self.edge_order = []
        
        # Color list records colors for edges incident to vertex, to ensure
        # they meet SU_q(2) requirements; allowed color list constant records
        # all triples that meet these requirements.
        
        self.color_list = []
        
        if COLOR_LIST:
            self.ALLOWED_COLOR_LIST = COLOR_LIST
        else:
            self.ALLOWED_COLOR_LIST = []
        
        # Arrow list keeps track of edge orientations pointing into (+1) or
        # out of (-1) the vertex. If zero is given, edge has no orientation.
        
        self.in_arrow = []
        
        # List of faces that are incident to the vertex
        
        self.face_list = []
        
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
            
        # if self != added_edge.start or self != added_edge.end:
        #     raise ValueError('Vertex {} not incident to added edge'.format(self.label))
        
        # Check whether edge has orientation; if so, compute whether this
        # orientation points towards (+) or away from (-) the vertex. This
        # value is stored separately, just in case a source/sink would be
        # created by the connected edge.
        
        if added_edge.orient:
            if (added_edge.orient > 0 and self == added_edge.end) or \
                (added_edge.orient < 0 and self == added_edge.start):
                    
                added_arrow = 1
            else:
                added_arrow = -1
        else:
            added_arrow = 0
        
        # Check here if source or sink created
        
        if abs(sum(self.in_arrow) + added_arrow) == 3:
            raise AttributeError('Edge orientation results in source or sink')
            
        # Check that colors assigned are valid; only check if all three colors
        # are assigned (i.e. no zeros appear in list)
        
        if added_edge.color:
            added_color = added_edge.color
        else:
            added_color = 0
            
        temp_color_list = sorted([color for color in self.color_list] + [added_color])
        if (0 not in temp_color_list) and len(temp_color_list) == 3 and \
            temp_color_list not in self.ALLOWED_COLOR_LIST:
            raise AttributeError('Edge color does not satisfy vertex rules')
        
        # All tests passed
            
        self.edge_order += [added_edge]
        self.color_list += [added_color]
        self.in_arrow += [added_arrow]
        
        return True
        
    #-------------------------------------------------------------------------#
    
    def replaceEdge(self, remove_edge = None, added_edge = None):
        """
            Remove edge from vertex cyclic order, place new edge in same location
        """
        
        if remove_edge not in self.edge_order:
            raise ValueError('Edge to remove not incident to vertex {}'.format(self.label))
            
        place = self.edge_order.index(remove_edge)
        
        # If remove_edge is a self-loop, replace the second instance of the
        # edge; assumes trivalent vertex, and that we are not replacing a
        # self-loop with another self-loop
            
        if added_edge.orient:
            if (added_edge.orient > 0 and self == added_edge.end) or \
                (added_edge.orient < 0 and self == added_edge.start):
                    
                replace_arrow = 1
            else:
                replace_arrow = -1
        else:
            replace_arrow = 0
            
            # FOR NOW, assume that there is no self-loop involved; FIX LATER
            
            # if self.edge_order[(place + 1) % 3] == remove_edge:
            #     self.edge_order[(place + 1) % 3] = added_edge
            #     place = (place + 1) % 3
            # else:
            #     self.edge_order[place] = added_edge
            # self.in_arrow[place] = 0
            
            # return True
        
        # Since self.in_arrow already has the old value, we added twice the
        # new value; if this is equal to +/- 3, then we have a source or sink.
        # If orientation does not change, then test is unnecessary.
    
        if replace_arrow != self.in_arrow[place] and abs(sum(self.in_arrow) + 2 * replace_arrow) == 3:
            raise AttributeError('Orientation change results in source or sink')
            
        # Check that colors are still valid at vertex
        
        if added_edge.color:
            replace_color = added_edge.color
        else:
            replace_color = 0
            
        if replace_color != self.color_list[place]:
            temp_color_list = [color for color in self.color_list]
            temp_color_list[place] = replace_color
            
            if (0 not in temp_color_list) and len(temp_color_list) == 3 and \
                temp_color_list not in self.ALLOWED_COLOR_LIST:
                raise AttributeError('Edge color does not satisfy vertex rules')
            
        # All tests passed
        
        self.edge_order[place] = added_edge
        self.color_list[place] = replace_color
        self.in_arrow[place] = replace_arrow
            
        # else:
        #     if self.edge_order[(place + 1) % 3] == remove_edge:
        #         self.edge_order[(place + 1) % 3] = added_edge
        #         place = (place + 1) % 3
        #     else:
        #         self.edge_order[place] = added_edge
        #     self.in_arrow[place] = replace_arrow
            
        #     return True
    
    #-------------------------------------------------------------------------#
    
    def removeEdge(self, remove_edge = None):
        
        if len(self.edge_order) == 0:
            raise ValueError(f'Vertex {self.label} already disconnected')
            
        if remove_edge in self.edge_order:
            place = self.edge_order.index(remove_edge)
            
            del self.edge_order[place]
            del self.color_list[place]
            del self.in_arrow[place]
        else:
            raise ValueError(f'Edge {remove_edge} not connected to vertex {self.label}')
            
    #-------------------------------------------------------------------------#
    
    def changeEdgeOrient(self, change_edge = None):
        """
            Reverse the orientation of change_edge at the vertex
        """
        
        if type(change_edge) != Edge:
            raise ValueError('Must have Edge object to change edge orientation')
            
        edge_index = self.edge_order.index(change_edge)
        new_in_arrow_list = [arrow for arrow in self.in_arrow]
        new_in_arrow_list[edge_index] = -new_in_arrow_list[edge_index]
        
        if abs(sum(new_in_arrow_list)) == 3:
            raise AttributeError('Orientation change results in source or sink')
        else:
            self.in_arrow[edge_index] = -self.in_arrow[edge_index]
        
    #-------------------------------------------------------------------------#
    
    def addFace(self, new_face):
        if type(new_face) != Face:
            raise ValueError('incident face must be Face object')
        else:
            self.face_list += [new_face]
        
    #-------------------------------------------------------------------------#
    
    def colorList(self):
        return [color for color in self.color_list if color != 0]
        
    #-------------------------------------------------------------------------#
    
    def inArrow(self):
        return [dir for dir in self.in_arrow]
        
    #-------------------------------------------------------------------------#
    
    def edgeOrder(self):
        return [edge for edge in self.edge_order]
        
    #-------------------------------------------------------------------------#
    
    def label(self):
        return self.label

#=============================================================================#

class Edge:
    
    def __init__(self, start = None, end = None, orient = None, color = None, twist = None):
        
        if start and type(start) != Vertex:
            raise ValueError('start must be a Vertex object')
        
        if end and type(end) != Vertex:
            raise ValueError('end must be a Vertex object')
            
        if orient and (type(orient) != int or abs(orient) != 1):
            raise ValueError('orient must be +/- 1')
            
        if color and (type(color) != int or color <= 0):
            raise ValueError('color must be positive integer')
        
        self.start = start
        self.end = end
        
        # Sign of twist gives whether twist is right-handed (+), or left-handed
        # (-). The orientation is given by +/- 1, whether from start to end (+1),
        # or end to start (-1). Color is an integer, which is twice the
        # representation value j.
        
        self.orient = orient
        self.color = color
        self.twist = twist
        
        # Record faces incident to the edge; the sides are given in terms of
        # the natural orientation of the edge from start to finish
        
        self.face_left = None
        self.face_right = None
        
    #-------------------------------------------------------------------------#
        
    def __repr__(self):
        
        # edge_str = "[" + repr(self.start) + ", " + repr(self.end)
        # if self.orient != None:
        #     edge_str += ", orient = " + repr(self.orient)
        # if self.color != None:
        #     edge_str += ", color = " + repr(self.color)
        # # if self.twist != None:
        # #     edge_str += ", twist = {}".format(self.twist)
        # edge_str += "]"
        
        if self.twist != None:
            return "[" + repr(self.start) + ", " + repr(self.end) + ", " + \
                repr(self.orient) + ", " + repr(self.color) + ", " + repr(self.twist) + \
                + "]"
        elif self.color != None:
            return "[" + repr(self.start) + ", " + repr(self.end) + ", " + \
                repr(self.orient) + ", " + repr(self.color) + "]"
        elif self.orient != None:
            return "[" + repr(self.start) + ", " + repr(self.end) + ", " + \
                repr(self.orient) + "]"
        else:
            return "[" + repr(self.start) + ", " + repr(self.end) + "]"
        
    #-------------------------------------------------------------------------#
    
    def __hash__(self):
        """
            Returns memory location of the edge as unique identifer
        """
        
        return id(self)
        
    #-------------------------------------------------------------------------#
    
    def clearColor(self):
        self.color = None
        
        start_index = self.start.edge_order.index(self)
        self.start.color_list[start_index] = 0
        
        end_index = self.end.edge_order.index(self)
        self.end.color_list[end_index] = 0
    
    #-------------------------------------------------------------------------#
    
    def setColor(self, color = None):
        
        # NOTE: Currently assumes self.start, self.end exist, and that edge
        # is already attached to both
        
        if color == None:
            start_place = self.start.edge_order.index(self)
            end_place = self.end.edge_order.index(self)
            
            self.color = None
            self.start.color_list[start_place] = 0
            self.end.color_list[end_place] = 0
            
            return True
        
        if type(color) != int or color <= 0:
            raise ValueError('color must be positive integer')
            
        if color == self.color:
            raise ValueError(f'color is already defined to be {color}')
        
        start_color_list = [color for color in self.start.color_list]
        start_place = self.start.edge_order.index(self)
        start_color_list[start_place] = color
        start_color_list.sort()
        
        end_color_list = [color for color in self.end.color_list]
        end_place = self.end.edge_order.index(self)
        end_color_list[end_place] = color
        end_color_list.sort()
        
        # Only check vertex color lists if they have no placeholders
        
        if (0 not in start_color_list) and (start_color_list not in self.start.ALLOWED_COLOR_LIST):
            raise AttributeError('color requirements violated at edge start')
        elif (0 not in end_color_list) and (end_color_list not in self.end.ALLOWED_COLOR_LIST):
            raise AttributeError('color requirements violated at edge end')
                   
        self.color = color
        self.start.color_list[start_place] = color
        self.end.color_list[end_place] = color
        
        # self.start.color_list = start_color_list
        # self.end.color_list = end_color_list
           
        return True
        
    #-------------------------------------------------------------------------#
    
    def setOrient(self, orient):
        
        # NOTE: Currently assumes self.start, self.end exist, and that edge
        # is already attached to both
        
        if type(orient) != int or abs(orient) != 1:
            raise ValueError('orient must be +/- 1')
            
        if orient == self.orient:
            return True
                
        start_place = self.start.edge_order.index(self)
        end_place = self.end.edge_order.index(self)
        
        if self.orient:
            if (self.orient > 0 and orient < 0) or (self.orient < 0 and orient > 0):
                
                # Orientation exists and flips; need to check that there is no
                # violation of source/sink rule at both incident vertices
                
                start_in_arrow_list = [arrow for arrow in self.start.in_arrow]
                start_in_arrow_list[start_place] = -start_in_arrow_list[start_place]
                
                end_in_arrow_list = [arrow for arrow in self.end.in_arrow]
                end_in_arrow_list[end_place] = -end_in_arrow_list[end_place]
                    
        else:
            
            # Edge has no previous orient, so do check at vertices with added
            # arrow at each, and verify no sources or sinks at both incident
            # vertices
            
            start_in_arrow_list = [arrow for arrow in self.start.in_arrow]
            start_place = self.start.edge_order.index(self)
            
            end_in_arrow_list = [arrow for arrow in self.end.in_arrow]
            end_place = self.end.edge_order.index(self)
            
            if orient > 0:
                start_in_arrow = -1
                end_in_arrow = 1
            else:
                start_in_arrow = 1
                end_in_arrow = -1
                
            start_in_arrow_list[start_place] = start_in_arrow
            end_in_arrow_list[end_place] = end_in_arrow
            
        # Verify that no sources or sinks are created at incident vertices
            
        if abs(sum(start_in_arrow_list)) != 3 and abs(sum(end_in_arrow_list)) != 3:
            self.orient = orient
            
            self.start.in_arrow = start_in_arrow_list
            self.end.in_arrow = end_in_arrow_list
            
            return True
        
        elif abs(sum(start_in_arrow_list)) == 3:
            raise AttributeError('Orientation change results in source/sink at start')
        else:
            raise AttributeError('Orientation change results in source/sink at end')
            
        # WHAT IS THIS PART OF CODE FOR?
            
        # # We need to check that orientations for start, end vertices are
        # # consistent with requirement of no sources, sinks, so we keep the
        # # original orient, and restore it if one of the vertex conditions is
        # # violated.
        
        # try:
            
        #     self.start.changeEdgeOrient(self)        
        #     self.end.changeEdgeOrient(self)
            
        #     self.orient = orient
            
        #     return True
        
        # except:
        #     return False
        
    #-------------------------------------------------------------------------#
    
    def changeStart(self, new_start = None):
        
        if type(new_start) != Vertex:
            raise ValueError('New edge start must be Vertex')
    
        if self in self.start.edgeOrder():
            self.start.removeEdge(self)
        else:
            raise AttributeError('Edge not in edge order for new start vertex')
            
        self.start = new_start
        
    #-------------------------------------------------------------------------#
    
    def changeEnd(self, new_end = None):
        
        if type(new_end) != Vertex:
            raise ValueError('New edge end must be Vertex')
            
        if self in self.end.edgeOrder():
            self.end.removeEdge(self)
        else:
            raise AttributeError('Edge not in edge order for new end vertex')
        
        self.end = new_end
        
    #-------------------------------------------------------------------------#
    
    def notSelfLoop(self):
        if self.start != self.end:
            return True
        
        return False
        
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
    
    def setLeftFace(self, left_face):
        if type(left_face) != Face:
            raise ValueError('left face must be a Face object')
        else:
            self.left_face = left_face
        
    #-------------------------------------------------------------------------#
    
    def setRightFace(self, right_face):
        if type(right_face) != Face:
            raise ValueError('right face must be a Face object')
        else:
            self.right_face = right_face
        
    #-------------------------------------------------------------------------#
    
    def setTwist(self, twist):
        
        if type(twist) != int:
            raise ValueError('twist must be an integer')
        
        self.twist = twist
        
    #-------------------------------------------------------------------------#
    
    def color(self):
        """
            Return color (twice the spin representation value).
        """
        
        return abs(self.color)
        
    #-------------------------------------------------------------------------#
    
    def orient(self):
        """
            Return orientation
        """
        
        return abs(self.orient)
        
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

class Face:
    
    def __init__(self):
        
        self.edge_list = []
        
    #-------------------------------------------------------------------------#
        
    def __repr__(self):
        return repr([edge for edge in self.edge_list])
    
    #-------------------------------------------------------------------------#
        
    def addEdge(self, added_edge = None):
        if type(added_edge) == Edge:
            self.edge_list += [added_edge]
        
#=============================================================================#

class Graph:
    
    def __init__(self, num_vert = None, r = None):
    
        # r value, to determine allowed incident edge colors; allowed color
        # list constant gives all allowed triples coming from r
            
        if r:
            if (type(r) != int or r <= 3):
                raise ValueError('r value must be >= 4')
            else:
                self.ALLOWED_COLOR_LIST = colorList(r)
        else:
            
            # Use default option of all 0's to allow testing with orientations
            # but no chosen value of r. This choice also avoids confusion if
            # colors are added to some of the edges.
            
            self.ALLOWED_COLOR_LIST = [[0, 0, 0]]
                
        self.R = r
        
        # Number of vertices
        
        self.vert_list = []
        
        if num_vert:
            if (type(num_vert) != int or num_vert <= 0):
                raise ValueError('Number of vertices must be a positive integer')
        
            if (num_vert % 2) == 1:
                raise ValueError('Trivalent graphs must have an even number of vertices')
                
            # Create vertices
                
            self.num_vert = num_vert
            
            for iii in range(num_vert):
                self.vert_list += [Vertex(label = iii, COLOR_LIST = self.ALLOWED_COLOR_LIST)]
        else:
            self.num_vert = 0
        
        # Boolean value to record whether edges are in canonical order
        
        self.canonical = False
        
        # Create edge list
        
        self.edge_list = []
        
        # Create list of faces
        
        self.face_list = []
        
    #-------------------------------------------------------------------------#
    
    def __repr__(self):
        return f'Graph of {self.num_vert} vertices and {len(self.edge_list)} edges'
        
    #-------------------------------------------------------------------------#
    
    def __copy__(self):
        
        # Create new graph; this automatically creates new vertices as well
        
        new_graph = Graph(num_vert = self.num_vert, r = self.R)
        
        # Copy edges with info
        
        for edge in self.edge_list:
            new_graph.addEdges([[edge.start.label, edge.end.label, edge.orient, \
                                edge.color, edge.twist]])
                
        return new_graph
        
    #-------------------------------------------------------------------------#
    
    def __eq__(self, other):
        
        # # First, we check that the number of vertices and edges are equal; after
        # # that, we check that the vertex degree lists are also the same.
        
        # if self.num_vert != other.num_vert or len(self.edge_list) != len(other.edge_list):
        #     return False
        
        # if sorted([len(vert.edge_order) for vert in self.vert_list]) != \
        #     sorted([len(vert.edge_order) for vert in other.vert_list]):
        #         return False
            
        # # Use hash functions to test structure equality
        
        # if self.hashTuple() != other.hashTuple():
        #     return False
        
        # # Tests for edge orientations, colors
        
        # if [edge.orient for edge in self.edge_list] != [edge.orient for edge in other.edge_list]:
        #     return False
        
        # if [edge.color for edge in self.edge_list] != [edge.color for edge in other.edge_list]:
        #     return False
        
        # return True
        
        #=====================================================================#
        
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
        
        for (iii, other_vert_flag) in product(range(3 * self.num_vert // 2), [True, False]):
            
            edge_match_dict = {}
            
            # Given an oriented starting edge in the current graph, we find the
            # orientation of the edge in the other graph, and use the vertex
            # which has the correct 'side' of the orientation as the current
            # graph, i.e. either both arrows are out of their respective
            # vertices, or both are in.
            
            # Since edges without orientation do not give a natural way to start
            # the process, we try both possibilities for the matching vertex
            # incident to the edge in the other graph.
            
            if self.edge_list[0].orient and other.edge_list[iii].orient:
                
                if (self.edge_list[0].orient > 0 and other.edge_list[iii].orient > 0) or \
                    (self.edge_list[0].orient < 0 and other.edge_list[iii].orient < 0):
                        
                    other_vert = other.edge_list[iii].start
                else:
                    other_vert = other.edge_list[iii].end
            else:
                if other_vert_flag:
                    other_vert = other.edge_list[iii].start
                else:
                    other_vert = other.edge_list[iii].end
            
            queue = [(self.edge_list[0], self.edge_list[0].start, \
                      other.edge_list[iii], other_vert)]
                          
            while len(queue) > 0:
                
                (current_self_edge, current_self_vert, \
                 current_other_edge, current_other_vert) = queue.pop()
               
                # Test whether matching the edge, vertex combo are consistenly mapped
                # from the self graph to the other graph, if one or the other has
                # already been visited, or whether we need to continue and add all
                # other edges incident to same vertex
                
                if edge_match_dict.get(current_self_edge, None):
                    if edge_match_dict[current_self_edge] != current_other_edge:
                        break       # Inconsistent edge matching
                    elif edge_match_dict[current_self_edge] == current_other_edge:
                        continue    # Edge already visited, do not repeat
                    
                # If the edges are oriented, determine whether the orientations
                # of the two edges are consistent, i.e. both current_self_edge
                # and current_other_edge point towards current_self_vert,
                # current_other_vert, or both away. If they are inconsistent,
                # then so is mapping.
                
                if current_self_edge.orient and current_other_edge.orient:
                    
                    # Orientations are not consistent, so move on to next choice
                    
                    if abs(current_self_edge.orient) != abs(current_other_edge.orient):
                        break
                    
                    if (current_self_vert == current_self_edge.start and current_self_edge.orient < 0) \
                        or (current_self_vert == current_self_edge.end and current_self_edge.orient > 0):
                        
                        current_dir = True      # Orientation towards current_self_vert
                        
                    else:
                        current_dir = False
                        
                    if (current_other_vert == current_other_edge.start and current_other_edge.orient < 0) \
                        or (current_other_vert == current_other_edge.end and current_other_edge.orient > 0):
                            
                        other_dir = True        # Orientation towards current_other_vert
                        
                    else:
                        other_dir = False
                        
                    if current_dir != other_dir:
                        break       # Inconsistent edge orientation
                        
                # Check that edge colors are equal, if they exist
                
                if current_self_edge.color and current_other_edge.color and \
                    current_self_edge.color != current_other_edge.color:
                        break       # Inconsistent edge colors
                        
                # Either edge orientations are consistent, or no orient value
                # and a mapping between edges does not already exist.
               
                edge_match_dict[current_self_edge] = current_other_edge
                    
                # Go through and find other edges incident to current_self_vert, and
                # add them to the queue. The vertices are those on the opposite side
                # of each edge from the current vertices. Note that if we add the
                # edge, vertex combos into the queue in CCW order, they will be
                # removed via pop in CW order.
                
                self_shift = current_self_vert.edge_order.index(current_self_edge)
                other_shift = current_other_vert.edge_order.index(current_other_edge)
                
                for shift in [-2, -1]:
                    next_self_edge = current_self_vert.edge_order[self_shift + shift]
                    if current_self_vert != next_self_edge.start:
                        next_self_vert = next_self_edge.start
                    else:
                        next_self_vert = next_self_edge.end
                    
                    next_other_edge = current_other_vert.edge_order[other_shift + shift]
                    if current_other_vert != next_other_edge.start:
                        next_other_vert = next_other_edge.start
                    else:
                        next_other_vert = next_other_edge.end
                        
                    queue += [(next_self_edge, next_self_vert, next_other_edge, next_other_vert)]
  
            # We have matched all edges in the two graphs, or else queue was
            # terminated early because of an inconsistency. If the former,
            # return True
            
            if len(edge_match_dict) == 3 * self.num_vert // 2 == len(set(edge_match_dict.values())):
                # return edge_match_dict
                return True
            
        # No consistent solutions
        
        # return edge_match_dict
        return False
    
    #-------------------------------------------------------------------------#
    
    def hashTuple(self):
        
        if self.canonical:
            return tuple([self.edge_list.index(edge) for vertex in self.vert_list \
                          for edge in vertex.edge_order])
        else:
            num_edges = 3 * self.num_vert // 2
            current_edge_index_dict = {edge: idx for idx, edge in enumerate(self.edge_list)}
            min_hash = tuple([current_edge_index_dict[edge] for vert in self.vert_list for edge in vert.edge_order])
            # final_edge_dict = {self.edge_list[label] : label for label in range(num_edges)}
            
            for start_vert_label, start_vert in enumerate(self.vert_list):
                current_cyclic_order = [edge for edge in start_vert.edge_order]
                
                # The zero shift choice will always have a tuple starting with
                # (0, 1, 2, ...), so we remove the other shifts.
                
                vert_map_dict = {vert: self.num_vert for vert in self.vert_list}
                inv_vert_map_list = [None] * self.num_vert
                temp_edge_dict = {edge: num_edges for edge in self.edge_list}
                temp_cyclic_order = [[num_edges] * 3 for _ in range(self.num_vert)]
                
                # Here, we begin with start_vertex, and label it and its
                # incident edges with new labels. We do not need to update
                # temp_cyclic_order, since the first three edges will
                # always have the same labels.
                
                vert_map_dict[start_vert], inv_vert_map_list[0] = 0, start_vert
                # current_vert_label, next_vert_label, next_edge_label = 0, 0, 0
                
                for cycle_index in range(3):
                    current_edge = current_cyclic_order[cycle_index]
                    
                    # Here, we assume a simple graph with no self-loops or
                    # multiple edges
                    
                    if start_vert == current_edge.start:
                        other_vert = current_edge.end
                    else:
                        other_vert = current_edge.start
                        
                    vert_map_dict[other_vert] = cycle_index + 1
                    inv_vert_map_list[cycle_index + 1] = other_vert
                    temp_edge_dict[current_edge] = cycle_index
                    
                temp_cyclic_order[0] = [temp_edge_dict[edge] for edge in start_vert.edge_order]
                
                # ADDED NEXT PART ----------------------------------------#
                
                temp_hash = [temp_edge_dict[edge] for edge in start_vert.edge_order]
                # print(temp_cyclic_order[0], min_shash)
                
                # if temp_cyclic_order[0] > list(min_hash[0:3]):
                #     continue
                
                #---------------------------------------------------------#
                    
                # Go through remaining vertices, and label both them and
                # their incident edges
                    
                current_vert_label, next_vert_label, next_edge_label = 1, 4, 3
                minFlag = False
                
                while (not minFlag) and current_vert_label < self.num_vert:
                    current_vert = inv_vert_map_list[current_vert_label]
                    # print(current_vert_label, current_vert)
                    
                    for edge in current_vert.edge_order:
                        if current_vert == edge.start:
                            other_vert = edge.end
                        else:
                            other_vert = edge.start
                            
                        if temp_edge_dict[edge] == num_edges:
                            temp_edge_dict[edge] = next_edge_label
                            next_edge_label += 1
                                
                        if vert_map_dict[other_vert] == self.num_vert:
                            vert_map_dict[other_vert] = next_vert_label
                            inv_vert_map_list[next_vert_label] = other_vert
                            next_vert_label += 1
                        
                    temp_cyclic_order[current_vert_label] = [temp_edge_dict[edge] for edge in current_vert.edge_order]
                    
                    # ADDED NEXT PART ------------------------------------#
                    
                    temp_hash += [temp_edge_dict[edge] for edge in current_vert.edge_order]
                    # min_label = min(this_vert_order)
                    # shift = (-this_vert_order.index(min_label)) % 3
                    
                    # this_vert_order = [this_vert_order[iii - shift] for iii in range(3)]
                    # print(temp_hash, min_hash, \
                    #     temp_hash <= list(min_hash[0 : 3 * (current_vert_label + 1)]))
                        
                    # if temp_hash > list(min_hash[0 : 3 * (current_vert_label + 1)]):
                    #     minFlag = True
                    #     break
                    
                    #-----------------------------------------------------#
                    
                    # Put minimal edge label first, since this allows early
                    # breaking if this list is
                    
                    current_vert_label += 1
                    
                # Since we drop the first vertex in self.vert_list, the
                # vertex labels must be one smaller than the actual numbers
                
                # for vert_label in range(self.num_vert):
                #     min_label = min(temp_cyclic_order[vert_label])
                    
                #     shift = (-temp_cyclic_order[vert_label].index(min_label)) % 3
                #     temp_cyclic_order[vert_label] = [temp_cyclic_order[vert_label][iii - shift] \
                #                                      for iii in range(3)]
                        
                temp_hash = tuple(edge_label for vert_label in range(self.num_vert) \
                                   for edge_label in temp_cyclic_order[vert_label])
                    
                if temp_hash < min_hash:
                # if (not minFlag) and temp_hash < min_hash:  # ADDED
                    min_hash = temp_hash
                    final_edge_dict = temp_edge_dict
                # else:
                    # print('Final:', False)
                    
                print(start_vert_label, temp_edge_dict, temp_cyclic_order)
                        
            # Record that hash function has been calculated for graph, and
            # update ordering of edges in self.edge_list
                        
            self.canonical = True
            self.edge_list = sorted(self.edge_list, key = lambda edge : final_edge_dict[edge])
                        
            return tuple(min_hash)

    #-------------------------------------------------------------------------#
    
    def graphSym(self, non_iso_edges = True):
        graph_sym_list = []
        
        num_edges = 3 * self.num_vert // 2
        
        # This code is a modified copy of the Graph.__eq__ code.
        
        # Use an edge-based solution for the multigraph isomorphism problem,
        # by matching only edges in the two graphs. We start with the first
        # edge in self, and try all possible starting matches in the edge list
        # for the other graph.
        
        for (iii, other_vert_flag) in product(range(num_edges), [True, False]):
            
            edge_match_dict = {}
            
            # Given an oriented starting edge in the current graph, we find the
            # orientation of the edge in the other graph, and use the vertex
            # which has the correct 'side' of the orientation as the current
            # graph, i.e. either both arrows are out of their respective
            # vertices, or both are in.
            
            # Since edges without orientation do not give a natural way to start
            # the process, we try both possibilities for the matching vertex
            # incident to the edge in the other graph.
            
            if self.edge_list[0].orient and self.edge_list[iii].orient:
                
                if (self.edge_list[0].orient > 0 and self.edge_list[iii].orient > 0) or \
                    (self.edge_list[0].orient < 0 and self.edge_list[iii].orient < 0):
                        
                    other_vert = self.edge_list[iii].start
                else:
                    other_vert = self.edge_list[iii].end
            else:
                if other_vert_flag:
                    other_vert = self.edge_list[iii].start
                else:
                    other_vert = self.edge_list[iii].end
            
            queue = [(self.edge_list[0], self.edge_list[0].start, \
                      self.edge_list[iii], other_vert)]
                          
            while len(queue) > 0:
                
                (current_self_edge, current_self_vert, \
                 current_other_edge, current_other_vert) = queue.pop()
               
                # Test whether matching the edge, vertex combo are consistenly mapped
                # from the self graph to the other graph, if one or the other has
                # already been visited, or whether we need to continue and add all
                # other edges incident to same vertex
                
                if edge_match_dict.get(current_self_edge, None):
                    if edge_match_dict[current_self_edge] != current_other_edge:
                        break       # Inconsistent edge matching
                    elif edge_match_dict[current_self_edge] == current_other_edge:
                        continue    # Edge already visited, do not repeat
                    
                # If the edges are oriented, determine whether the orientations
                # of the two edges are consistent, i.e. both current_self_edge
                # and current_other_edge point towards current_self_vert,
                # current_other_vert, or both away. If they are inconsistent,
                # then so is mapping.
                
                if current_self_edge.orient and current_other_edge.orient:
                    
                    # Orientations are not consistent, so move on to next choice
                    
                    if abs(current_self_edge.orient) != abs(current_other_edge.orient):
                        break
                    
                    if (current_self_vert == current_self_edge.start and current_self_edge.orient < 0) \
                        or (current_self_vert == current_self_edge.end and current_self_edge.orient > 0):
                        
                        current_dir = True      # Orientation towards current_self_vert
                        
                    else:
                        current_dir = False
                        
                    if (current_other_vert == current_other_edge.start and current_other_edge.orient < 0) \
                        or (current_other_vert == current_other_edge.end and current_other_edge.orient > 0):
                            
                        other_dir = True        # Orientation towards current_other_vert
                        
                    else:
                        other_dir = False
                        
                    if current_dir != other_dir:
                        break       # Inconsistent edge orientation
                        
                # Check that edge colors are equal, if they exist
                
                if current_self_edge.color and current_other_edge.color and \
                    current_self_edge.color != current_other_edge.color:
                        break       # Inconsistent edge colors
                        
                # Either edge orientations are consistent, or no orient value
                # and a mapping between edges does not already exist.
               
                edge_match_dict[current_self_edge] = current_other_edge
                    
                # Go through and find other edges incident to current_self_vert, and
                # add them to the queue. The vertices are those on the opposite side
                # of each edge from the current vertices. Note that if we add the
                # edge, vertex combos into the queue in CCW order, they will be
                # removed via pop in CW order.
                
                self_shift = current_self_vert.edge_order.index(current_self_edge)
                other_shift = current_other_vert.edge_order.index(current_other_edge)
                
                for shift in [-2, -1]:
                    next_self_edge = current_self_vert.edge_order[self_shift + shift]
                    if current_self_vert != next_self_edge.start:
                        next_self_vert = next_self_edge.start
                    else:
                        next_self_vert = next_self_edge.end
                    
                    next_other_edge = current_other_vert.edge_order[other_shift + shift]
                    if current_other_vert != next_other_edge.start:
                        next_other_vert = next_other_edge.start
                    else:
                        next_other_vert = next_other_edge.end
                        
                    queue += [(next_self_edge, next_self_vert, next_other_edge, next_other_vert)]
  
            # We have matched all edges in the two graphs, or else queue was
            # terminated early because of an inconsistency. If the former,
            # return True
            
            if len(edge_match_dict) == num_edges == len(set(edge_match_dict.values())):
                graph_sym_list += [[self.edge_list.index(edge_match_dict[key]) \
                                    for key in self.edge_list]]
                    
        # If desired, return only list of non-isomorphic edges; otherwise, return
        # all symmetries, given in terms of edge label mappings
        
        if non_iso_edges:
            edge_union_find = UnionFind(num_edges)
            
            for sym in graph_sym_list:
                for edge_label, map_label in enumerate(sym):
                    edge_union_find.join(edge_label, map_label)
                    
            return edge_union_find.printRoots()
            
        return graph_sym_list

    #-------------------------------------------------------------------------#
        
    def addEdges(self, added_edge_list):
        """
            Add edges to graph, given as list of vertex labels [start, end, (orient), (color), (twist)]
        """
        
        # Added edge list must be (1) a list of (2) lists in the form [start,
        # end, (orient), (color), (twist)], where (3) start, end are integers from 0 to num_vert - 1, and
        # (4) start < end.
        
        if type(added_edge_list) != list:
            raise ValueError('Added edges must be given in list')
            
        if any([(type(edge) != list or not(2 <= len(edge) <= 5) \
                 or any([type(label) not in [int, type(None)] for label in edge])) for edge in added_edge_list]):
            raise ValueError('Entries in added edge list must be in form [start, end, (orient), (color), (twist)]')
            
        if any([(min(edge[:2]) < 0 or max(edge[:2]) >= self.num_vert) for edge in added_edge_list]):
            raise ValueError('Vertex labels must be between 0 and {}'.format(self.num_vert - 1))
            
        # Edges are added to vertex CCW in order given in added edge list;
        # Vertex.connectEdge tests whether too many edges added to vertex, which
        # also ensures total number of edges does not exceed 3/2 number of vertices.

        # Also, vertices are given as integers in added_edge_list, so we use
        # self.vert_list to change them into the appropriate Vertex objects.
        
        for edge in added_edge_list:
            start_vert = self.vert_list[edge[0]]
            end_vert = self.vert_list[edge[1]]
            
            if len(edge) == 2:
                new_edge = Edge(start = start_vert, end = end_vert)
            elif len(edge) == 3:
                new_edge = Edge(start = start_vert, end = end_vert, orient = edge[2])
            elif len(edge) == 4:
                new_edge = Edge(start = start_vert, end = end_vert, orient = edge[2], \
                                color = edge[3])
            else:
                new_edge = Edge(start = start_vert, end = end_vert, orient = edge[2], \
                                color = edge[3], twist = edge[4])
            
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
    
    def oneMove(self, edge_label = None):
        if edge_label != None:
            
            if type(edge_label) != int or edge_label < 0:
                raise ValueError('edge label must be positive integer')
                
            if edge_label >= 3 * self.num_vert // 2:
                raise ValueError('edge label must be between 0 and {}'.format(3 * self.num_vert // 2))
            
            current_edge = self.edge_list[edge_label]
            current_end = current_edge.end
                
            # First, we create two new vertices. The two new vertices will have
            # labels larger than any other current vertices in the graph, so we
            # put them as the end of the vertex list.
            
            self.vert_list += [Vertex(label = self.num_vert, COLOR_LIST = self.ALLOWED_COLOR_LIST), \
                               Vertex(label = self.num_vert + 1, COLOR_LIST = self.ALLOWED_COLOR_LIST)]
            self.num_vert += 2
            
            # We have the original edge ab, and we want to add vertices x, y
            # so that we can put in edges ax, bx, xy, yy (self-loop). To 
            # preserve the cyclic order for the original vertices, we need to
            # do the following:
            #
            #   (1) change ab to ax, add bx immediately after
            #   (2) add xy, yy at the end of the edge list
            #
            # Note that if we add bx immediately *before* ax, then we get a
            # self-loop on the other side of the original edge. The current
            # order puts the self-loop on the left-hand side of the edge as
            # one travels from start to end; putting it before places it on the
            # right-hand side, going in the same direction.
            
            edge_index = self.edge_list.index(current_edge)
            current_edge.end = self.vert_list[-2]
            
            edge_bx = Edge(start = current_end, end = self.vert_list[-2])
            edge_xy = Edge(start = self.vert_list[-2], end = self.vert_list[-1])
            edge_yy = Edge(start = self.vert_list[-1], end = self.vert_list[-1])
            
            self.edge_list = self.edge_list[:(edge_index + 1)] + [edge_bx] + \
                self.edge_list[(edge_index + 1):]
            self.edge_list += [edge_xy, edge_yy]
            
            current_end.replaceEdge(remove_edge = current_edge, added_edge = edge_bx)
            
            self.vert_list[-2].connectEdge(current_edge)
            self.vert_list[-2].connectEdge(edge_bx)
            self.vert_list[-2].connectEdge(edge_xy)
            
            self.vert_list[-1].connectEdge(edge_xy)
            self.vert_list[-1].connectEdge(edge_yy)
            self.vert_list[-1].connectEdge(edge_yy)
            
            # Return edges that have changed in graph edge list, so that edge
            # colors/orientations can be modified for new edges
            
            return [edge_xy, edge_yy]
            
    #-------------------------------------------------------------------------#
    
    def twoMove(self, edge_label = None):
        
        if edge_label != None:
            
            if type(edge_label) != int or edge_label < 0:
                raise ValueError('edge label must be positive integer')
                
            if edge_label >= 3 * self.num_vert // 2:
                raise ValueError('edge label must be between 0 and {}'.format(3 * self.num_vert // 2))
            
            current_edge = self.edge_list[edge_label]
            current_end = current_edge.end
                
            # First, we create two new vertices. The two new vertices will have
            # labels larger than any other current vertices in the graph, so we
            # put them as the end of the vertex list.
            
            self.vert_list += [Vertex(label = self.num_vert, COLOR_LIST = self.ALLOWED_COLOR_LIST), \
                               Vertex(label = self.num_vert + 1, COLOR_LIST = self.ALLOWED_COLOR_LIST)]
            self.num_vert += 2
            
            # We have the original edge ab, and we want to add vertices x, y
            # so that we can put in two edges xy. To preserve the cyclic
            # order, we have to change ab to ax, then add xy_1, by, xy_2
            # immediately after. This is necessary to preserve the cyclic
            # orders for a, b (edge by right after ax), as well as those for
            # x, y:
            #
            #   x: a e1 e2
            #   y: e1 b e2
            
            edge_index = self.edge_list.index(current_edge)
            current_edge.end = self.vert_list[-2]
            
            edge_by = Edge(start = current_end, end = self.vert_list[-1])
            edge_xy1 = Edge(start = self.vert_list[-2], end = self.vert_list[-1])
            edge_xy2 = Edge(start = self.vert_list[-2], end = self.vert_list[-1])
            
            self.edge_list = self.edge_list[:(edge_index + 1)] + \
                [edge_xy1, edge_by, edge_xy2] + self.edge_list[(edge_index + 1):]
            
            current_end.replaceEdge(remove_edge = current_edge, added_edge = edge_by)
            
            self.vert_list[-2].connectEdge(current_edge)
            self.vert_list[-2].connectEdge(edge_xy1)
            self.vert_list[-2].connectEdge(edge_xy2)
            
            # Since y is on the other side of the face determined by the two
            # new edges, the
            
            self.vert_list[-1].connectEdge(edge_xy1)
            self.vert_list[-1].connectEdge(edge_by)
            self.vert_list[-1].connectEdge(edge_xy2)
            
            # Return edges that have changed in graph edge list, so that edge
            # colors/orientations can be modified for new edges
            
            return [edge_by, edge_xy1, edge_xy2]
            
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
            
            self.vert_list += [Vertex(label = self.num_vert, COLOR_LIST = self.ALLOWED_COLOR_LIST), \
                               Vertex(label = self.num_vert + 1, COLOR_LIST = self.ALLOWED_COLOR_LIST)]
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
            
            for iii in [1, 2]:
                if adj_edges[iii].start != current_vert:
                    adj_edges[iii].end = None
                else:
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
                if adj_edges[1].orient != None:
                    adj_edges[1].orient = -adj_edges[1].orient
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
                if adj_edges[2].orient != None:
                    adj_edges[2].orient = -adj_edges[2].orient
            adj_edges[2].end = self.vert_list[-1]
                
            self.vert_list[-1].connectEdge(adj_edges[2])
            
            # Return edges that have changed in graph edge list, so that edge
            # colors/orientations can be modified for new edges
            
            return [edge_01, edge_02, edge_12]
            
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
            
            if current_start == current_end:
                raise ValueError('start, end vertices must be distinct')
                
            # First, we create two new vertices. The two new vertices will have
            # labels larger than any other current vertices in the graph, so we
            # put them as the end of the vertex list.
            
            self.vert_list += [Vertex(label = self.num_vert, COLOR_LIST = self.ALLOWED_COLOR_LIST), \
                               Vertex(label = self.num_vert + 1, COLOR_LIST = self.ALLOWED_COLOR_LIST)]
            self.num_vert += 2
            
            # We have the current edge with vertices x < y. The cyclic order of
            # x is of the form ax, bx, xy, while that for y is cy, dy, xy;
            # however, the order in of these edges in the graph edge list is
            # not definite, since there are three possible orderings of the
            # edges in the cyclic order for each vertex x, y.
            
            # We create two new vertices w, z, where w is self.vert_list[-2],
            # z is self.vert_list[-1]; when a, b, c, d are all distinct, we
            # go through the this process:
            #
            #   (1) identify ax, bx, then remove edge xy from cyclic order of x
            #   (2a) if a = b = x (so cyclic order is xx xx xy), change first
            #       xx -> xw, add xz xw immediately after
            #   (2b) if a != b, change edge ax to aw, add xw immediately after;
            #       add edge xz immediately after bx
            #   (3) identify cy, dy, then remove edge xy from cyclic order of y
            #   (4a) if c = d = y (so cyclic order is yy yy xy), change first
            #       yy -> yz, add yw yz immediately after
            #   (4b) if c != d, change edge cy to cz, add yz immediately after;
            #       add edge wy immediately after dy
            #
            # Since we do not move ax -> aw, cy -> cz, the cyclic orders for
            # a, c remain the same; similarly, the edges bx, dy have not been
            # changed at all, unless there is a double edge between vertices.
            # This gives the following orderings:
            #
            #   x: wbz
            #   y: zdw
            #   w: axy
            #   z: xcy
            #
            # If a = b = x (i.e. a self-loop on vertex x), then the cyclic
            # order changes as xw(1) xz zw(2), and cy dy xy -> cz yz dy wy.
            #
            #   x: w1 z w2
            #   y: zdw
            #   w: x1 x2 y
            #   z: xcy
            #
            # If b = y, c = x (i.e. double edge between x, y), then the cyclic
            # order changes ax xy(1) xy(2) and xy(1) dy xy(2), where we assume
            # the move is performed on xy(2). Then ax xy(1) xy(2) ->
            # aw xw xz(1) xz(2) and xy(1) dy xy(2) -> xz(1) yz dy wy, since (2b)
            # keep xy(1) ('bx') the same and puts xz(2) immediately after, but
            # then (4b) changes xy(1) ('cy') to xz(1) ('cz'), and puts yz
            # immediately after. Thus, three copies of xz are listed, but only
            # two edges exist, and their final order is xz(1) yz zx(2). The
            # cyclic order becomes:
            #
            #   x: w z1 z2
            #   y: zdw
            #   w: axy
            #   z: x1 y x2
            
            # (1) identify ax, bx, then remove edge xy from cyclic order of x
            
            x_index = current_start.edge_order.index(current_edge)
            
            edge_ax = current_start.edge_order[x_index - 2]
            index_ax = self.edge_list.index(edge_ax)
            
            edge_bx = current_start.edge_order[x_index - 1]     # current_edge not a self-loop
            
            current_start.removeEdge(current_edge)
            
            edge_xz = Edge(start = current_start, end = self.vert_list[-1])
            edge_xw = Edge(start = current_start, end = self.vert_list[-2])
            
            if edge_ax == edge_bx:      # current_start has self-loop
            
                # (2a) if a = b = x (so cyclic order is xx xx xy), change first
                #   xx -> xw, add xz xw immediately after
                
                edge_ax.end = self.vert_list[-2]
                current_start.removeEdge(edge_bx)
                
                self.edge_list = self.edge_list[:(index_ax + 1)] + [edge_xz, \
                    edge_xw] + self.edge_list[(index_ax + 1):]
                    
                # new_edge_list = ['2a', edge_xz, edge_xw]
                
            else:
                
                # (2b) if a != b, change edge ax to aw, add xw immediately after;
                #   add edge xz immediately after bx
                
                if edge_ax.start == current_start:
                    edge_ax.start = edge_ax.end
                    if edge_ax.color != None:
                        edge_ax.orient = -edge_ax.orient
                edge_ax.end = self.vert_list[-2]
                current_start.removeEdge(edge_ax)
                
                self.edge_list = self.edge_list[:(index_ax + 1)] + [edge_xw] + \
                    self.edge_list[(index_ax + 1):]
                    
                index_bx = self.edge_list.index(edge_bx)
                
                self.edge_list = self.edge_list[:(index_bx + 1)] + [edge_xz] + \
                    self.edge_list[(index_bx + 1):]
                    
                # new_edge_list = ['2b', edge_xz]
                    
            current_start.connectEdge(edge_xz)
            current_start.connectEdge(edge_xw)
            
            self.vert_list[-2].connectEdge(edge_ax) 
            self.vert_list[-2].connectEdge(edge_xw)
            
            self.vert_list[-1].connectEdge(edge_xz)
            
            new_edge_list = [edge_xz, edge_xw]
            
            # (3) identify cy, dy, then remove edge xy from cyclic order of y
                
            y_index = current_end.edge_order.index(current_edge)
            
            edge_cy = current_end.edge_order[y_index - 2]
            index_cy = self.edge_list.index(edge_cy)
            
            edge_dy = current_end.edge_order[y_index - 1]     # current_edge not a self-loop
            
            current_end.removeEdge(current_edge)
            
            edge_yw = Edge(start = current_end, end = self.vert_list[-2])
            edge_yz = Edge(start = current_end, end = self.vert_list[-1])
            
            if edge_cy == edge_dy:      # current_start has self-loop
            
                # (4a) if c = d = y (so cyclic order is yy yy xy), change first
                #   yy -> yz, add yw yz immediately after
                
                edge_cy.end = self.vert_list[-1]
                current_end.removeEdge(edge_cy)
                
                self.edge_list = self.edge_list[:(index_cy + 1)] + [edge_yw, \
                    edge_yz] + self.edge_list[(index_cy + 1):]
                    
                # new_edge_list[0] += '4a'
                
            else:
                
                # (4b) if c != d, change edge cy to cz, add yz immediately after;
                #   add edge yw immediately after dy
                
                if edge_cy.start == current_end:
                    edge_cy.start = edge_cy.end
                    if edge_cy.orient != None:
                        edge_cy.orient = -edge_cy.orient
                edge_cy.end = self.vert_list[-1]
                current_end.removeEdge(edge_cy)
                
                self.edge_list = self.edge_list[:(index_cy + 1)] + [edge_yz] + \
                    self.edge_list[(index_cy + 1):]
                
                index_dy = self.edge_list.index(edge_dy)
                
                self.edge_list = self.edge_list[:(index_dy + 1)] + [edge_yw] + \
                    self.edge_list[(index_dy + 1):]
                    
                # new_edge_list[0] += '4b'
                
            current_end.connectEdge(edge_yw)
            current_end.connectEdge(edge_yz)
            
            self.vert_list[-2].connectEdge(edge_yw)
            
            self.vert_list[-1].connectEdge(edge_cy)
            self.vert_list[-1].connectEdge(edge_yz)
            
            self.edge_list.remove(current_edge)
            del current_edge
            
            new_edge_list += [edge_yw, edge_yz]
            
            # Return edges that have been added to graph edge list, so that edge
            # orientations, colors can be modified for new edges; note that
            # original edge was deleted, so this returns *four* edges, even
            # though only two vertices were created.
            
            return new_edge_list
        
    #-------------------------------------------------------------------------#
    
    def fiveMove(self):
        pass
        
    #-------------------------------------------------------------------------#
    
    def findFaces(self):
        
        # Find all possible ways of traveling through each graph vertex such
        # that the incoming, outgoing edges are CCW as seen in cyclic order
            
        corner_queue = []
        for vert in self.vert_list:
            corner_queue += [(vert, vert.edge_order[0], vert.edge_order[1]), \
                             (vert, vert.edge_order[1], vert.edge_order[2]), \
                             (vert, vert.edge_order[2], vert.edge_order[0])]
                
        # Travel along all possible corners, adding to appropriate face
                
        while len(corner_queue) > 0:
            
            # Traveling CCW around face means we travel CW around vertex, so the
            # incoming edge label appears after the outgoing label in cyclic order
            
            (current_vert, next_edge, start_edge) = corner_queue.pop()
            
            # Create new face, update incident vertex and starting edge
            
            current_face = Face()
            self.face_list += [current_face]
            
            current_edge = start_edge
            
            if current_vert == start_edge.start:
                start_edge.setLeftFace(current_face)
            else:
                start_edge.setRightFace(current_face)
                
            current_vert.addFace(current_face)
            current_face.addEdge(start_edge)
            
            # Go through the rest of the face
            
            while next_edge != start_edge:
                
                # Update current vertex, edge
                
                current_edge = next_edge
            
                if current_vert == current_edge.start:
                    current_vert = current_edge.end
                    current_edge.setLeftFace(current_face)
                else:
                    current_vert = current_edge.start
                    current_edge.setRightFace(current_face)
                    
                # Add current face to vertex, and current edge to face
                    
                current_vert.addFace(current_face)
                current_face.addEdge(current_edge)
                
                # Find next edge to travel down, remove the vertex and edge
                # pair from queue
                
                current_vert_idx = current_vert.edge_order.index(current_edge)
                next_edge = current_vert.edge_order[current_vert_idx - 1]
                
                corner_queue.remove((current_vert, next_edge, current_edge))
        
    #-------------------------------------------------------------------------#
    
    def pachner22(self, edge_label = None, no_multi = True):
        
        if edge_label != None:
            
            if type(edge_label) != int or edge_label < 0:
                raise ValueError('edge label must be positive integer')
                
            if edge_label >= 3 * self.num_vert // 2:
                raise ValueError('edge label must be between 0 and {}'.format(3 * self.num_vert // 2))
            
            current_edge = self.edge_list[edge_label]
            current_start = current_edge.start
            current_end = current_edge.end
            
            if current_start == current_end:
                raise ValueError('start, end vertices must be distinct')
                
            # If faces have not been created, do so here
            
            if self.face_list == []:
                self.findFaces()
            
            # This is similar to fourMove() defined earlier, except no new
            # vertices are created; instead, existing edges have their start,
            # end attributes changed around. We also remove the orientation and
            # color (if they exists) of current_edge, and return these at end,
            # so they can be modified later.
            
            # We have the current edge with vertices x < y. The starting cyclic
            # order of x is of the form ax bx xy, while that for y is cy dy xy. 
            # We want to change it so that the final cyclic order of x is
            # bx cx xy, while that of y is dy ay xy.
            
            start_current_edge_index = current_start.edge_order.index(current_edge)
            end_current_edge_index = current_end.edge_order.index(current_edge)
            
            if no_multi:
                
                # If no_multi is True (i.e. no multiple edges allowed between
                # vertices), we must check that the two faces not shared by
                # x, y do not have an edge in common
                
                union_face_list = list(set(current_start.face_list).union(current_end.face_list))
                intersection_face_list = list(set(current_start.face_list).intersection(current_end.face_list))
                
                unshared_face_list = [face for face in union_face_list \
                                      if face not in intersection_face_list]

                if len(unshared_face_list) < 2 or len(list(set(unshared_face_list[0].edge_list).intersection(unshared_face_list[1].edge_list))) > 0:
                    raise AttributeError('Pachner 2-2 move gives graph not dual to triangulation')
                    
            # (0) remove orientation, color from xy, if necessary; in_arrow and
            # color_list for start, end vertices are reconstructed later
            
            if current_edge.orient:
                current_edge.orient = None
                
            if current_edge.color:
                current_edge.color = None
        
            # (1) Change ax to ay
            
            edge_ax = current_start.edge_order[start_current_edge_index - 2]
            
            if edge_ax.start == current_start:
                edge_ax.start = current_end
            else:
                edge_ax.end = current_end
                
            if edge_ax.start.label > edge_ax.end.label:
                edge_ax.start, edge_ax.end = edge_ax.end, edge_ax.start
                if edge_ax.orient:
                    edge_ax.orient = -edge_ax.orient
                
            # (2) Change cy to cx
            
            edge_cy = current_end.edge_order[end_current_edge_index - 2]
                
            if edge_cy.start == current_end:
                edge_cy.start = current_start
            else:
                edge_cy.end = current_start
                
            if edge_cy.start.label > edge_cy.end.label:
                edge_cy.start, edge_cy.end = edge_cy.end, edge_cy.start
                if edge_cy.orient:
                    edge_cy.orient = -edge_cy.orient
                
            # (3) Change cyclic order for vertex x to be bx cx xy, for vertex
            # y to be dy ay xy. While this is done, ensure that in_arrow is
            # correct for x, y.
            
            edge_bx = current_start.edge_order[start_current_edge_index - 1]
            edge_dy = current_end.edge_order[end_current_edge_index - 1]
            
            temp_color_list = [edge.color for edge in [edge_ax, edge_bx, edge_cy, edge_dy]]
            current_start.color_list = [temp_color_list[iii] for iii in [1, 2]] + [0]
            current_end.color_list = [temp_color_list[iii] for iii in [3, 0]] + [0]
            
            current_start.edge_order = []
            current_start.in_arrow = []
            
            for edge in [edge_bx, edge_cy, current_edge]:
                current_start.edge_order += [edge]
                
                if edge.orient:
                    if (edge.orient > 0 and current_start == edge.end) or \
                        (edge.orient < 0 and current_start == edge.start):
                        current_start.in_arrow += [1]
                    else:
                        current_start.in_arrow += [-1]
                else:
                    current_start.in_arrow += [0]
                    
            current_end.edge_order = []
            current_end.in_arrow = []
                    
            for edge in [edge_dy, edge_ax, current_edge]:
                current_end.edge_order += [edge]
                if edge.orient:
                    if (edge.orient > 0 and current_end == edge.end) or \
                        (edge.orient < 0 and current_end == edge.start):
                        current_end.in_arrow += [1]
                    else:
                        current_end.in_arrow += [-1]
                else:
                    current_end.in_arrow += [0]
            
            # (4) Change edge list for graph; to be consistent with vertex
            # cyclic orders, we need to permute the order of the edges in the
            # edge list so that
            #
            #   ax bx xy -> bx cx xy
            #   cy dy xy -> dy ay xy
            #
            # However, we also have to worry about how the ordering of the
            # vertices a, b, c, d are changed by this permutation. So to keep
            # the cyclic orders for these vertices the same, we only move the
            # edge xy in the graph edge list; this edge is placed in the
            # earliest possible spot in the graph edge list, because why not.
            #
            # We also remove current_edge from the edge_list, to avoid
            # shifting the indices of the other edges by 1.
            
            self.edge_list.remove(current_edge)
            edge_list = [edge_ax, edge_bx, edge_cy, edge_dy]
            index_list = [self.edge_list.index(edge) for edge in edge_list]
            
            # for iii in range(4):
            #     self.edge_list[index_list[iii - 1]] = edge_list[iii]
        
            if index_list[2] < index_list[1]:           # cy comes before bx
                slot_list = [iii for iii in range(3 * self.num_vert // 2) \
                              if index_list[2] < iii <= index_list[1]]
            else:                                       # bx comes before cy
                slot_list = [iii for iii in range(3 * self.num_vert // 2) \
                              if (iii <= index_list[1] or index_list[2] < iii)]
                
            if index_list[0] < index_list[3]:           # ax comes before dy
                slot_list = [label for label in slot_list \
                              if index_list[0] < label <= index_list[3]]
            else:                                       # dy comes before ax
                slot_list = [label for label in slot_list \
                              if (label <= index_list[3] or index_list[0] < label)]
            
            slot = slot_list[0]
            self.edge_list = self.edge_list[:slot] + [current_edge] + self.edge_list[slot:]
                
            return current_edge
        
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
    
    def allowedColorList(self):
        return self.ALLOWED_COLOR_LIST
        
    #-------------------------------------------------------------------------#
    
    def colorList(self):
        return [edge.color for edge in self.edge_list]
    
    #-------------------------------------------------------------------------#
    
    def edgeList(self):
        return [edge for edge in self.edge_list]
    
    #-------------------------------------------------------------------------#
    
    def orientList(self):
        return [edge.orient for edge in self.edge_list]
    
    #-------------------------------------------------------------------------#