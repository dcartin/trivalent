# trivalent documentation

`trivalent` is a Python package for studying
embedded trivalent graphs and their transformations.

```python
from trivalent import Graph, decode_orient_state, encode_orient_state

# Create |V| = 8 standard graph (graph ID #0) with implicitly ordered edge list

graphNum = 0
edgeList = [[0, 1], [0, 6], [0, 7], [4, 6], [6, 7], [5, 7], \
            [1, 2], [3, 4], [4, 5], [2, 5], [1, 3], [2, 3]]
G = Graph(edgeList)

# Find graph properties

print(G.num_vert, G.num_edges, G.num_faces)

# Positive graph automorphisms

symGrp = G.find_sym()

# Encode, canonicalize an orientation bitmask

orientBits = 1961
state = encode_orient_state(graphNum, orientBits, G.num_edges)
canonical = G.min_orient_state(state, symGrp)

# Decode canonical state into graph ID, orientation bitmask

print(decode_orient_state(canonical, G.num_edges))
```

## Documentation

- [Graph API](reference/graph.md)
- [States API](reference/states.md)