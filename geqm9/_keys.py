"""Keys file to overcome TorchScript constants bug."""

import sys

if sys.version_info[1] >= 8:
    from typing import Final
else:
    from typing_extensions import Final

# DataDict keys

# [n_graphs, dim] (possibly equivariant) output feature of graph
GRAPH_OUTPUT_KEY: Final[str] = "graph_output"