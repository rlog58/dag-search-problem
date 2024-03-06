from typing import Any

import networkx as nx
import numpy as np

from main.encoding.base import DAGEncoding


class CellularEncoding(DAGEncoding):
    def to_digraph(self, encoded: Any) -> nx.DiGraph:
        pass

    def to_adjacency_matrix(self, encoded: Any) -> np.ndarray:
        pass
