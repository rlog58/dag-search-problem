from abc import ABC, abstractmethod
from typing import Any

import networkx as nx
import numpy as np


class DAGEncoding(ABC):
    @abstractmethod
    def to_adjacency_matrix(self, encoded: Any) -> np.ndarray:
        pass

    @abstractmethod
    def to_digraph(self, encoded: Any) -> nx.DiGraph:
        pass
