from typing import Sequence, Iterator

import networkx as nx
import numpy as np

from main.encoding.base import DAGEncoding


class AdjacencyEncoding(DAGEncoding):
    def __init__(self, dim: int):
        self._dim = dim
        self._size = (dim ** 2 - dim) // 2

    @property
    def dim(self):
        return self._dim

    @property
    def size(self):
        return self._size

    def _from_adjacency_matrix_index(self, i: int, j: int) -> int:
        return (2 * self.dim * i - 3 * i - i * i + 2 * j - 2) // 2

    def to_adjacency_matrix(self, encoded: Sequence[int]) -> np.ndarray:
        assert len(encoded) == self.size, \
            "Sequence length does not match encoding size. " \
            f"Expected: {self._size}, got: {len(encoded)}"

        matrix = np.zeros((self.dim, self.dim), np.uint8)
        for i in range(self.dim - 1):
            for j in range(i + 1, self.dim):
                matrix[i][j] = encoded[self._from_adjacency_matrix_index(i, j)]

        return matrix

    def from_adjacency_matrix(self, matrix: np.ndarray, check_order: bool = True) -> Iterator[int]:
        assert matrix.shape == (self.dim, self.dim), \
            "Matrix shape does not match the encoding dimension. " \
            f"Expected: {(self.dim, self.dim)}, got: {matrix.shape}"
        if check_order:
            for i in range(1, self.dim):
                for j in range(i):
                    if matrix[i][j] > 0:
                        raise ValueError(
                            "Matrix edges expected to be ordered ascendingly, "
                            f"but edge {i} -> {j} was found."
                        )

        for i in range(self.dim - 1):
            for j in range(i + 1, self.dim):
                yield matrix[i][j]

    def to_digraph(self, encoded: Sequence[int]) -> nx.DiGraph:
        assert len(encoded) == self.size, \
            "Sequence length does not match encoding size. " \
            f"Expected: {self.size}, got: {len(encoded)}"
        return nx.from_numpy_array(self.to_adjacency_matrix(encoded), create_using=nx.DiGraph)

    def from_digraph(self, digraph: nx.DiGraph, check_order: bool = True) -> Iterator[int]:
        assert digraph.number_of_nodes() == self.dim, \
            f"The number of vertexes does not match the encoding dimension. " \
            f"Expected: {self.dim}, got: {digraph.number_of_nodes()}"
        if check_order:
            for edge in digraph.edges:
                if edge[0] > edge[1]:
                    raise ValueError(
                        "Matrix edges expected to be ordered ascendingly, "
                        f"but edge {edge[0]} -> {edge[1]} was found."
                    )

        matrix = nx.to_numpy_array(digraph, dtype=np.uint8)
        return self.from_adjacency_matrix(matrix, check_order=False)

    def to_str(self, encoded: Sequence[int]) -> str:
        assert len(encoded) == self.size, \
            "Sequence length does not match encoding size. " \
            f"Expected: {self.size}, got: {len(encoded)}"
        return ''.join(str(elem) for elem in encoded)

    def from_str(self, encoded_string: str) -> Iterator[int]:
        assert len(encoded_string) == self.size, \
            "String length does not match encoding size. " \
            f"Expected: {self.size}, got: {len(encoded_string)}"
        return (int(char) for char in encoded_string)


# TODO: remove after introducing tests
if __name__ == "__main__":
    # create encoding based on up to 4 vertexes DAG
    encoding = AdjacencyEncoding(4)
    print("Encoding size:", encoding.size)

    encoded_dag_str = '100110'
    encoded_dag = list(encoding.from_str(encoded_dag_str))

    print("DAG as vector (upper-triangular matrix values):\n", encoded_dag)
    encoded_dag_matrix = encoding.to_adjacency_matrix(encoded_dag)

    print("DAG adjacency matrix:\n", encoded_dag_matrix)

    digraph = encoding.to_digraph(encoded_dag)

    print("DAG as nx.DiGraph (__str__):\n", digraph)
