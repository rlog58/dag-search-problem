from typing import Iterator, Sequence

import networkx as nx
import numpy as np


def to_binary(n: int, base: int = None, pad: str = '0'):
    if base is None:
        base = ''
        pad = ''
    else:
        base = str(base)

    format_expr = '{0:' + pad + base + 'b}'

    return np.array([int(i) for i in format_expr.format(n)], dtype=np.uint8)


def is_one_target_digraph(digraph: nx.DiGraph) -> bool:
    has_target = False
    for _, out_degree in digraph.out_degree:
        if out_degree == 0:
            if has_target:
                return False
            has_target = True

    return has_target


def is_one_source_digraph(digraph: nx.DiGraph) -> bool:
    has_source = False
    for _, in_degree in digraph.in_degree:
        if in_degree == 0:
            if has_source:
                return False
            has_source = True

    return has_source


def filter_non_isomorphic_digraphs(digraphs: Sequence[nx.DiGraph]) -> Iterator[nx.DiGraph]:
    n = len(digraphs)

    if n < 2:
        return (_ for _ in digraphs)

    isomorphic_mask = np.zeros(n, dtype=np.bool_)

    for i in range(n - 1):
        if not isomorphic_mask[i]:
            for j in range(i + 1, n):
                if not isomorphic_mask[j]:
                    isomorphic_mask[j] = nx.is_isomorphic(digraphs[i], digraphs[j])
            yield digraphs[i]

    if not isomorphic_mask[n - 1]:
        yield digraphs[n - 1]
