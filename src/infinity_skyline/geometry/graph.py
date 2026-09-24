# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP - Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   IC - Institute of Computing            ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀   Computer Science Department              ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · IC · 2026                    ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝

"""The scene as a graph: images are nodes, verified pairs are edges.

Two questions are answered here. Which images belong to the same scene at all
(an intruder from another shoot simply never connects, so it falls outside the
largest connected component), and in which order the ones that do belong were
swept, since a panorama is a chain and the chain is recovered by spectral
seriation of the weighted graph.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import networkx as nx
import numpy as np

from ..config import GraphConfig
from ..io import MatchRecord
from ..logging_setup import get_logger

logger = get_logger(__name__)


def build_graph(
    matches: Mapping[tuple[str, str], MatchRecord],
    names: Sequence[str],
    config: GraphConfig,
) -> nx.Graph:
    """Graph of the verified pairs, with every image present as a node.

    Images with no surviving pair stay in the graph as isolated nodes: that is
    exactly how the intruder shows up, and dropping it would hide the finding.
    """
    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from(names)

    for record in matches.values():
        if record.homography is None:
            continue
        if record.n_inliers < config.min_inliers:
            continue
        if record.inlier_ratio < config.min_inlier_ratio:
            continue
        graph.add_edge(
            record.query,
            record.train,
            weight=record.n_inliers,
            inlier_ratio=record.inlier_ratio,
            rmse=float(record.meta.get("reprojection_rmse", float("nan"))),
            homography=record.homography,
            # nx.Graph does not keep the pair order, and the stored homography
            # maps query onto train, so compose needs to know which is which to
            # decide whether it has to invert the matrix.
            query=record.query,
            train=record.train,
        )

    logger.info(
        "graph: %d nodes, %d edges (min_inliers=%d, min_inlier_ratio=%.2f)",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        config.min_inliers,
        config.min_inlier_ratio,
    )
    return graph


def connectivity_matrix(graph: nx.Graph, names: Sequence[str]) -> np.ndarray:
    """``(N, N)`` inlier counts in ``names`` order; 0 where there is no edge."""
    nodes = list(names)
    matrix = np.zeros((len(nodes), len(nodes)), dtype=np.int64)
    index = {name: i for i, name in enumerate(nodes)}
    for query, train, weight in graph.edges(data="weight", default=0):
        if query in index and train in index:
            matrix[index[query], index[train]] = int(weight)
            matrix[index[train], index[query]] = int(weight)
    return matrix


def largest_component(graph: nx.Graph) -> list[str]:
    """Nodes of the biggest connected component, in graph insertion order."""
    if graph.number_of_nodes() == 0:
        return []
    component = max(nx.connected_components(graph), key=len)
    return [node for node in graph.nodes if node in component]


def rejected_images(graph: nx.Graph, config: GraphConfig) -> list[str]:
    """Nodes outside the largest component: images that are not in this scene."""
    if not config.reject_isolated:
        return []
    keep = set(largest_component(graph))
    rejected = [node for node in graph.nodes if node not in keep]
    if rejected:
        logger.info("rejected %d image(s) outside the main component: %s", len(rejected), rejected)
    return rejected


def _spectral_order(sub: nx.Graph, nodes: list[str]) -> list[str]:
    """Sort one connected component by its Fiedler vector."""
    if len(nodes) <= 2:
        return list(nodes)
    adjacency = np.array(
        [
            [float(sub[u][v].get("weight", 1.0)) if sub.has_edge(u, v) else 0.0 for v in nodes]
            for u in nodes
        ],
        dtype=float,
    )
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    # eigh, not eig: the Laplacian is symmetric, so the spectrum is real.
    values, vectors = np.linalg.eigh(laplacian)
    fiedler = vectors[:, np.argsort(values)[1]]
    return [nodes[i] for i in np.argsort(fiedler)]


def infer_order(graph: nx.Graph, nodes: Sequence[str]) -> list[str]:
    """Order ``nodes`` along the sweep by spectral seriation (the Fiedler vector).

    The diameter path used before dropped frames: a shortcut edge between non
    neighbours (0547--0551, 80 inliers) short circuits the shortest path, so
    0548 and 0555 fell off the chain and a heuristic reinserted them in the
    wrong slot. The Fiedler vector weighs every edge at once, keeping all nodes.
    """
    sub = graph.subgraph(nodes)
    if sub.number_of_nodes() == 0:
        return []
    if sub.number_of_nodes() <= 2:
        return list(nodes)

    # A disconnected subgraph has eigenvalue 0 with multiplicity > 1, so its
    # "second smallest" says nothing: order each component on its own instead.
    components = sorted(nx.connected_components(sub), key=len, reverse=True)
    order: list[str] = []
    for component in components:
        order.extend(_spectral_order(sub, [n for n in nodes if n in component]))
    return order
