"""The scene as a graph: images are nodes, verified pairs are edges.

Two questions are answered here. Which images belong to the same scene at all
(an intruder from another shoot simply never connects, so it falls outside the
largest connected component), and in which order the ones that do belong were
swept, since a panorama is a chain and the chain is the graph diameter path.
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


def infer_order(graph: nx.Graph, nodes: Sequence[str]) -> list[str]:
    """Order ``nodes`` along the sweep, using the diameter path of their subgraph."""
    sub = graph.subgraph(nodes)
    if sub.number_of_nodes() == 0:
        return []
    if sub.number_of_nodes() <= 2:
        return list(nodes)

    # Diameter path: the longest of all shortest paths. all_pairs works on a
    # disconnected subgraph too, it just never leaves each component.
    lengths = dict(nx.all_pairs_shortest_path_length(sub))
    source, target = max(
        ((u, v) for u, reachable in lengths.items() for v in reachable),
        key=lambda pair: lengths[pair[0]][pair[1]],
    )
    order: list[str] = list(nx.shortest_path(sub, source, target))

    # Nodes off the main chain (a branch, or a second component) are slotted in
    # next to whichever already placed neighbour they match most strongly.
    for node in nodes:
        if node in order:
            continue
        placed = [(sub[node][other].get("weight", 0), other) for other in sub[node] if other in order]
        if placed:
            anchor = max(placed)[1]
            order.insert(order.index(anchor) + 1, node)
        else:
            order.append(node)
    return order
