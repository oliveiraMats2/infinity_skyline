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

"""Connectivity graph: edge thresholds, the intruder, and the sweep order."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from infinity_skyline.config import GraphConfig
from infinity_skyline.geometry.graph import (
    build_graph,
    connectivity_matrix,
    infer_order,
    largest_component,
    rejected_images,
)
from infinity_skyline.io import MatchRecord

CHAIN: list[str] = ["img_0", "img_1", "img_2", "img_3", "img_4"]
INTRUDER: str = "intruder"
NAMES: list[str] = CHAIN + [INTRUDER]


def _record(query: str, train: str, n_matches: int, n_inliers: int) -> MatchRecord:
    mask = np.zeros(n_matches, dtype=bool)
    mask[:n_inliers] = True
    return MatchRecord(
        query=query,
        train=train,
        pairs=np.zeros((n_matches, 2), dtype=np.int32),
        distances=np.zeros(n_matches, dtype=np.float32),
        n_raw=n_matches * 2,
        inlier_mask=mask,
        homography=np.eye(3, dtype=np.float64),
        meta={"reprojection_rmse": 0.8, "success": True},
    )


def _chain_matches() -> dict[tuple[str, str], MatchRecord]:
    matches: dict[tuple[str, str], MatchRecord] = {}
    for left, right in zip(CHAIN, CHAIN[1:]):
        matches[(left, right)] = _record(left, right, 200, 180)  # ratio 0.90
    for name in CHAIN:
        matches[(name, INTRUDER)] = _record(name, INTRUDER, 40, 5)  # 5 inliers, ratio 0.125
    return matches


def _config(**overrides: object) -> GraphConfig:
    return GraphConfig(min_inliers=30, min_inlier_ratio=0.25, **overrides)  # type: ignore[arg-type]


def test_build_graph_keeps_the_chain_and_isolates_the_intruder() -> None:
    graph = build_graph(_chain_matches(), NAMES, _config())

    assert set(graph.nodes) == set(NAMES)
    assert set(map(frozenset, graph.edges)) == {
        frozenset((a, b)) for a, b in zip(CHAIN, CHAIN[1:])
    }
    assert graph.degree(INTRUDER) == 0
    edge = graph.edges["img_0", "img_1"]
    assert edge["weight"] == 180
    assert edge["inlier_ratio"] == pytest.approx(0.9)


def test_largest_component_and_rejection_split_the_scene() -> None:
    graph = build_graph(_chain_matches(), NAMES, _config())

    assert sorted(largest_component(graph)) == CHAIN
    assert rejected_images(graph, _config()) == [INTRUDER]


def test_rejection_is_opt_out() -> None:
    graph = build_graph(_chain_matches(), NAMES, _config(reject_isolated=False))
    assert rejected_images(graph, _config(reject_isolated=False)) == []


def test_infer_order_walks_the_chain() -> None:
    graph = build_graph(_chain_matches(), NAMES, _config())
    order = infer_order(graph, largest_component(graph))

    # the diameter path has no canonical direction, so either end may come first
    assert order in (CHAIN, list(reversed(CHAIN)))


def test_relative_and_absolute_criteria_are_both_enforced() -> None:
    names = ["p", "q", "r", "s"]
    matches = {
        # plenty of raw matches, but only a tenth of them agree geometrically
        ("p", "q"): _record("p", "q", 1000, 100),
        # perfectly consistent, yet far too few points to trust the fit
        ("r", "s"): _record("r", "s", 20, 20),
    }
    graph = build_graph(matches, names, _config())

    assert graph.number_of_edges() == 0
    assert len(largest_component(graph)) == 1  # nothing survived either gate


def test_connectivity_matrix_is_symmetric_with_a_zero_diagonal() -> None:
    graph = build_graph(_chain_matches(), NAMES, _config())
    matrix = connectivity_matrix(graph, NAMES)

    assert matrix.shape == (len(NAMES), len(NAMES))
    np.testing.assert_array_equal(matrix, matrix.T)
    np.testing.assert_array_equal(np.diag(matrix), np.zeros(len(NAMES), dtype=matrix.dtype))
    assert matrix[0, 1] > 0  # consecutive images are connected
    assert matrix[0, 2] == 0  # non consecutive ones were never matched
    assert matrix[len(CHAIN), :].sum() == 0  # the intruder is connected to nothing


def test_a_sweep_that_closes_on_itself_is_ordered_around_the_ring() -> None:
    # 12 frames on a circle, each overlapping the next two, and the last the first:
    # the Fiedler vector alone folds this in half and interleaves the two sides.
    names = [f"f{i:02d}" for i in range(12)]
    graph = nx.Graph()
    for i in range(12):
        graph.add_edge(names[i], names[(i + 1) % 12], weight=500)
        graph.add_edge(names[i], names[(i + 2) % 12], weight=200)
    order = infer_order(graph, names)
    steps = {(names.index(b) - names.index(a)) % 12 for a, b in zip(order, order[1:])}
    assert steps in ({1}, {11})  # one direction around the ring, no jumps
