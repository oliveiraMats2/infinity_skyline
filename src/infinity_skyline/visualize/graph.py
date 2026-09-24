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

"""Stage 5.3: the connectivity graph, with the rejected images flagged."""

from __future__ import annotations

from math import ceil, sqrt
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend, must be set before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from ..config import VisualizeConfig  # noqa: E402
from ..geometry.graph import connectivity_matrix  # noqa: E402
from ..logging_setup import get_logger  # noqa: E402

logger = get_logger(__name__)

ACCEPTED_COLOR: str = "#1f77b4"
REJECTED_COLOR: str = "#d62728"
_LAYOUT_SEED: int = 42
_MIN_EDGE_WIDTH: float = 0.8
_MAX_EDGE_WIDTH: float = 6.0
_CELL: float = 1.6  # spacing between packed components


def _layout_component(graph: nx.Graph, name: str) -> dict[Any, Any]:
    """Layout of a single connected component, normalized to a unit box."""
    if graph.number_of_nodes() == 1:
        return {next(iter(graph.nodes)): np.zeros(2)}
    if name == "circular":
        positions = nx.circular_layout(graph)
    elif name == "kamada_kawai":
        positions = nx.kamada_kawai_layout(graph)
    else:
        positions = nx.spring_layout(graph, seed=_LAYOUT_SEED, iterations=200)

    coords = np.asarray(list(positions.values()), dtype=float)
    span = coords.max(axis=0) - coords.min(axis=0)
    span[span < 1e-9] = 1.0
    center = (coords.max(axis=0) + coords.min(axis=0)) / 2.0
    return {node: (np.asarray(xy, dtype=float) - center) / span for node, xy in positions.items()}


def _layout(graph: nx.Graph, name: str) -> dict[Any, Any]:
    """Lay each connected component out on its own, then pack them into a grid.

    Laying out the whole graph at once collapses every component to a point as soon
    as a rejected intruder splits it off, because the layout is normalized globally.
    """
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    columns = max(1, ceil(sqrt(len(components))))
    positions: dict[Any, Any] = {}
    for index, nodes in enumerate(components):
        placed = _layout_component(graph.subgraph(nodes), name)
        offset = np.array([(index % columns) * _CELL, -(index // columns) * _CELL])
        for node, xy in placed.items():
            positions[node] = xy + offset
    return positions


def _short(name: str) -> str:
    """Drop a shared prefix like 'IMG_' so the label fits inside the node."""
    return name[4:] if name.startswith("IMG_") else name


def draw_graph(
    graph: nx.Graph,
    rejected: Sequence[str],
    config: VisualizeConfig,
    out_path: Path,
) -> None:
    """Render the graph to ``out_path``: intruders in red, edge width by inliers."""
    rejected_set = set(rejected)
    nodes = list(graph.nodes)
    positions = _layout(graph, config.graph.layout)

    weights = [float(data.get("weight", 0.0)) for _, _, data in graph.edges(data=True)]
    top = max(weights) if weights else 1.0
    widths = [
        _MIN_EDGE_WIDTH + (_MAX_EDGE_WIDTH - _MIN_EDGE_WIDTH) * (w / top) for w in weights
    ]

    figure, axes = plt.subplots(figsize=(11, 8))
    nx.draw_networkx_edges(graph, positions, width=widths, alpha=0.55, ax=axes)
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=nodes,
        node_size=1100,
        node_color=[REJECTED_COLOR if n in rejected_set else ACCEPTED_COLOR for n in nodes],
        edgecolors=["black" if n in rejected_set else "white" for n in nodes],
        linewidths=[3.0 if n in rejected_set else 0.8 for n in nodes],
        ax=axes,
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        labels={n: _short(str(n)) for n in nodes},
        font_size=8,
        font_color="white",
        font_weight="bold",
        ax=axes,
    )
    if config.graph.annotate_weights:
        nx.draw_networkx_edge_labels(
            graph,
            positions,
            edge_labels={
                (u, v): str(int(data.get("weight", 0)))
                for u, v, data in graph.edges(data=True)
            },
            font_size=7,
            ax=axes,
        )

    axes.set_title(
        f"Grafo de conectividade ({len(nodes)} imagens, {len(weights)} arestas, "
        f"{len(rejected_set)} rejeitada(s))"
    )
    axes.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", markersize=11,
                   markerfacecolor=ACCEPTED_COLOR, markeredgecolor="white",
                   label="aceita (componente principal)"),
            Line2D([], [], marker="o", linestyle="", markersize=11,
                   markerfacecolor=REJECTED_COLOR, markeredgecolor="black",
                   label="rejeitada (fora da componente principal)"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    axes.margins(0.12)
    axes.set_axis_off()
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=config.figures.dpi)
    plt.close(figure)  # closing matters: this runs inside loops
    logger.info("graph figure saved to %s (%d rejected)", out_path, len(rejected_set))


def save_graph_figures(
    graph: nx.Graph,
    rejected: Sequence[str],
    config: VisualizeConfig,
    run_dir: Path,
) -> list[Path]:
    """Every graph figure of one run, returned in the order they were written.

    The node-edge drawing is the readable one only while the scene is small; past
    roughly 20 images it degenerates into a hairball, so the same information is
    also written as a connectivity matrix, which stays legible at any size.
    """
    rejected_set = set(rejected)
    names = sorted(graph.nodes)
    # an asterisk is the whole legend: the rejected images are what the reader looks for
    labels = [f"{_short(str(n))}*" if n in rejected_set else _short(str(n)) for n in names]

    graph_path = config.figures.path(run_dir, "graph", "graph")
    draw_graph(graph, rejected, config, graph_path)
    paths = [graph_path]

    matrix = connectivity_matrix(graph, names)
    figure, axes = plt.subplots(figsize=(9, 8))
    heat = axes.imshow(matrix, cmap="viridis")
    figure.colorbar(heat, ax=axes, label="inliers")
    axes.set_xticks(range(len(names)))
    axes.set_xticklabels(labels, rotation=90, fontsize=7)
    axes.set_yticks(range(len(names)))
    axes.set_yticklabels(labels, fontsize=7)
    if len(names) <= 20:
        # the per cell number only fits while the matrix is small; above that the
        # colour alone carries the value and the text would overlap into noise
        half = matrix.max() / 2.0
        for i in range(len(names)):
            for j in range(len(names)):
                axes.text(
                    j, i, str(matrix[i, j]), ha="center", va="center", fontsize=6,
                    color="white" if matrix[i, j] < half else "black",
                )
    axes.set_title("Matriz de conectividade (* = rejeitada)")
    figure.tight_layout()
    matrix_path = config.figures.path(run_dir, "graph", "connectivity")
    figure.savefig(matrix_path, dpi=config.figures.dpi)
    plt.close(figure)
    paths.append(matrix_path)

    weights = [float(data.get("weight", 0.0)) for _, _, data in graph.edges(data=True)]
    if not weights:
        # a graph of isolated nodes is a legitimate outcome (every pair rejected),
        # so the missing histogram is a warning and not a failure of the run
        logger.warning("graph has no edges, skipping the edge weight histogram")
        return paths

    figure, axes = plt.subplots(figsize=(8, 5))
    axes.hist(weights, bins=min(20, len(set(weights))), color=ACCEPTED_COLOR, edgecolor="white")
    axes.set_xlabel("Inliers por aresta")
    axes.set_ylabel("Numero de arestas")
    axes.set_title(f"Distribuicao do peso das arestas ({len(weights)} arestas)")
    figure.tight_layout()
    weights_path = config.figures.path(run_dir, "graph", "edge_weights")
    figure.savefig(weights_path, dpi=config.figures.dpi)
    plt.close(figure)
    paths.append(weights_path)

    logger.info("saved %d graph figures to %s", len(paths), graph_path.parent)
    return paths
