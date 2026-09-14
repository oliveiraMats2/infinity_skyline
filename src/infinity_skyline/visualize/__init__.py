"""Figure rendering: keypoints, matches, graph, panorama and benchmark plots."""

from __future__ import annotations

from .graph import draw_graph
from .keypoints import draw_keypoints
from .matches import draw_before_after, draw_matches
from .panorama import draw_deghosting_comparison, draw_progressive
from .report import plot_benchmark

__all__ = [
    "draw_graph",
    "draw_keypoints",
    "draw_before_after",
    "draw_matches",
    "draw_deghosting_comparison",
    "draw_progressive",
    "plot_benchmark",
]
