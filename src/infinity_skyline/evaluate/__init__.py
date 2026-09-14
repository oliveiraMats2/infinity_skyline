"""Benchmark orchestration and the pure metric functions it reports."""

from __future__ import annotations

from .benchmark import run_benchmark, summarize
from .metrics import (
    distortion_score,
    inlier_ratio,
    reprojection_rmse,
    seam_line_continuity,
    spatial_dispersion,
)

__all__ = [
    "run_benchmark",
    "summarize",
    "distortion_score",
    "inlier_ratio",
    "reprojection_rmse",
    "seam_line_continuity",
    "spatial_dispersion",
]
