"""Benchmark figures built from the long format metrics frame."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend, must be set before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ..config import FiguresConfig  # noqa: E402
from ..logging_setup import get_logger  # noqa: E402

logger = get_logger(__name__)


def _has(frame: pd.DataFrame, columns: Sequence[str], figure_name: str) -> bool:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        logger.warning("skipping %s: missing columns %s", figure_name, missing)
        return False
    return True


def _save(
    figure: plt.Figure,
    figures: FiguresConfig,
    run_dir: Path,
    name: str,
    paths: list[Path],
) -> None:
    path = figures.path(run_dir, "report", name)
    figure.tight_layout()
    figure.savefig(path, dpi=figures.dpi)
    plt.close(figure)
    paths.append(path)


def plot_benchmark(frame: pd.DataFrame, figures: FiguresConfig, run_dir: Path) -> list[Path]:
    """Write the benchmark charts under ``run_dir`` and return the paths written."""
    paths: list[Path] = []

    if _has(frame, ("detector", "detect_time_ms"), "detect_time"):
        grouped = frame.groupby("detector")
        errors = (
            grouped["detect_time_ms"].std().fillna(0.0)
            if "detect_time_std_ms" not in frame.columns
            else grouped["detect_time_std_ms"].mean()
        )
        figure, axes = plt.subplots(figsize=(7, 5))
        grouped["detect_time_ms"].mean().plot(
            kind="bar", yerr=errors, capsize=4, ax=axes, color="#4c72b0", rot=0
        )
        axes.set_title("Tempo medio de deteccao por detector")
        axes.set_ylabel("ms")
        axes.set_xlabel("detector")
        _save(figure, figures, run_dir, "detect_time", paths)

    if _has(frame, ("detector", "matcher", "match_time_ms"), "match_time"):
        pivot = frame.pivot_table(
            index="detector", columns="matcher", values="match_time_ms", aggfunc="mean"
        )
        figure, axes = plt.subplots(figsize=(8, 5))
        pivot.plot(kind="bar", ax=axes, rot=0)
        axes.set_title("Tempo medio de matching por detector e matcher")
        axes.set_ylabel("ms")
        axes.set_xlabel("detector")
        axes.legend(title="matcher")
        _save(figure, figures, run_dir, "match_time", paths)

    if _has(frame, ("detector", "n_keypoints", "descriptor_bytes"), "keypoints_cost"):
        grouped = frame.groupby("detector")
        figure, (left, right) = plt.subplots(1, 2, figsize=(11, 5))
        grouped["n_keypoints"].mean().plot(kind="bar", ax=left, color="#55a868", rot=0)
        left.set_title("Keypoints por imagem")
        left.set_ylabel("quantidade")
        grouped["descriptor_bytes"].mean().plot(kind="bar", ax=right, color="#c44e52", rot=0)
        right.set_title("Bytes por descritor")
        right.set_ylabel("bytes")
        _save(figure, figures, run_dir, "keypoints_cost", paths)

    if _has(frame, ("detector", "matcher", "inlier_ratio"), "inlier_ratio_box"):
        groups = list(frame.groupby(["detector", "matcher"])["inlier_ratio"])
        figure, axes = plt.subplots(figsize=(9, 5))
        axes.boxplot([series.to_numpy() for _, series in groups])
        axes.set_xticks(range(1, len(groups) + 1))
        axes.set_xticklabels([f"{det}\n{mat}" for (det, mat), _ in groups], fontsize=8)
        axes.set_title("Razao de inliers por detector e matcher")
        axes.set_ylabel("inliers / matches filtrados")
        _save(figure, figures, run_dir, "inlier_ratio_box", paths)

    if _has(frame, ("detector", "matcher", "reprojection_rmse"), "reprojection_rmse"):
        pivot = frame.pivot_table(
            index="detector", columns="matcher", values="reprojection_rmse", aggfunc="mean"
        )
        figure, axes = plt.subplots(figsize=(8, 5))
        pivot.plot(kind="bar", ax=axes, rot=0)
        axes.set_title("RMSE de reprojecao medio")
        axes.set_ylabel("pixels")
        axes.legend(title="matcher")
        _save(figure, figures, run_dir, "reprojection_rmse", paths)

    if _has(frame, ("detector", "filtered_matches", "n_inliers"), "matches_vs_inliers"):
        figure, axes = plt.subplots(figsize=(7, 6))
        for name, subset in frame.groupby("detector"):
            axes.scatter(
                subset["filtered_matches"], subset["n_inliers"], label=str(name), alpha=0.7
            )
        axes.set_title("Matches filtrados versus inliers")
        axes.set_xlabel("matches filtrados")
        axes.set_ylabel("inliers")
        axes.legend(title="detector")
        _save(figure, figures, run_dir, "matches_vs_inliers", paths)

    logger.info("benchmark figures written: %d", len(paths))
    return paths
