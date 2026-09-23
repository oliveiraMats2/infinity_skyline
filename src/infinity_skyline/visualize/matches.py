"""Stage 4.4: side by side match figures, raw versus geometrically filtered."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import cv2
import matplotlib

matplotlib.use("Agg")  # headless backend, must be set before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from ..config import VisualizeConfig  # noqa: E402
from ..io import KeypointRecord, MatchRecord, save_image  # noqa: E402
from ..logging_setup import get_logger  # noqa: E402
from .keypoints import to_bgr  # noqa: E402

logger = get_logger(__name__)

#: BGR colours: inliers survive RANSAC, outliers do not.
INLIER_COLOR: tuple[int, int, int] = (0, 255, 0)
OUTLIER_COLOR: tuple[int, int, int] = (0, 0, 255)

#: The same colours for matplotlib, which does not speak BGR tuples.
_INLIER_HEX: str = "#55a868"
_OUTLIER_HEX: str = "#c44e52"

#: Above this many images the per cell counts stop being readable.
_ANNOTATE_MAX: int = 20

_FLAGS: int = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS


def _mask_of(match: MatchRecord) -> np.ndarray | None:
    if match.inlier_mask is None:
        return None
    return np.asarray(match.inlier_mask, dtype=bool)


def _select(
    match: MatchRecord,
    inliers_only: bool,
    max_lines: int,
    seed: int,
) -> np.ndarray:
    """Indices of the matches to draw, capped at ``max_lines`` by random sampling."""
    indices = np.arange(match.n_filtered)
    mask = _mask_of(match)
    if inliers_only:
        if mask is None:
            logger.warning(
                "%s -> %s: inliers_only requested but no inlier_mask; drawing all matches",
                match.query,
                match.train,
            )
        else:
            indices = indices[mask]
    if indices.size > max_lines:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=max_lines, replace=False))
    return indices


def _render(
    image_a: np.ndarray,
    rec_a: KeypointRecord,
    image_b: np.ndarray,
    rec_b: KeypointRecord,
    match: MatchRecord,
    indices: np.ndarray,
) -> np.ndarray:
    kp_a = rec_a.cv_keypoints()
    kp_b = rec_b.cv_keypoints()
    dmatches = match.cv_matches()
    left, right = to_bgr(image_a), to_bgr(image_b)
    mask = _mask_of(match)
    if mask is None:
        chosen = [dmatches[int(i)] for i in indices]
        return cv2.drawMatches(
            left, kp_a, right, kp_b, chosen, None, matchColor=INLIER_COLOR, flags=_FLAGS
        )
    outliers = [dmatches[int(i)] for i in indices if not mask[int(i)]]
    inliers = [dmatches[int(i)] for i in indices if mask[int(i)]]
    canvas = cv2.drawMatches(
        left, kp_a, right, kp_b, outliers, None, matchColor=OUTLIER_COLOR, flags=_FLAGS
    )
    # Second pass over the same canvas so both colours share one figure.
    canvas = cv2.drawMatches(
        left,
        kp_a,
        right,
        kp_b,
        inliers,
        canvas,
        matchColor=INLIER_COLOR,
        flags=_FLAGS | cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG,
    )
    return canvas


def draw_matches(
    image_a: np.ndarray,
    rec_a: KeypointRecord,
    image_b: np.ndarray,
    rec_b: KeypointRecord,
    match: MatchRecord,
    config: VisualizeConfig,
    inliers_only: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """One side by side figure with at most ``config.matches.max_lines`` links."""
    indices = _select(match, inliers_only, config.matches.max_lines, seed)
    return _render(image_a, rec_a, image_b, rec_b, match, indices)


def draw_before_after(
    image_a: np.ndarray,
    rec_a: KeypointRecord,
    image_b: np.ndarray,
    rec_b: KeypointRecord,
    match: MatchRecord,
    config: VisualizeConfig,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """``(raw, filtered)``: every match, then only the RANSAC inliers."""
    max_lines = config.matches.max_lines
    raw = _render(
        image_a, rec_a, image_b, rec_b, match, _select(match, False, max_lines, seed)
    )
    filtered = _render(
        image_a, rec_a, image_b, rec_b, match, _select(match, True, max_lines, seed)
    )
    return raw, filtered


def _distance_hist(
    matches: Mapping[tuple[str, str], MatchRecord],
    config: VisualizeConfig,
    run_dir: Path,
) -> Path | None:
    """Descriptor distance of inliers against outliers, pooled over every pair."""
    inliers, outliers = [], []
    for match in matches.values():
        mask = _mask_of(match)
        if mask is None:
            continue  # no RANSAC ran on this pair, so it has no side to fall on
        inliers.append(match.distances[mask])
        outliers.append(match.distances[~mask])
    if not inliers:
        logger.warning("no pair carries an inlier_mask: skipping the distance_hist figure")
        return None

    kept, dropped = np.concatenate(inliers), np.concatenate(outliers)
    # Shared edges, otherwise the two series land on different bins and cannot be read.
    bins = np.histogram_bin_edges(np.concatenate([kept, dropped]), bins=50)
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.hist(dropped, bins=bins, color=_OUTLIER_HEX, alpha=0.6, label="outliers")
    axes.hist(kept, bins=bins, color=_INLIER_HEX, alpha=0.6, label="inliers")
    axes.set_title(
        f"Distancia dos descritores ({kept.size} inliers, {dropped.size} outliers)"
    )
    axes.set_xlabel("distancia")
    axes.set_ylabel("contagem")
    axes.legend()
    figure.tight_layout()
    path = config.figures.path(run_dir, "matches", "distance_hist")
    figure.savefig(path, dpi=config.figures.dpi)
    plt.close(figure)  # closing matters: this runs inside loops
    return path


def _inlier_matrix(
    records: Mapping[str, KeypointRecord],
    matches: Mapping[tuple[str, str], MatchRecord],
    config: VisualizeConfig,
    run_dir: Path,
) -> Path:
    """Symmetric N x N heatmap of the inlier count of every pair."""
    names = sorted(records)
    index = {name: i for i, name in enumerate(names)}
    counts = np.zeros((len(names), len(names)), dtype=float)
    for (a, b), match in matches.items():
        if a in index and b in index:
            counts[index[a], index[b]] = counts[index[b], index[a]] = match.n_inliers
    np.fill_diagonal(counts, 0.0)  # an image against itself says nothing

    side = max(6.0, 0.45 * len(names))
    figure, axes = plt.subplots(figsize=(side + 1.5, side))
    mesh = axes.imshow(counts, cmap="viridis")
    axes.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    axes.set_yticks(range(len(names)), names, fontsize=7)
    figure.colorbar(mesh, ax=axes, label="inliers")
    if len(names) <= _ANNOTATE_MAX:
        for i in range(len(names)):
            for j in range(len(names)):
                axes.text(
                    j, i, str(int(counts[i, j])),
                    ha="center", va="center", fontsize=7, color="white",
                )
    axes.set_title(f"Inliers por par ({len(names)} imagens)")
    figure.tight_layout()
    path = config.figures.path(run_dir, "matches", "inlier_matrix")
    figure.savefig(path, dpi=config.figures.dpi)
    plt.close(figure)  # closing matters: this runs inside loops
    return path


def save_match_figures(
    images: Mapping[str, np.ndarray],
    records: Mapping[str, KeypointRecord],
    matches: Mapping[tuple[str, str], MatchRecord],
    config: VisualizeConfig,
    run_dir: Path,
) -> list[Path]:
    """Write the before/after panels of every pair plus the two summary figures.

    Returns every path written. Pairs left with no filtered match are skipped:
    there is nothing to draw and cv2 would only produce two empty canvases.
    """
    paths: list[Path] = []
    for (a, b), match in tqdm(matches.items(), desc="fig:matches", unit="pair"):
        if match.n_filtered == 0:
            continue
        raw, filtered = draw_before_after(
            images[a], records[a], images[b], records[b], match, config
        )
        for suffix, canvas in (("raw", raw), ("filtered", filtered)):
            path = config.figures.path(run_dir, "matches", f"matches_{a}__{b}_{suffix}")
            save_image(path, canvas)
            paths.append(path)

    histogram = _distance_hist(matches, config, run_dir)
    if histogram is not None:
        paths.append(histogram)
    paths.append(_inlier_matrix(records, matches, config, run_dir))
    logger.info("match figures saved: %d files under %s", len(paths), paths[-1].parent)
    return paths
