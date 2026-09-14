"""Stage 4.4: side by side match figures, raw versus geometrically filtered."""

from __future__ import annotations

import cv2
import numpy as np

from ..config import VisualizeConfig
from ..io import KeypointRecord, MatchRecord
from ..logging_setup import get_logger
from .keypoints import to_bgr

logger = get_logger(__name__)

#: BGR colours: inliers survive RANSAC, outliers do not.
INLIER_COLOR: tuple[int, int, int] = (0, 255, 0)
OUTLIER_COLOR: tuple[int, int, int] = (0, 0, 255)

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
