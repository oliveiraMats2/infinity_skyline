"""Robust geometric verification of a match set.

RANSAC (or MAGSAC) fits the model named by ``geometry.model`` to the filtered
matches of a pair, which both rejects the outliers the Lowe test let through and
produces the warp ``compose`` needs. The record is mutated in place: the same
``MatchRecord`` that came out of matching carries its inlier mask, its model
matrix and its reprojection error onward.
"""

from __future__ import annotations

from typing import Mapping

import cv2
import numpy as np
from tqdm.auto import tqdm

from ..config import Config, GeometryConfig
from ..io import KeypointRecord, MatchRecord
from ..logging_setup import get_logger

logger = get_logger(__name__)


def _points(record: KeypointRecord, indices: np.ndarray) -> np.ndarray:
    """Keypoint coordinates for ``indices`` as the ``(N, 1, 2)`` float32 cv2 wants."""
    return record.keypoints[indices, :2].reshape(-1, 1, 2).astype(np.float32)


def reprojection_errors(
    src: np.ndarray,
    dst: np.ndarray,
    homography: np.ndarray,
) -> np.ndarray:
    """Per point euclidean distance between ``H @ src`` and ``dst``."""
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2).astype(np.float32), homography)
    difference = projected.reshape(-1, 2) - dst.reshape(-1, 2)
    return np.linalg.norm(difference, axis=1).astype(np.float64)


def estimate_homography(
    query: KeypointRecord,
    train: KeypointRecord,
    record: MatchRecord,
    config: GeometryConfig,
    min_matches: int,
) -> MatchRecord:
    """Fit the geometric model mapping ``query`` points onto ``train`` points.

    Mutates and returns ``record``, setting ``homography``, ``inlier_mask`` and
    the ``success`` / ``model`` / ``reprojection_rmse`` meta keys. A pair that is
    too small, or that RANSAC cannot solve, is a failure and not an exception:
    the pipeline must survive one bad pair out of hundreds.
    """
    record.meta["model"] = config.model
    n_matches = int(record.pairs.shape[0])

    if n_matches < min_matches:
        record.inlier_mask = np.zeros(n_matches, dtype=bool)
        record.homography = None
        record.meta["success"] = False
        record.meta["reason"] = "below_min_matches"
        record.meta["reprojection_rmse"] = float("nan")
        logger.debug(
            "%s -> %s: %d matches < min_matches=%d, skipping RANSAC",
            record.query,
            record.train,
            n_matches,
            min_matches,
        )
        return record

    src = _points(query, record.pairs[:, 0])
    dst = _points(train, record.pairs[:, 1])
    ransac = config.ransac
    method = getattr(cv2, ransac.method)

    if config.model == "fundamental":
        matrix, mask = cv2.findFundamentalMat(
            src,
            dst,
            method,
            ransac.reproj_threshold,
            ransac.confidence,
            ransac.max_iters,
        )
    else:
        matrix, mask = cv2.findHomography(
            srcPoints=src,
            dstPoints=dst,
            method=method,
            ransacReprojThreshold=ransac.reproj_threshold,
            maxIters=ransac.max_iters,
            confidence=ransac.confidence,
        )

    if matrix is None or mask is None:
        record.inlier_mask = np.zeros(n_matches, dtype=bool)
        record.homography = None
        record.meta["success"] = False
        record.meta["reason"] = "no_model"
        record.meta["reprojection_rmse"] = float("nan")
        logger.debug("%s -> %s: RANSAC found no model", record.query, record.train)
        return record

    inliers = mask.ravel().astype(bool)[:n_matches]
    record.inlier_mask = inliers
    record.homography = np.asarray(matrix, dtype=np.float64)
    record.meta["success"] = True
    record.meta.pop("reason", None)

    if config.model == "fundamental":
        # A fundamental matrix maps points to epipolar lines, not to points, so a
        # point to point reprojection error is not defined for it.
        rmse = float("nan")
    elif inliers.any():
        # The metric is the residual of the model on the data it was fitted to,
        # so outliers, which are by definition arbitrarily far, stay out of it.
        errors = reprojection_errors(src[inliers], dst[inliers], record.homography)
        rmse = float(np.sqrt(np.mean(np.square(errors))))
    else:
        rmse = float("nan")
    record.meta["reprojection_rmse"] = rmse

    logger.debug(
        "%s -> %s: %d/%d inliers (%.2f), rmse=%.3f px",
        record.query,
        record.train,
        int(inliers.sum()),
        n_matches,
        record.inlier_ratio,
        rmse,
    )
    return record


def estimate_all(
    keypoints: Mapping[str, KeypointRecord],
    matches: Mapping[tuple[str, str], MatchRecord],
    config: Config,
) -> dict[tuple[str, str], MatchRecord]:
    """Verify every pair, in place. Returns the same mapping as a plain dict."""
    verified: dict[tuple[str, str], MatchRecord] = dict(matches)
    for record in tqdm(verified.values(), desc="ransac", unit="pair"):
        estimate_homography(
            keypoints[record.query],
            keypoints[record.train],
            record,
            config.geometry,
            config.filters.min_matches,
        )
    n_ok = sum(1 for record in verified.values() if record.meta.get("success"))
    logger.info("geometric verification: %d/%d pairs kept a model", n_ok, len(verified))
    return verified
