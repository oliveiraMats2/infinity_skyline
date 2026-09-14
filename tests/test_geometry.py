"""Homography estimation against a ground truth transform."""

from __future__ import annotations

import numpy as np
import pytest

from infinity_skyline.config import GeometryConfig, RansacConfig
from infinity_skyline.geometry.homography import estimate_homography, reprojection_errors
from infinity_skyline.io import KeypointRecord, MatchRecord

#: Rotation, translation and a small amount of perspective. Known exactly.
TRUE_H: np.ndarray = np.array(
    [
        [np.cos(np.deg2rad(7.0)), -np.sin(np.deg2rad(7.0)), 12.0],
        [np.sin(np.deg2rad(7.0)), np.cos(np.deg2rad(7.0)), -5.0],
        [1.5e-5, 2.5e-5, 1.0],
    ],
    dtype=np.float64,
)


def _normalized(homography: np.ndarray) -> np.ndarray:
    return np.asarray(homography, dtype=np.float64) / float(homography[2, 2])


def cv_perspective(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Project ``(N, 2)`` points through a 3x3 homography."""
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    projected = homogeneous @ np.asarray(homography, dtype=np.float64).T
    return projected[:, :2] / projected[:, 2:3]


def _record(name: str, points: np.ndarray) -> KeypointRecord:
    keypoints = np.zeros((points.shape[0], 7), dtype=np.float32)
    keypoints[:, :2] = points
    keypoints[:, 2] = 4.0
    return KeypointRecord(
        image=name,
        detector="sift",
        keypoints=keypoints,
        descriptors=np.zeros((points.shape[0], 128), dtype=np.float32),
        shape=(480, 640),
    )


def _matches(query: str, train: str, count: int) -> MatchRecord:
    index = np.arange(count, dtype=np.int32)
    return MatchRecord(
        query=query,
        train=train,
        pairs=np.column_stack([index, index]),
        distances=np.full(count, 50.0, dtype=np.float32),
        n_raw=count * 2,
    )


def _geometry_config() -> GeometryConfig:
    return GeometryConfig(ransac=RansacConfig(method="RANSAC", reproj_threshold=3.0))


def _scene(n: int = 100, seed: int = 20240914) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    source = np.column_stack([rng.uniform(20.0, 620.0, n), rng.uniform(20.0, 460.0, n)])
    return source, cv_perspective(TRUE_H, source)


def test_clean_correspondences_recover_the_true_homography() -> None:
    source, target = _scene()
    record = estimate_homography(
        _record("q", source), _record("t", target), _matches("q", "t", len(source)),
        _geometry_config(), min_matches=10,
    )

    assert record.meta["success"] is True
    assert record.homography is not None
    np.testing.assert_allclose(_normalized(record.homography), _normalized(TRUE_H), atol=1e-3)
    assert record.inlier_mask is not None
    assert record.inlier_mask.sum() == len(source)
    assert record.meta["reprojection_rmse"] == pytest.approx(0.0, abs=1e-2)


def test_ransac_survives_thirty_percent_gross_outliers() -> None:
    source, target = _scene()
    rng = np.random.default_rng(7)
    n_bad = int(0.3 * len(source))
    corrupted = np.zeros(len(source), dtype=bool)
    corrupted[rng.choice(len(source), n_bad, replace=False)] = True
    offsets = rng.choice([-1.0, 1.0], size=(n_bad, 2)) * rng.uniform(80.0, 250.0, (n_bad, 2))
    target = target.copy()
    target[corrupted] += offsets

    record = estimate_homography(
        _record("q", source), _record("t", target), _matches("q", "t", len(source)),
        _geometry_config(), min_matches=10,
    )

    assert record.meta["success"] is True
    assert record.homography is not None
    np.testing.assert_allclose(_normalized(record.homography), _normalized(TRUE_H), atol=1e-2)

    mask = np.asarray(record.inlier_mask, dtype=bool)
    clean = ~corrupted
    assert mask[clean].mean() > 0.95  # the untouched points are recognised
    assert mask[corrupted].sum() <= 2  # and the gross ones are not


def test_too_few_matches_short_circuits_the_estimation() -> None:
    source, target = _scene(n=6)
    record = estimate_homography(
        _record("q", source), _record("t", target), _matches("q", "t", 6),
        _geometry_config(), min_matches=10,
    )

    assert record.meta["success"] is False
    assert record.homography is None
    assert record.n_inliers == 0


def test_reprojection_errors_vanish_for_the_exact_homography() -> None:
    source, target = _scene(n=25)
    errors = reprojection_errors(source, target, TRUE_H)

    assert errors.shape == (25,)
    # float32 is the dtype of OpenCV keypoints, so sub 1e-4 px residue is the floor.
    np.testing.assert_allclose(errors, np.zeros(25), atol=1e-3)


def test_reprojection_errors_measure_a_known_shift() -> None:
    source, target = _scene(n=10)
    shifted = target + np.array([3.0, 4.0])  # a 5 pixel offset on every point
    np.testing.assert_allclose(
        reprojection_errors(source, shifted, TRUE_H), np.full(10, 5.0), atol=1e-3
    )
