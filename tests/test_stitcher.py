"""Learned matches become keypoint indices: snap within 1 px, append otherwise."""

from __future__ import annotations

import numpy as np

from infinity_skyline.io import KeypointRecord
from infinity_skyline.matching.match import keypoint_indices


def test_snap_to_existing_keypoint_or_append_a_new_one() -> None:
    existing = np.zeros((2, 7), dtype=np.float32)
    existing[:, :2] = [(10.0, 10.0), (50.0, 50.0)]
    record = KeypointRecord(
        image="a", detector="reference_superpoint", keypoints=existing,
        descriptors=np.zeros((2, 256), dtype=np.float32), shape=(100, 100),
    )
    points = np.array([(50.5, 50.5), (10.0, 11.0), (30.0, 30.0)])  # 0.71 px, 1 px, far
    confidence = np.array([0.9, 0.8, 0.7])

    assert keypoint_indices(record, points, confidence).tolist() == [1, 0, 2]
    # only the far point was appended, carrying its confidence as the response
    assert record.n_keypoints == 3
    np.testing.assert_allclose(record.keypoints[2], [30.0, 30.0, 8.0, -1.0, 0.7, 0.0, -1.0], rtol=1e-6)

    # a detector free record (LoFTR) builds every keypoint from its matches
    empty = KeypointRecord(
        image="b", detector="reference_loftr", keypoints=np.zeros((0, 7), dtype=np.float32),
        descriptors=np.zeros((0, 0), dtype=np.float32), shape=(100, 100),
    )
    assert keypoint_indices(empty, points, confidence).tolist() == [0, 1, 2]
