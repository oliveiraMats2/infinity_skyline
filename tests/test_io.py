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

"""Persistence layer: keypoint packing, npz round-trips and the image helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from infinity_skyline.io import (
    KeypointRecord,
    MatchRecord,
    array_to_keypoints,
    keypoints_to_array,
    list_images,
    load_keypoints,
    load_matches,
    pair_stem,
    resize_to_max,
    save_keypoints,
    save_matches,
)


def _sample_keypoints() -> list[cv2.KeyPoint]:
    return [
        cv2.KeyPoint(x=12.5, y=7.25, size=3.5, angle=45.5, response=0.125, octave=2, class_id=-1),
        cv2.KeyPoint(x=0.0, y=199.75, size=16.0, angle=-1.0, response=0.0625, octave=0, class_id=3),
        cv2.KeyPoint(x=640.5, y=1.5, size=9.75, angle=359.5, response=0.5, octave=5, class_id=7),
    ]


def _keypoint_record() -> KeypointRecord:
    return KeypointRecord(
        image="scene_01",
        detector="sift",
        keypoints=keypoints_to_array(_sample_keypoints()),
        descriptors=np.arange(3 * 128, dtype=np.float32).reshape(3, 128),
        shape=(480, 640),
        times_ms=np.array([1.5, 2.5, 3.5], dtype=np.float64),
        meta={"truncated": False, "requested": 5000},
    )


def test_keypoint_array_round_trip_preserves_every_field() -> None:
    originals = _sample_keypoints()
    restored = array_to_keypoints(keypoints_to_array(originals))

    assert len(restored) == len(originals)
    for before, after in zip(originals, restored):
        assert after.pt == pytest.approx(before.pt, abs=1e-3)
        assert after.size == pytest.approx(before.size, abs=1e-5)
        assert after.angle == pytest.approx(before.angle, abs=1e-3)
        assert after.response == pytest.approx(before.response, abs=1e-6)
        assert after.octave == before.octave
        assert after.class_id == before.class_id


def test_empty_keypoint_list_keeps_the_column_layout() -> None:
    array = keypoints_to_array([])
    assert array.shape == (0, 7)
    assert array_to_keypoints(array) == []


def test_keypoint_record_file_round_trip(tmp_path: Path) -> None:
    record = _keypoint_record()
    path = tmp_path / "nested" / "scene_01__sift.npz"
    save_keypoints(path, record)
    loaded = load_keypoints(path)

    assert loaded.image == record.image
    assert loaded.detector == record.detector
    assert tuple(loaded.shape) == record.shape
    np.testing.assert_allclose(loaded.keypoints, record.keypoints)
    np.testing.assert_allclose(loaded.descriptors, record.descriptors)
    np.testing.assert_allclose(loaded.times_ms, record.times_ms)
    assert loaded.meta == record.meta
    assert loaded.n_keypoints == 3
    assert loaded.detect_time_ms == pytest.approx(2.5)


def test_descriptor_bytes_reflects_the_detector_footprint() -> None:
    sift = KeypointRecord(
        image="a", detector="sift", keypoints=np.zeros((4, 7), np.float32),
        descriptors=np.zeros((4, 128), np.float32), shape=(10, 10),
    )
    orb = KeypointRecord(
        image="a", detector="orb", keypoints=np.zeros((4, 7), np.float32),
        descriptors=np.zeros((4, 32), np.uint8), shape=(10, 10),
    )
    assert sift.descriptor_bytes == 512
    assert orb.descriptor_bytes == 32


def _match_record(*, geometry: bool) -> MatchRecord:
    pairs = np.array([[0, 5], [1, 6], [2, 7], [3, 8]], dtype=np.int32)
    distances = np.array([10.5, 20.25, 30.125, 40.0], dtype=np.float32)
    mask = np.array([True, False, True, True]) if geometry else None
    homography = (
        np.array([[1.0, 0.01, 12.5], [-0.02, 1.0, -3.25], [1e-5, 2e-5, 1.0]])
        if geometry
        else None
    )
    return MatchRecord(
        query="scene_01",
        train="scene_02",
        pairs=pairs,
        distances=distances,
        n_raw=91,
        inlier_mask=mask,
        homography=homography,
        times_ms=np.array([4.0, 6.0], dtype=np.float64),
        meta={"reprojection_rmse": 0.75, "success": geometry},
    )


@pytest.mark.parametrize("geometry", [False, True])
def test_match_record_file_round_trip(tmp_path: Path, geometry: bool) -> None:
    record = _match_record(geometry=geometry)
    path = tmp_path / f"pair_{geometry}.npz"
    save_matches(path, record)
    loaded = load_matches(path)

    assert loaded.query == record.query
    assert loaded.train == record.train
    assert loaded.n_raw == 91
    np.testing.assert_allclose(loaded.pairs, record.pairs)
    np.testing.assert_allclose(loaded.distances, record.distances)
    np.testing.assert_allclose(loaded.times_ms, record.times_ms)
    assert loaded.meta == record.meta

    if geometry:
        assert loaded.inlier_mask is not None and loaded.homography is not None
        np.testing.assert_allclose(loaded.inlier_mask, record.inlier_mask)
        np.testing.assert_allclose(loaded.homography, record.homography)
        assert loaded.n_inliers == 3
        assert loaded.inlier_ratio == pytest.approx(0.75)
    else:
        assert loaded.inlier_mask is None
        assert loaded.homography is None
        assert loaded.n_inliers == 0
        assert loaded.inlier_ratio == pytest.approx(0.0)


def test_resize_to_max_never_upscales() -> None:
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    assert resize_to_max(image, 400).shape == (50, 100, 3)
    assert resize_to_max(image, None).shape == (50, 100, 3)


def test_resize_to_max_preserves_the_aspect_ratio() -> None:
    image = np.zeros((300, 900, 3), dtype=np.uint8)
    resized = resize_to_max(image, 300)
    height, width = resized.shape[:2]
    assert max(height, width) == 300
    assert width / height == pytest.approx(900 / 300, rel=0.02)


def test_list_images_filters_by_extension_and_sorts(tmp_path: Path) -> None:
    for name in ("c.png", "a.png", "b.PNG", "notes.txt", "raw.cr2", "photo.jpg"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "subdir").mkdir()

    found = list_images(tmp_path, [".png"])
    assert [p.name for p in found] == ["a.png", "b.PNG", "c.png"]

    both = list_images(tmp_path, [".png", ".jpg"])
    assert [p.name for p in both] == ["a.png", "b.PNG", "c.png", "photo.jpg"]


def test_pair_stem_uses_stems_only() -> None:
    assert pair_stem(Path("/data/img_01.png"), "img_02.png") == "img_01__img_02"


def test_convert_caps_at_1920x1080_in_either_orientation() -> None:
    from infinity_skyline.tools.convert_to_png import fit_max_size

    assert fit_max_size(np.zeros((3024, 4032, 3), np.uint8)).shape[:2] == (1080, 1440)
    assert fit_max_size(np.zeros((4032, 3024, 3), np.uint8)).shape[:2] == (1440, 1080)
    assert fit_max_size(np.zeros((2160, 3840, 3), np.uint8)).shape[:2] == (1080, 1920)
    assert fit_max_size(np.zeros((600, 800, 3), np.uint8)).shape[:2] == (600, 800)
