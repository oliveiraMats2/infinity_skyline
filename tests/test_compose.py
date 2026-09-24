"""Panorama composition: chaining, reference choice, canvas guard and deghosting."""

from __future__ import annotations

import cv2
import networkx as nx
import numpy as np
import pytest

from scipy.spatial.transform import Rotation

from infinity_skyline.compose import (
    _bundle_adjust,
    _intrinsics,
    _rotation_from_homography,
    chain_homographies,
    compose_panorama,
    pick_reference,
)
from infinity_skyline.config import BlendConfig, ComposeConfig, SeamConfig

#: Horizontal offset between the two crops, in pixels of the source scene.
SHIFT: int = 140


def _translation(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _normalized(homography: np.ndarray) -> np.ndarray:
    return np.asarray(homography, dtype=np.float64) / float(homography[2, 2])


def _scene() -> np.ndarray:
    """Dark canvas with one bright rectangle plus dim texture lines around it."""
    canvas = np.full((200, 400, 3), 10, dtype=np.uint8)
    canvas[60:140, 120:280] = 220  # the object the seam must not duplicate
    canvas[20:24, :] = 90  # texture for the seam finder, below the bright threshold
    canvas[170:174, :] = 90
    canvas[:, 320:324] = 90
    canvas[:, 40:44] = 90
    return canvas


def _overlapping_pair() -> tuple[dict[str, np.ndarray], nx.Graph]:
    scene = _scene()
    left = np.ascontiguousarray(scene[:, 0:260])
    right = np.ascontiguousarray(scene[:, SHIFT:400])

    graph = nx.Graph()
    graph.add_node("left")
    graph.add_node("right")
    graph.add_edge(
        "left",
        "right",
        weight=400,
        inlier_ratio=0.95,
        rmse=0.4,
        homography=_translation(-SHIFT, 0.0),  # maps left coordinates onto right
        query="left",
        train="right",
    )
    return {"left": left, "right": right}, graph


def _bright_components(panorama: np.ndarray) -> int:
    image = panorama
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = (image > 128).astype(np.uint8)
    n_labels, _ = cv2.connectedComponents(binary, connectivity=8)
    return n_labels - 1  # label 0 is the background


def _config(seam: str, blend: str) -> ComposeConfig:
    return ComposeConfig(
        reference="first",
        projection="planar",
        canvas_max=4000,
        seam=SeamConfig(finder=seam, scale=1.0),
        blend=BlendConfig(method=blend, bands=3, sharpness=0.02),
        exposure="none",
        save_naive=False,
    )


def test_simple_path_does_not_duplicate_the_object() -> None:
    images, graph = _overlapping_pair()
    result = compose_panorama(images, ["left", "right"], graph, _config("none", "none"))

    assert result.panorama.size > 0
    assert result.panorama.shape[1] >= 380  # both crops actually landed on the canvas
    assert _bright_components(result.panorama) == 1


@pytest.mark.skipif(
    not hasattr(cv2, "detail_GraphCutSeamFinder"),
    reason="cv2.detail_GraphCutSeamFinder is missing: install opencv-contrib-python",
)
def test_graphcut_and_multiband_do_not_duplicate_the_object() -> None:
    images, graph = _overlapping_pair()
    result = compose_panorama(images, ["left", "right"], graph, _config("graphcut", "multiband"))

    assert _bright_components(result.panorama) == 1
    assert set(result.global_homographies) == {"left", "right"}


def test_chain_homographies_matches_the_hand_made_product() -> None:
    h_ab = _translation(-SHIFT, 0.0)
    h_bc = _translation(-130.0, 12.0)

    graph = nx.Graph()
    graph.add_edge("a", "b", weight=300, inlier_ratio=0.9, rmse=0.5,
                   homography=h_ab, query="a", train="b")
    graph.add_edge("b", "c", weight=280, inlier_ratio=0.9, rmse=0.5,
                   homography=h_bc, query="b", train="c")

    chained = chain_homographies(["a", "b", "c"], graph, "a")

    expected_b = np.linalg.inv(h_ab)
    expected_c = np.linalg.inv(h_ab) @ np.linalg.inv(h_bc)
    np.testing.assert_allclose(_normalized(chained["a"]), np.eye(3), atol=1e-9)
    np.testing.assert_allclose(_normalized(chained["b"]), _normalized(expected_b), atol=1e-9)
    np.testing.assert_allclose(_normalized(chained["c"]), _normalized(expected_c), atol=1e-9)


def test_pick_reference_modes() -> None:
    order = ["a", "b", "c", "d", "e"]
    assert pick_reference(order, "center") == "c"
    assert pick_reference(order, "first") == "a"
    assert pick_reference(order, 3) == "d"
    assert pick_reference(order, 0) == "a"


def test_canvas_max_too_small_names_the_offending_image() -> None:
    images, graph = _overlapping_pair()
    config = _config("none", "none")
    config.canvas_max = 10

    with pytest.raises(ValueError) as excinfo:
        compose_panorama(images, ["left", "right"], graph, config)
    assert any(name in str(excinfo.value) for name in ("left", "right"))


def test_bundle_adjust_recovers_rotations_and_focal_from_a_bad_start() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    names = ["a", "b", "c", "d", "e"]
    sources = {name: frame for name in names}
    truth = {
        name: Rotation.from_euler("yxz", [yaw, 2.0 * i - 4.0, 1.0 - i], degrees=True).as_matrix()
        for i, (name, yaw) in enumerate(zip(names, (-50, -25, 0, 25, 50)))
    }
    truth["c"] = np.eye(3)
    k = _intrinsics(500.0, frame).astype(np.float64)

    graph = nx.Graph()
    for query, train in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("a", "c"), ("c", "e")]:
        exact = k @ truth[train].T @ truth[query] @ np.linalg.inv(k)
        graph.add_edge(query, train, weight=100, homography=exact, query=query, train=train)

    # A chain gone a few degrees astray per frame, and a focal 20 percent off.
    rng = np.random.default_rng(0)
    noise = {name: Rotation.from_rotvec(rng.normal(0.0, 0.03, 3)).as_matrix() for name in names}
    start = {name: k @ truth[name] @ noise[name] @ np.linalg.inv(k) for name in names}
    refined, focal = _bundle_adjust(graph, sources, start, "c", 600.0)

    assert abs(focal - 500.0) < 1.0
    for name in names:
        rotation = _rotation_from_homography(_intrinsics(focal, frame), refined[name])
        np.testing.assert_allclose(rotation, truth[name], atol=1e-3)
