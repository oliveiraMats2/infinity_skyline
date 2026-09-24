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

"""Panorama composition: stages 5.2, 6.1, 6.2 and 6.3.

The pipeline is the classic OpenCV stitching detail flow, driven by the graph
built in :mod:`geometry.graph`, with the bundle adjustment optional:

1. chain the pairwise homographies of the graph into one homography per image,
   all expressed in the frame of a single reference image (stage 5.2), and
   optionally refine them jointly over every edge with a bundle adjustment;
2. warp every source into that frame, planar or onto a cylinder/sphere, and size
   the canvas from the union of the warped corners (stage 6.1);
3. compensate exposure, then cut an optimal seam through the overlaps (6.2);
4. blend across the seam with a multiband, feather or linear blender (6.3).

Every image is warped into its own bounding box, not into a full canvas copy, so
the peak memory is the sum of the warped tiles and not N times the canvas.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import cv2
import networkx as nx
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm

from .config import BlendConfig, ComposeConfig, SeamConfig
from .logging_setup import get_logger

logger = get_logger(__name__)

#: ``compose.projection`` to the surface name understood by ``cv2.PyRotationWarper``.
PROJECTION_NAMES: dict[str, str] = {
    "planar": "plane",
    "cylindrical": "cylindrical",
    "spherical": "spherical",
}

#: ``compose.exposure`` to the ``cv2.detail.ExposureCompensator_*`` enum name.
EXPOSURE_TYPES: dict[str, str] = {
    "gain": "ExposureCompensator_GAIN",
    "gain_blocks": "ExposureCompensator_GAIN_BLOCKS",
}

#: ``compose.wave_correct`` to the ``cv2.detail.WAVE_CORRECT_*`` enum name.
WAVE_CORRECT_KINDS: dict[str, str] = {
    "horiz": "WAVE_CORRECT_HORIZ",
    "vert": "WAVE_CORRECT_VERT",
}

#: Warped tile: image, its 8 bit mask and its top left corner on the canvas.
Tile = tuple[np.ndarray, np.ndarray, tuple[int, int]]


@dataclass(slots=True)
class ComposeResult:
    """Everything stage 6.4 and 6.5 need to judge the mosaic."""

    panorama: np.ndarray
    naive: np.ndarray | None = None
    seam_mask: np.ndarray | None = None
    warped_corners: list[tuple[int, int]] = field(default_factory=list)
    global_homographies: dict[str, np.ndarray] = field(default_factory=dict)
    # One panorama per ``blend.compare`` method, same tiles and seams, and its time.
    blends: dict[str, np.ndarray] = field(default_factory=dict)
    blend_times_ms: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# cv2.detail lookup
# --------------------------------------------------------------------------- #
def _cv_attr(*names: str) -> Any:
    """First existing dotted attribute of ``cv2`` among ``names``.

    The ``detail_*`` classes moved between ``cv2`` and ``cv2.detail`` across 4.x
    releases, so each call site lists the spellings it accepts.
    """
    for name in names:
        found: Any = cv2
        for part in name.split("."):
            found = getattr(found, part, None)
            if found is None:
                break
        if found is not None:
            return found
    raise RuntimeError(
        "this OpenCV build exposes none of: "
        + ", ".join(f"cv2.{n}" for n in names)
        + "; install opencv-contrib-python"
    )


# --------------------------------------------------------------------------- #
# stage 5.2: homography chaining
# --------------------------------------------------------------------------- #
def _edge_homography(graph: nx.Graph, source: str, target: str) -> np.ndarray:
    """Homography mapping points of ``source`` onto points of ``target``."""
    data = graph.edges[source, target]
    homography = np.asarray(data["homography"], dtype=np.float64)
    if data.get("train") == source:
        # The edge stores query -> train, and we are walking it backwards.
        return np.linalg.inv(homography)
    return homography


def chain_homographies(
    order: Sequence[str],
    graph: nx.Graph,
    reference: str,
) -> dict[str, np.ndarray]:
    """Compose pairwise homographies along the graph path to ``reference``.

    The path runs over the maximum spanning tree, so the chain goes through the
    edges with the most inliers rather than through the fewest hops. Error
    multiplies along the chain, and one weak edge taken as a shortcut costs far
    more than an extra strong hop: routing this way took the worst frame of the
    default scene from 29 degrees of spurious roll down to 17.

    The reference gets the identity. A node with no path to the reference is
    logged and left out, so a disconnected intruder never reaches the canvas.
    """
    routes = nx.maximum_spanning_tree(graph, weight="weight")
    chained: dict[str, np.ndarray] = {}
    for name in order:
        if name == reference:
            chained[name] = np.eye(3, dtype=np.float64)
            continue
        if name not in graph or reference not in graph:
            logger.warning("'%s' is not a node of the graph; skipping it", name)
            continue
        try:
            path: list[str] = nx.shortest_path(routes, name, reference)
        except nx.NetworkXNoPath:
            logger.warning(
                "no path from '%s' to the reference '%s'; skipping it", name, reference
            )
            continue
        total = np.eye(3, dtype=np.float64)
        try:
            for step_source, step_target in zip(path, path[1:]):
                total = _edge_homography(graph, step_source, step_target) @ total
        except np.linalg.LinAlgError:
            logger.warning(
                "singular homography on the path from '%s' to '%s'; skipping it",
                name,
                reference,
            )
            continue
        if not np.isfinite(total).all() or abs(total[2, 2]) < 1e-12:
            logger.warning("degenerate chained homography for '%s'; skipping it", name)
            continue
        chained[name] = total / total[2, 2]
    return chained


def pick_reference(order: Sequence[str], reference: str | int) -> str:
    """Resolve ``compose.reference`` ("center", "first" or an index) to a name."""
    if not order:
        raise ValueError("cannot pick a reference frame: the image order is empty")
    if isinstance(reference, int) and not isinstance(reference, bool):
        if not -len(order) <= reference < len(order):
            raise ValueError(
                f"compose.reference index {reference} is out of range for "
                f"{len(order)} ordered images"
            )
        return order[reference]
    if reference == "first":
        return order[0]
    if reference == "center":
        return order[len(order) // 2]
    raise ValueError(
        f"compose.reference {reference!r} is not 'center', 'first' or an integer index"
    )


# --------------------------------------------------------------------------- #
# stage 6.1: warping and canvas
# --------------------------------------------------------------------------- #
def _as_bgr(image: np.ndarray) -> np.ndarray:
    """cv2.detail wants 8UC3 everywhere; grayscale sources are promoted here."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _image_corners(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    return np.array(
        [[[0.0, 0.0]], [[width, 0.0]], [[width, height]], [[0.0, height]]],
        dtype=np.float64,
    )


def _check_canvas(
    names: Sequence[str],
    rois: Sequence[tuple[int, int, int, int]],
    width: int,
    height: int,
    canvas_max: int,
) -> None:
    """Refuse to allocate a canvas blown up by a degenerate homography."""
    if width <= canvas_max and height <= canvas_max and width > 0 and height > 0:
        return
    worst = max(range(len(rois)), key=lambda i: max(rois[i][2], rois[i][3]))
    raise ValueError(
        f"the canvas would be {width}x{height} px, past compose.canvas_max="
        f"{canvas_max}; '{names[worst]}' alone warps to {rois[worst][2]}x"
        f"{rois[worst][3]} px, so its homography is degenerate"
    )


def _normalize_rois(
    rois: list[tuple[int, int, int, int]]
) -> tuple[list[tuple[int, int, int, int]], int, int]:
    """Shift every ROI so the canvas origin is (0, 0) and return the canvas size."""
    offset_x = min(x for x, _, _, _ in rois)
    offset_y = min(y for _, y, _, _ in rois)
    shifted = [(x - offset_x, y - offset_y, w, h) for x, y, w, h in rois]
    width = max(x + w for x, _, w, _ in shifted)
    height = max(y + h for _, y, _, h in shifted)
    return shifted, width, height


def _warp_planar(
    sources: Mapping[str, np.ndarray],
    names: Sequence[str],
    homographies: Mapping[str, np.ndarray],
    canvas_max: int,
) -> tuple[list[Tile], int, int]:
    """Plain perspective warp of every source into the reference plane."""
    rois: list[tuple[int, int, int, int]] = []
    for name in names:
        projected = cv2.perspectiveTransform(_image_corners(sources[name]), homographies[name])
        if not np.isfinite(projected).all():
            raise ValueError(
                f"'{name}' projects to non finite coordinates; its homography is degenerate"
            )
        points = projected.reshape(-1, 2)
        low = np.floor(points.min(axis=0)).astype(np.int64)
        high = np.ceil(points.max(axis=0)).astype(np.int64)
        rois.append((int(low[0]), int(low[1]), int(high[0] - low[0]), int(high[1] - low[1])))

    shifted, width, height = _normalize_rois(rois)
    _check_canvas(names, shifted, width, height, canvas_max)

    offset_x = min(x for x, _, _, _ in rois)
    offset_y = min(y for _, y, _, _ in rois)
    tiles: list[Tile] = []
    for name, (x, y, tile_w, tile_h) in tqdm(
        list(zip(names, shifted)), desc="Warping to the reference frame", unit="img"
    ):
        # Canvas translation and per tile translation folded into one matrix.
        translation = np.array(
            [[1.0, 0.0, -(offset_x + x)], [0.0, 1.0, -(offset_y + y)], [0.0, 0.0, 1.0]]
        )
        matrix = translation @ homographies[name]
        image = sources[name]
        warped = cv2.warpPerspective(
            image, matrix, (tile_w, tile_h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        mask = cv2.warpPerspective(
            np.full(image.shape[:2], 255, dtype=np.uint8), matrix, (tile_w, tile_h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
        )
        tiles.append((warped, mask, (x, y)))
    return tiles, width, height


def _focals_from_homography(homography: np.ndarray) -> tuple[float | None, float | None]:
    """Focal lengths implied by H under a pure rotation model, or None each.

    cv2.detail.focalsFromHomography exists but is useless from Python: its focal
    values are C++ output references that the binding does not return. This is
    the same closed form (Shum and Szeliski) that OpenCV's autocalib.cpp uses.
    """
    h = np.asarray(homography, dtype=np.float64).ravel()

    def _solve(d1: float, d2: float, v1: float, v2: float) -> float | None:
        if v1 < v2:
            v1, v2 = v2, v1
            d1, d2 = d2, d1
        if v1 > 0.0 and v2 > 0.0:
            return float(np.sqrt(v1 if abs(d1) > abs(d2) else v2))
        if v1 > 0.0:
            return float(np.sqrt(v1))
        return None

    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = h[6] * h[7]
        d2 = (h[7] - h[6]) * (h[7] + h[6])
        f1 = _solve(
            d1,
            d2,
            -(h[0] * h[1] + h[3] * h[4]) / d1,
            (h[0] * h[0] + h[3] * h[3] - h[1] * h[1] - h[4] * h[4]) / d2,
        )
        d1 = h[0] * h[3] + h[1] * h[4]
        d2 = h[0] * h[0] + h[1] * h[1] - h[3] * h[3] - h[4] * h[4]
        f0 = _solve(d1, d2, -h[2] * h[5] / d1, (h[5] * h[5] - h[2] * h[2]) / d2)

    def _finite(value: float | None) -> float | None:
        return value if value is not None and np.isfinite(value) and value > 0.0 else None

    return _finite(f0), _finite(f1)


def _resolve_focal(
    config: ComposeConfig, graph: nx.Graph, frame_width: int
) -> float | None:
    """Focal in pixels: taken from the config when stated, estimated otherwise.

    Stating it matters. The closed form in :func:`_focals_from_homography` recovers the
    focal from the perspective terms of the homography, and those vanish under a small
    rotation, so the estimate degenerates exactly on the well behaved sweeps this
    pipeline targets.
    """
    if config.focal_px is not None:
        logger.info("focal: %.1f px, from compose.focal_px", config.focal_px)
        return config.focal_px
    if config.focal_mm is not None and config.sensor_width_mm is not None:
        focal = config.focal_mm / config.sensor_width_mm * frame_width
        fov = 2.0 * np.degrees(np.arctan(frame_width / (2.0 * focal)))
        logger.info(
            "focal: %.1f px, from %.1f mm on a %.1f mm sensor at %d px wide (%.1f deg fov)",
            focal, config.focal_mm, config.sensor_width_mm, frame_width, fov,
        )
        return focal
    if config.focal_mm is not None or config.sensor_width_mm is not None:
        logger.warning(
            "compose.focal_mm and compose.sensor_width_mm only work as a pair; "
            "falling back to estimating the focal from the homographies"
        )
    return _estimate_focal(graph)


def _estimate_focal(graph: nx.Graph) -> float | None:
    """Median focal over the edge homographies, or None when it cannot be had."""
    focals: list[float] = []
    for _, _, data in graph.edges(data=True):
        homography = data.get("homography")
        if homography is None:
            continue
        f0, f1 = _focals_from_homography(homography)
        focals.extend(f for f in (f0, f1) if f is not None)
    if not focals:
        logger.warning(
            "no focal length could be estimated from the graph homographies; "
            "falling back to the planar projection"
        )
        return None
    focal = float(np.median(focals))
    spread = max(focals) / min(focals)
    logger.info(
        "estimated focal: %.1f px from %d edge estimates over %d edges",
        focal, len(focals), graph.number_of_edges(),
    )
    if spread > 2.0:
        logger.warning(
            "the focal estimates span %.0f to %.0f px, a factor of %.1f: the closed form "
            "is ill conditioned under small rotations. Set compose.focal_mm and "
            "compose.sensor_width_mm (or compose.focal_px) to place the frames correctly",
            min(focals), max(focals), spread,
        )
    return focal


def _rotation_from_homography(intrinsics: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Rotation implied by ``homography``, in the convention the warper wants.

    With the reference at identity the homography is H = K R^-1 K^-1, so the
    camera rotation is R = K^-1 H^-1 K. But cv2.PyRotationWarper turns an image
    point into a world ray with ``R * K^-1``, which is the inverse of that, so
    what it must be handed is K^-1 H K. Passing the camera rotation instead lays
    the frames out in reverse and the whole sweep comes back mirrored.
    The SVD projects the noisy result onto the nearest true rotation matrix.
    """
    # H and -H are the same projective transform, but findHomography normalizes
    # H[2,2] to 1, and that flips the sign of the whole matrix whenever the true
    # [2,2] is negative, which happens once a frame is far enough from the
    # reference. The determinant then goes negative, the SVD below falls into the
    # reflection branch, and the rotation comes back wrong by more than 100
    # degrees. Restoring the sign is what keeps the end frames upright.
    scaled = np.asarray(homography, dtype=np.float64)
    if np.linalg.det(scaled) < 0.0:
        scaled = -scaled
    matrix = intrinsics.astype(np.float64)
    rotation = np.linalg.inv(matrix) @ scaled @ matrix
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        rotation = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    return np.ascontiguousarray(rotation, dtype=np.float32)


def _intrinsics(focal: float, image: np.ndarray) -> np.ndarray:
    """K with the principal point at the centre of ``image``."""
    height, width = image.shape[:2]
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _bundle_adjust(
    graph: nx.Graph,
    sources: Mapping[str, np.ndarray],
    homographies: Mapping[str, np.ndarray],
    reference: str,
    focal: float,
) -> tuple[dict[str, np.ndarray], float]:
    """Refine every rotation and the shared focal jointly over all graph edges.

    The chain reads only the spanning tree, so the edges it leaves out, the ones
    that close a loop, constrain nothing. Here every edge votes: a grid of points
    of the query frame, mapped by the edge homography into the train frame, must
    land on the same pixels through the cameras, ``K W_t^T W_q K^-1 p``, with W the
    warper rotation of :func:`_rotation_from_homography`. The reference stays at
    identity; the rotation vectors of the others and the log focal are solved by a
    robust least squares. The result goes back out as homographies to the
    reference, ``K_ref W K^-1``, so both warp paths take it unchanged. When the
    solve fails or does not lower the residual, the chained estimates are kept.
    """
    names = list(homographies)
    index = {name: i for i, name in enumerate(names)}
    free = [i for i, name in enumerate(names) if name != reference]
    start = np.array(
        [_rotation_from_homography(_intrinsics(focal, sources[n]), homographies[n]) for n in names]
    )
    centers = np.array([_intrinsics(1.0, sources[n])[:2, 2] for n in names], dtype=np.float64)

    query_ids, train_ids, query_pts, train_pts = [], [], [], []
    for _, _, data in graph.edges(data=True):
        query, train = data["query"], data["train"]
        if query not in index or train not in index:
            continue
        height, width = sources[query].shape[:2]
        grid = np.stack(
            np.meshgrid(np.linspace(0, width, 8), np.linspace(0, height, 6)), axis=-1
        ).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(grid, np.asarray(data["homography"], np.float64))
        grid, mapped = grid.reshape(-1, 2), mapped.reshape(-1, 2)
        inside = ((mapped >= 0) & (mapped < sources[train].shape[1::-1])).all(axis=1)
        query_ids += [index[query]] * int(inside.sum())
        train_ids += [index[train]] * int(inside.sum())
        query_pts.append(grid[inside])
        train_pts.append(mapped[inside])
    if not query_ids:
        logger.warning("no edge overlaps another frame; skipping bundle adjustment")
        return dict(homographies), focal
    query_pts, train_pts = np.concatenate(query_pts), np.concatenate(train_pts)

    def rotations(params: np.ndarray) -> np.ndarray:
        out = np.repeat(np.eye(3)[None], len(names), axis=0)
        out[free] = Rotation.from_rotvec(params[1:].reshape(-1, 3)).as_matrix()
        return out

    def residuals(params: np.ndarray) -> np.ndarray:
        f, rot = np.exp(params[0]), rotations(params)
        rays = np.column_stack(((query_pts - centers[query_ids]) / f, np.ones(len(query_pts))))
        world = np.einsum("nij,nj->ni", rot[query_ids], rays)
        camera = np.einsum("nji,nj->ni", rot[train_ids], world)
        return (f * camera[:, :2] / camera[:, 2:] + centers[train_ids] - train_pts).ravel()

    def rms(params: np.ndarray) -> float:
        # Per point, over x and y together.
        return float(np.sqrt(2.0 * np.mean(residuals(params) ** 2)))

    x0 = np.concatenate([[np.log(focal)], Rotation.from_matrix(start[free]).as_rotvec().ravel()])
    try:
        solved = least_squares(residuals, x0, loss="soft_l1", f_scale=2.0, x_scale="jac")
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.warning("bundle adjustment failed (%s); keeping the chained estimates", exc)
        return dict(homographies), focal
    before, after = rms(x0), rms(solved.x)
    logger.info(
        "bundle adjustment: %d points over %d edges, rms %.2f -> %.2f px, focal %.1f -> %.1f px",
        len(query_pts), len(set(zip(query_ids, train_ids))), before, after, focal,
        np.exp(solved.x[0]),
    )
    if not solved.success or not after < before:
        logger.warning("bundle adjustment did not improve the residual; keeping the chained estimates")
        return dict(homographies), focal

    focal = float(np.exp(solved.x[0]))
    k_ref = _intrinsics(focal, sources[reference]).astype(np.float64)
    refined = {
        name: k_ref @ rotation @ np.linalg.inv(_intrinsics(focal, sources[name]))
        for name, rotation in zip(names, rotations(solved.x))
    }
    return refined, focal


def _mean_roll_degrees(rotations: Sequence[np.ndarray]) -> float:
    """Mean roll of the rotations, in degrees, zero when the horizon is level."""
    return float(np.mean([np.degrees(np.arctan2(r[1, 0], r[0, 0])) for r in rotations]))


def _wave_correct(rotations: list[np.ndarray], kind: str) -> list[np.ndarray]:
    """Straighten the horizon by pushing the accumulated roll back towards zero.

    The rotations come from :func:`_rotation_from_homography`, already float32 and
    contiguous, which is what ``waveCorrect`` insists on.
    """
    if kind == "none":
        return rotations
    detail = _cv_attr("detail")
    before = _mean_roll_degrees(rotations)
    try:
        corrected = detail.waveCorrect(rotations, getattr(detail, WAVE_CORRECT_KINDS[kind]))
    except cv2.error as exc:
        logger.warning(
            "wave correction (%s) failed (%s); keeping the uncorrected rotations", kind, exc
        )
        return rotations
    # Some builds return the corrected list, others edit it in place.
    out = list(corrected) if corrected is not None else rotations
    logger.info(
        "wave correct (%s): mean roll %.2f -> %.2f deg over %d rotations",
        kind, before, _mean_roll_degrees(out), len(out),
    )
    return out


def _warp_rotation(
    sources: Mapping[str, np.ndarray],
    names: Sequence[str],
    homographies: Mapping[str, np.ndarray],
    projection: str,
    focal: float,
    canvas_max: int,
    wave_correct: str,
) -> tuple[list[Tile], int, int]:
    """Warp onto a cylinder or a sphere with ``cv2.PyRotationWarper``."""
    warper = _cv_attr("PyRotationWarper")(PROJECTION_NAMES[projection], focal)
    intrinsics_all: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    for name in names:
        intrinsics = _intrinsics(focal, sources[name])
        intrinsics_all.append(intrinsics)
        rotations.append(_rotation_from_homography(intrinsics, homographies[name]))

    # Straighten before the ROIs: the canvas is sized from the corrected orientation.
    rotations = _wave_correct(rotations, wave_correct)

    rois: list[tuple[int, int, int, int]] = []
    for name, intrinsics, rotation in zip(names, intrinsics_all, rotations):
        height, width = sources[name].shape[:2]
        x, y, roi_w, roi_h = warper.warpRoi((width, height), intrinsics, rotation)
        rois.append((int(x), int(y), int(roi_w), int(roi_h)))

    shifted, canvas_w, canvas_h = _normalize_rois(rois)
    _check_canvas(names, shifted, canvas_w, canvas_h, canvas_max)

    tiles: list[Tile] = []
    for name, intrinsics, rotation, (x, y, _, _) in tqdm(
        list(zip(names, intrinsics_all, rotations, shifted)),
        desc=f"Warping to the {projection} surface",
        unit="img",
    ):
        image = sources[name]
        _, warped = warper.warp(
            image, intrinsics, rotation, cv2.INTER_LINEAR, cv2.BORDER_REFLECT
        )
        _, mask = warper.warp(
            np.full(image.shape[:2], 255, dtype=np.uint8),
            intrinsics,
            rotation,
            cv2.INTER_NEAREST,
            cv2.BORDER_CONSTANT,
        )
        tiles.append((np.asarray(warped), np.asarray(mask), (x, y)))
    return tiles, canvas_w, canvas_h


# --------------------------------------------------------------------------- #
# stage 6.2: exposure and seams
# --------------------------------------------------------------------------- #
def _compensate_exposure(tiles: list[Tile], mode: str) -> list[Tile]:
    """Equalise the gains so the seam does not fall on a brightness step."""
    detail = _cv_attr("detail")
    compensator = detail.ExposureCompensator_createDefault(
        getattr(detail, EXPOSURE_TYPES[mode])
    )
    corners = [corner for _, _, corner in tiles]
    images = [image for image, _, _ in tiles]
    masks = [mask for _, mask, _ in tiles]
    compensator.feed(corners, images, masks)
    out: list[Tile] = []
    for index, (image, mask, corner) in enumerate(
        tqdm(tiles, desc=f"Compensating exposure ({mode})", unit="img")
    ):
        applied = compensator.apply(index, corner, image, mask)
        # Some builds return the corrected image, others edit it in place.
        out.append((applied if isinstance(applied, np.ndarray) else image, mask, corner))
    return out


def _find_seams(tiles: list[Tile], config: SeamConfig) -> list[np.ndarray]:
    """Optimal cut through the overlaps, computed at ``config.scale``."""
    masks = [mask for _, mask, _ in tiles]
    if config.finder == "none":
        return masks

    if config.finder == "graphcut":
        finder = _cv_attr("detail_GraphCutSeamFinder", "detail.GraphCutSeamFinder")(
            "COST_COLOR"
        )
    else:
        finder = _cv_attr("detail_DpSeamFinder", "detail.DpSeamFinder")("COLOR")

    scale = config.scale
    small_images: list[np.ndarray] = []
    small_masks: list[Any] = []
    small_corners: list[tuple[int, int]] = []
    for image, mask, (x, y) in tqdm(
        tiles, desc=f"Preparing seams ({config.finder})", unit="img"
    ):
        size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
        # GraphCutSeamFinder.find insists on float32 images and UMat masks.
        small_images.append(
            cv2.resize(image, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        )
        small_masks.append(cv2.UMat(cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)))
        small_corners.append((round(x * scale), round(y * scale)))

    found = finder.find(small_images, small_corners, small_masks)
    # find() edits the masks in place on most builds and returns them on others.
    resolved = list(found) if found is not None else small_masks

    kernel = np.ones((3, 3), dtype=np.uint8)
    seamed: list[np.ndarray] = []
    for small, full in zip(resolved, masks):
        array = small.get() if isinstance(small, cv2.UMat) else np.asarray(small)
        upscaled = cv2.resize(
            array, (full.shape[1], full.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        # The upscaling opens a crack of up to 1/scale pixels between two neighbouring
        # regions, so dilate by that much to close it; the AND keeps it inside the image.
        upscaled = cv2.dilate(upscaled, kernel, iterations=int(np.ceil(1.0 / scale)))
        seamed.append(cv2.bitwise_and(upscaled, full))
    return seamed


# --------------------------------------------------------------------------- #
# stage 6.3: blending
# --------------------------------------------------------------------------- #
def _blend(
    tiles: list[Tile],
    masks: Sequence[np.ndarray],
    width: int,
    height: int,
    config: BlendConfig,
) -> np.ndarray:
    """Merge the tiles across their seams into the final canvas.

    Linear ignores the seams: it averages the whole overlap of the original tile
    masks, which is what sets it apart from feather, a narrow ramp across the cut.
    """
    if config.method == "linear":
        return _average(tiles, width, height, ramp=True)
    if config.method == "none":
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        for (image, _, (x, y)), mask in tqdm(
            list(zip(tiles, masks)), desc="Blending (overwrite)", unit="img"
        ):
            region = canvas[y : y + image.shape[0], x : x + image.shape[1]]
            np.copyto(region, image, where=(mask > 0)[..., None])
        return canvas

    if config.method == "multiband":
        # A band cannot survive below one pixel, so the pyramid depth is capped
        # by the shortest side of the canvas.
        max_bands = max(1, int(np.log2(max(2, min(width, height)))))
        bands = min(config.bands, max_bands)
        if bands < config.bands:
            logger.warning(
                "blend.bands=%d exceeds the %dx%d canvas; clamped to %d",
                config.bands,
                width,
                height,
                bands,
            )
        blender = _cv_attr("detail_MultiBandBlender", "detail.MultiBandBlender")(0, bands)
    else:
        blender = _cv_attr("detail_FeatherBlender", "detail.FeatherBlender")(
            float(config.sharpness)
        )

    blender.prepare((0, 0, width, height))
    for (image, _, corner), mask in tqdm(
        list(zip(tiles, masks)), desc=f"Blending ({config.method})", unit="img"
    ):
        blender.feed(
            np.ascontiguousarray(image.astype(np.int16)),
            np.ascontiguousarray(mask),
            corner,
        )
    blended, _ = blender.blend(None, None)
    return np.clip(np.asarray(blended), 0, 255).astype(np.uint8)


def _average(
    tiles: list[Tile], width: int, height: int, ramp: bool = False
) -> np.ndarray:
    """Stage 6.4 reference: plain average of the overlaps, so ghosting shows.

    With ``ramp`` every tile weighs by its distance to its own border instead,
    which is linear blending.
    """
    accumulator = np.zeros((height, width, 3), dtype=np.float32)
    counter = np.zeros((height, width), dtype=np.float32)
    for image, mask, (x, y) in tqdm(
        tiles, desc=f"Averaging ({'linear' if ramp else 'naive'})", unit="img"
    ):
        tile_h, tile_w = image.shape[:2]
        if ramp:
            # The zero border makes the tile edge count as border too when the
            # mask fills the whole tile, as it does under a pure translation.
            padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            valid = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
        else:
            valid = (mask > 0).astype(np.float32)
        accumulator[y : y + tile_h, x : x + tile_w] += image.astype(np.float32) * valid[..., None]
        counter[y : y + tile_h, x : x + tile_w] += valid
    np.maximum(counter, 1.0, out=counter)
    return np.clip(accumulator / counter[..., None], 0, 255).astype(np.uint8)


def _build_seam_mask(
    masks: Sequence[np.ndarray],
    corners: Sequence[tuple[int, int]],
    width: int,
    height: int,
) -> np.ndarray:
    """Per pixel source index, 1 based, 0 where the canvas is empty."""
    if len(masks) > 255:
        logger.warning(
            "%d images do not fit a uint8 seam mask; indices past 255 wrap around",
            len(masks),
        )
    canvas = np.zeros((height, width), dtype=np.uint8)
    for index, (mask, (x, y)) in enumerate(zip(masks, corners), start=1):
        region = canvas[y : y + mask.shape[0], x : x + mask.shape[1]]
        region[mask > 0] = np.uint8(index % 256)
    return canvas


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def compose_panorama(
    images: Mapping[str, np.ndarray],
    order: Sequence[str],
    graph: nx.Graph,
    config: ComposeConfig,
) -> ComposeResult:
    """Build the panorama from ``images`` following ``order`` and ``graph``."""
    reference = pick_reference(order, config.reference)
    logger.info("reference frame: '%s' (compose.reference=%s)", reference, config.reference)

    homographies = chain_homographies(order, graph, reference)
    names = [name for name in order if name in homographies and name in images]
    if not names:
        raise ValueError(
            f"no image of the given order could be chained to the reference '{reference}'"
        )
    if len(names) < len(order):
        logger.warning("%d of %d images reach the canvas", len(names), len(order))

    sources = {name: _as_bgr(images[name]) for name in names}

    focal: float | None = None
    if config.projection != "planar" or config.bundle_adjust:
        frame_width = max(image.shape[1] for image in sources.values())
        focal = _resolve_focal(config, graph, frame_width)
    if config.bundle_adjust and focal is None:
        logger.warning("no focal to start from; skipping bundle adjustment")
    elif config.bundle_adjust:
        homographies, focal = _bundle_adjust(
            graph, sources, {name: homographies[name] for name in names}, reference, focal
        )

    tiles: list[Tile] | None = None
    canvas_w = canvas_h = 0
    if config.projection != "planar" and focal is not None:
        try:
            tiles, canvas_w, canvas_h = _warp_rotation(
                sources, names, homographies, config.projection, focal,
                config.canvas_max, config.wave_correct,
            )
        except (cv2.error, RuntimeError, np.linalg.LinAlgError) as exc:
            # The rotation warper is an approximation here: without bundle_adjust
            # the rotations come from the chained homographies alone, so a bad
            # estimate degrades to the planar warp instead of failing.
            logger.warning(
                "the %s warper failed (%s); falling back to the planar projection",
                config.projection,
                exc,
            )
            tiles = None
    if tiles is None:
        tiles, canvas_w, canvas_h = _warp_planar(
            sources, names, homographies, config.canvas_max
        )
    logger.info("canvas: %dx%d px from %d warped images", canvas_w, canvas_h, len(tiles))

    if config.exposure != "none":
        tiles = _compensate_exposure(tiles, config.exposure)
    else:
        logger.info("exposure compensation skipped (compose.exposure=none)")
    naive = _average(tiles, canvas_w, canvas_h) if config.save_naive else None

    seam_masks = _find_seams(tiles, config.seam)
    blends: dict[str, np.ndarray] = {}
    blend_times_ms: dict[str, float] = {}
    for method in dict.fromkeys([config.blend.method, *config.blend.compare]):
        started = time.perf_counter()
        blends[method] = _blend(
            tiles, seam_masks, canvas_w, canvas_h,
            config.blend.model_copy(update={"method": method}),
        )
        blend_times_ms[method] = (time.perf_counter() - started) * 1000.0
        logger.info("blend %s: %.0f ms", method, blend_times_ms[method])
    corners = [corner for _, _, corner in tiles]

    return ComposeResult(
        panorama=blends[config.blend.method],
        naive=naive,
        seam_mask=_build_seam_mask(seam_masks, corners, canvas_w, canvas_h),
        warped_corners=corners,
        global_homographies={name: homographies[name] for name in names},
        blends={method: blends[method] for method in config.blend.compare},
        blend_times_ms={method: blend_times_ms[method] for method in config.blend.compare},
    )
