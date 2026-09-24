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

"""Stage 3.4: draw the detected keypoints over the source image."""

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
from ..io import KeypointRecord, save_image  # noqa: E402
from ..logging_setup import get_logger  # noqa: E402

logger = get_logger(__name__)

#: Indices of the columns used here, in :data:`io.KEYPOINT_COLUMNS` order.
_SIZE_COLUMN: int = 2
_RESPONSE_COLUMN: int = 4

#: Bins per axis of the spatial density map, over the normalized unit square.
_DENSITY_BINS: int = 50

#: Harris parameters (blockSize, Sobel ksize, k) and the corner threshold as a
#: fraction of the peak response, fixed here: the panel is illustrative only.
_HARRIS_BLOCK: int = 2
_HARRIS_KSIZE: int = 3
_HARRIS_K: float = 0.04
_CORNER_FRACTION: float = 0.01


def to_bgr(image: np.ndarray) -> np.ndarray:
    """Promote a grayscale image to 3 channels so coloured overlays show up."""
    return image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def draw_keypoints(
    image: np.ndarray,
    record: KeypointRecord,
    config: VisualizeConfig,
) -> np.ndarray:
    """Overlay ``record`` keypoints on ``image``, strongest ones first.

    With ``config.keypoints.rich`` each keypoint is drawn with its scale circle
    and orientation stroke, which is the whole point of comparing SIFT against
    ORB and AKAZE visually.
    """
    canvas = to_bgr(image)
    keypoints = record.cv_keypoints()
    max_draw = config.keypoints.max_draw
    if len(keypoints) > max_draw:
        # Keep the strongest by response: a random sample would hide the good ones.
        order = np.argsort(record.keypoints[:, _RESPONSE_COLUMN])[::-1][:max_draw]
        keypoints = [keypoints[int(i)] for i in order]
        logger.debug(
            "%s: drawing %d of %d keypoints", record.image, max_draw, record.n_keypoints
        )
    flags = (
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        if config.keypoints.rich
        else cv2.DRAW_MATCHES_FLAGS_DEFAULT
    )
    return cv2.drawKeypoints(
        canvas,
        keypoints,
        None,
        color=tuple(int(c) for c in config.keypoints.color),
        flags=flags,
    )


def save_detection_figure(
    image: np.ndarray,
    record: KeypointRecord,
    config: VisualizeConfig,
    path: Path,
) -> None:
    """Show the detection pipeline on one image: image, Harris R, corners, keypoints."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    response = cv2.cornerHarris(np.float32(gray), _HARRIS_BLOCK, _HARRIS_KSIZE, _HARRIS_K)
    # Corners are thresholded local maxima: dilation is a 3x3 non-max suppression.
    corners = (response > _CORNER_FRACTION * response.max()) & (
        response == cv2.dilate(response, None)
    )
    ys, xs = np.nonzero(corners)
    rgb = cv2.cvtColor(to_bgr(image), cv2.COLOR_BGR2RGB)

    figure, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("Imagem")
    # Symmetric log: R spans orders of magnitude and its sign separates corners
    # (positive) from edges (negative), so a linear map shows only the top peaks.
    peak = float(np.abs(response).max()) or 1.0
    mesh = axes[1].imshow(
        response,
        cmap="RdBu_r",
        norm=matplotlib.colors.SymLogNorm(linthresh=peak * 1e-4, vmin=-peak, vmax=peak),
    )
    axes[1].set_title("Resposta R (Harris)")
    figure.colorbar(mesh, ax=axes[1], fraction=0.046)
    axes[2].imshow(rgb)
    axes[2].plot(xs, ys, "r+", markersize=4)
    axes[2].set_title(f"Cantos ({len(xs)})")
    if record.n_keypoints:
        axes[3].imshow(cv2.cvtColor(draw_keypoints(image, record, config), cv2.COLOR_BGR2RGB))
    else:
        axes[3].imshow(rgb)
        axes[3].text(
            0.5, 0.5, "sem keypoints (metodo sem detector)", transform=axes[3].transAxes,
            ha="center", color="white", backgroundcolor="black",
        )
    axes[3].set_title(f"Keypoints multiescala ({record.detector}, {record.n_keypoints})")
    for axis in axes:
        axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(path, dpi=config.figures.dpi)
    plt.close(figure)  # closing matters: this runs inside loops


def save_keypoint_figures(
    images: Mapping[str, np.ndarray],
    records: Mapping[str, KeypointRecord],
    config: VisualizeConfig,
    run_dir: Path,
) -> list[Path]:
    """Write one overlay and one detection figure per image, plus the pooled distribution.

    The overlays answer "where did this detector fire on this image"; the pooled
    figure answers "how does this detector behave on the scene as a whole", which
    is what separates SIFT from ORB and AKAZE. Returns every path written.
    """
    paths: list[Path] = []
    for name, record in tqdm(records.items(), desc="fig:keypoints", unit="img"):
        path = config.figures.path(run_dir, "keypoints", f"keypoints_{name}")
        save_image(path, draw_keypoints(images[name], record, config))
        paths.append(path)
        path = config.figures.path(run_dir, "keypoints", f"detection_{name}")
        save_detection_figure(images[name], record, config, path)
        paths.append(path)

    populated = [record for record in records.values() if record.n_keypoints]
    if not populated:
        logger.warning("no keypoints in the scene: skipping the distribution figure")
        return paths

    pooled = np.vstack([record.keypoints for record in populated])
    # Divide by each image's own (width, height) so images of different sizes pool
    # into the same unit square instead of the largest one dominating the map.
    normalized = np.vstack(
        [
            record.keypoints[:, :2] / np.array(record.shape[::-1], dtype=float)
            for record in populated
        ]
    )

    figure, (response, scale, density) = plt.subplots(1, 3, figsize=(15, 4.5))
    response.hist(pooled[:, _RESPONSE_COLUMN], bins=50, color="#4c72b0")
    response.set_title("Resposta dos keypoints")
    response.set_xlabel("response")
    response.set_ylabel("contagem")

    scale.hist(pooled[:, _SIZE_COLUMN], bins=50, color="#55a868")
    scale.set_title("Escala dos keypoints")
    scale.set_xlabel("size (px)")
    # Log count: the scale distribution is dominated by the smallest octave, and on
    # a linear axis the coarse keypoints that carry the wide baseline matches sit at
    # a few counts each and are simply invisible next to a bar of 17000.
    scale.set_yscale("log")
    scale.set_ylabel("contagem (log)")

    counts, _, _ = np.histogram2d(
        normalized[:, 0], normalized[:, 1], bins=_DENSITY_BINS, range=[[0.0, 1.0], [0.0, 1.0]]
    )
    # Transposed and with y running downwards, so the map reads like the image does.
    mesh = density.imshow(counts.T, origin="upper", extent=(0.0, 1.0, 1.0, 0.0), cmap="viridis")
    density.set_title("Densidade espacial")
    density.set_xlabel("x / largura")
    density.set_ylabel("y / altura")
    figure.colorbar(mesh, ax=density, label="keypoints por celula")

    detectors = ", ".join(sorted({record.detector for record in populated}))
    figure.suptitle(
        f"Distribuicao dos keypoints: {detectors} "
        f"({pooled.shape[0]} keypoints em {len(populated)} imagens)"
    )
    figure.tight_layout()
    path = config.figures.path(run_dir, "keypoints", "distribution")
    figure.savefig(path, dpi=config.figures.dpi)
    plt.close(figure)  # closing matters: this runs inside loops
    paths.append(path)
    logger.info("keypoint figures saved: %d files under %s", len(paths), path.parent)
    return paths
