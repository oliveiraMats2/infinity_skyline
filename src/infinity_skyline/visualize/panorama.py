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

"""Stages 5.4 and 6.4: progressive alignment frames and the deghosting close up."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")  # headless backend, must be set before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from ..compose import compose_panorama  # noqa: E402
from ..config import ComposeConfig, FiguresConfig  # noqa: E402
from ..io import save_image  # noqa: E402
from ..logging_setup import get_logger  # noqa: E402

logger = get_logger(__name__)

_LABEL_HEIGHT: int = 36
_MIN_ZOOM: int = 2
_TARGET_CROP_PIXELS: int = 600
_SEPARATOR_WIDTH: int = 8


def draw_progressive(
    images: Mapping[str, np.ndarray],
    order: Sequence[str],
    graph: nx.Graph,
    config: ComposeConfig,
    figures: FiguresConfig,
    run_dir: Path,
    fast: bool = True,
) -> list[Path]:
    """Compose the first ``k`` images for ``k = 2..len(order)``, one frame each.

    ``fast`` drops exposure, seam finding and the extra blends from every step: they
    are about 75 percent of a run and the steps only have to show the alignment.
    """
    if fast:
        config = config.model_copy(
            update={
                "exposure": "none",
                "save_naive": False,
                "seam": config.seam.model_copy(update={"finder": "none"}),
                "blend": config.blend.model_copy(update={"method": "none", "compare": []}),
            }
        )
    paths: list[Path] = []
    for k in tqdm(
        range(2, len(order) + 1), desc="progressive panorama", unit="image"
    ):
        result = compose_panorama(images, list(order[:k]), graph, config)
        path = figures.path(run_dir, "panorama", f"progressive_{k:02d}")
        save_image(path, result.panorama)
        paths.append(path)
    logger.info("saved %d progressive frames to %s", len(paths), run_dir)
    return paths


def _to_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _worst_region(difference: np.ndarray, window: int) -> tuple[int, int, int, int]:
    """Window with the largest mean absolute difference: that is where ghosts live.

    Takes the difference map rather than the two images because the caller plots the
    very same map, and one ``absdiff`` over a panorama is not worth paying twice.
    """
    averaged = cv2.blur(difference, (window, window))
    _, _, _, (cx, cy) = cv2.minMaxLoc(averaged)
    height, width = difference.shape[:2]
    x = int(np.clip(cx - window // 2, 0, max(0, width - window)))
    y = int(np.clip(cy - window // 2, 0, max(0, height - window)))
    return x, y, min(window, width), min(window, height)


def _labelled_crop(image: np.ndarray, box: tuple[int, int, int, int], text: str, zoom: int) -> np.ndarray:
    x, y, w, h = box
    patch = image[y : y + h, x : x + w]
    if patch.ndim == 2:
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    patch = cv2.resize(patch, (w * zoom, h * zoom), interpolation=cv2.INTER_NEAREST)
    patch = cv2.copyMakeBorder(
        patch, _LABEL_HEIGHT, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    cv2.putText(
        patch, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
    )
    return patch


def save_panorama_figures(
    panorama: np.ndarray,
    naive: np.ndarray,
    figures: FiguresConfig,
    run_dir: Path,
    crop: tuple[int, int, int, int] | None = None,
) -> list[Path]:
    """The deghosting close up plus the difference map that justifies its crop.

    The two read together: the map says where the naive average and the blended
    result disagree over the whole panorama, the rectangle says which of those
    disagreements the magnified comparison is showing.
    """
    height = min(panorama.shape[0], naive.shape[0])
    width = min(panorama.shape[1], naive.shape[1])
    panorama = panorama[:height, :width]
    naive = naive[:height, :width]
    # computed once and used twice: to pick the crop, and as the plotted map
    difference = cv2.absdiff(_to_gray(panorama), _to_gray(naive)).astype(np.float32)

    if crop is None:
        window = int(np.clip(min(height, width) // 8, 32, min(height, width)))
        crop = _worst_region(difference, window)
        logger.info("deghosting crop chosen automatically at %s", crop)
    x, y, w, h = crop
    x, y = int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))
    w, h = int(min(w, width - x)), int(min(h, height - y))
    box = (x, y, w, h)

    zoom = max(_MIN_ZOOM, int(round(_TARGET_CROP_PIXELS / max(w, h))))
    left = _labelled_crop(naive, box, "without deghosting", zoom)
    right = _labelled_crop(panorama, box, "with deghosting", zoom)
    separator = np.full((left.shape[0], _SEPARATOR_WIDTH, 3), 255, dtype=left.dtype)
    comparison_path = figures.path(run_dir, "panorama", "deghosting")
    save_image(comparison_path, np.hstack([left, separator, right]))
    logger.info("deghosting comparison saved to %s (zoom %dx)", comparison_path, zoom)

    figure, axes = plt.subplots(figsize=(11, 6))
    heat = axes.imshow(difference, cmap="inferno")
    figure.colorbar(heat, ax=axes, label="|panorama - naive average|")
    axes.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#39ff14", linewidth=2))
    axes.set_title("Absolute difference between the panorama and the naive average")
    axes.set_axis_off()
    figure.tight_layout()
    difference_path = figures.path(run_dir, "panorama", "difference")
    figure.savefig(difference_path, dpi=figures.dpi)
    plt.close(figure)
    return [comparison_path, difference_path]
