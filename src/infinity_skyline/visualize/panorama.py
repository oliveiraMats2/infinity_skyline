"""Stages 5.4 and 6.4: progressive alignment frames and the deghosting close up."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import cv2
import networkx as nx
import numpy as np
from tqdm.auto import tqdm

from ..compose import compose_panorama
from ..config import ComposeConfig
from ..io import save_image
from ..logging_setup import get_logger

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
    out_dir: Path,
) -> list[Path]:
    """Compose the first ``k`` images for ``k = 2..len(order)``, one PNG each."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for k in tqdm(
        range(2, len(order) + 1), desc="progressive panorama", unit="image"
    ):
        result = compose_panorama(images, list(order[:k]), graph, config)
        path = out_dir / f"progressive_{k:02d}.png"
        save_image(path, result.panorama)
        paths.append(path)
    logger.info("saved %d progressive frames to %s", len(paths), out_dir)
    return paths


def _to_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _worst_region(
    panorama: np.ndarray, naive: np.ndarray, window: int
) -> tuple[int, int, int, int]:
    """Window with the largest mean absolute difference: that is where ghosts live."""
    difference = cv2.absdiff(_to_gray(panorama), _to_gray(naive)).astype(np.float32)
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


def draw_deghosting_comparison(
    panorama: np.ndarray,
    naive: np.ndarray,
    out_path: Path,
    crop: tuple[int, int, int, int] | None = None,
) -> None:
    """Same region of the naive average and of the seam blended result, magnified."""
    height = min(panorama.shape[0], naive.shape[0])
    width = min(panorama.shape[1], naive.shape[1])
    panorama = panorama[:height, :width]
    naive = naive[:height, :width]

    if crop is None:
        window = int(np.clip(min(height, width) // 8, 32, min(height, width)))
        crop = _worst_region(panorama, naive, window)
        logger.info("deghosting crop chosen automatically at %s", crop)
    x, y, w, h = crop
    x, y = int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))
    w, h = int(min(w, width - x)), int(min(h, height - y))
    box = (x, y, w, h)

    zoom = max(_MIN_ZOOM, int(round(_TARGET_CROP_PIXELS / max(w, h))))
    left = _labelled_crop(naive, box, "sem deghosting", zoom)
    right = _labelled_crop(panorama, box, "com deghosting", zoom)
    separator = np.full((left.shape[0], _SEPARATOR_WIDTH, 3), 255, dtype=left.dtype)
    save_image(out_path, np.hstack([left, separator, right]))
    logger.info("deghosting comparison saved to %s (zoom %dx)", out_path, zoom)
