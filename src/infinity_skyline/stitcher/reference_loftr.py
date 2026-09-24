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

"""LoFTR (Sun et al., 2021) as ``reference_loftr``, both detector and matcher.

LoFTR is detector free: it matches dense features of the image pair directly, so
its "detector" returns nothing and every keypoint is born from a match (see
``matching.match.keypoint_indices``). kornia holds the pretrained outdoor model.
"""

from __future__ import annotations

from functools import cache
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from ..io import resize_to_max
from .reference_superpoint import device, to_tensor

#: Longest side LoFTR runs at, as in its outdoor evaluation. The coarse stage scores
#: every cell of one image against every cell of the other, so at the full 1600 px
#: working width that matrix alone is about 3 GB and a pair takes 15 to 80 s on MPS.
MAX_DIMENSION: int = 840


class LoFTRDetector:
    """``cv2.Feature2D`` stand in that detects nothing: the keypoints come from matching."""

    def detectAndCompute(
        self, image: np.ndarray, mask: np.ndarray | None
    ) -> tuple[list[cv2.KeyPoint], None]:
        return [], None


@cache
def _model() -> Any:
    from kornia.feature import LoFTR

    return LoFTR(pretrained="outdoor").to(device()).eval()


def _padded(image: np.ndarray) -> np.ndarray:
    """Zero pad bottom and right to multiples of 8, which LoFTR needs; coordinates are unchanged."""
    height, width = image.shape
    return np.pad(image, ((0, -height % 8), (0, -width % 8)))


def match(image0: np.ndarray, image1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Matched ``(M, 2)`` points in each image, their confidence, and the forward time in ms.

    The pair runs at ``MAX_DIMENSION`` and the points are scaled back to the size of
    the images given, the working resolution.
    """
    import torch

    small0, small1 = resize_to_max(image0, MAX_DIMENSION), resize_to_max(image1, MAX_DIMENSION)
    batch = {"image0": to_tensor(_padded(small0)), "image1": to_tensor(_padded(small1))}
    model = _model()  # loading stays out of the timed forward pass
    with torch.inference_mode():
        start = perf_counter()
        output = model(batch)
        confidence = output["confidence"].cpu().numpy()  # the copy waits for the device
        elapsed_ms = (perf_counter() - start) * 1000.0
    # (x, y) scale per axis: rounding the resized sides makes them differ slightly
    scale0 = np.divide(image0.shape[::-1], small0.shape[::-1])
    scale1 = np.divide(image1.shape[::-1], small1.shape[::-1])
    points0 = output["keypoints0"].cpu().numpy() * scale0
    points1 = output["keypoints1"].cpu().numpy() * scale1
    return points0, points1, confidence, elapsed_ms
