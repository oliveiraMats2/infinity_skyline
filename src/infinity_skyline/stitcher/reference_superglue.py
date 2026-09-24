"""SuperGlue (Sarlin et al., 2020) as the matcher ``reference_superglue``.

The Hugging Face model re-detects SuperPoint keypoints on both images and matches
them with its graph network, so it reads the image pair, not our descriptors. The
coordinates come back at full working resolution, no resize.
"""

from __future__ import annotations

from functools import cache
from time import perf_counter
from typing import Any

import numpy as np

from .reference_superpoint import device, to_tensor

#: Hugging Face checkpoint, trained on outdoor scenes like ours.
MODEL_ID: str = "magic-leap-community/superglue_outdoor"

#: Minimum matching score, the default of the original SuperGlue release.
#: The Hugging Face checkpoint ships 0.0, which keeps every mutual match.
MATCH_THRESHOLD: float = 0.2


@cache
def _model() -> Any:
    from transformers import SuperGlueForKeypointMatching

    model = SuperGlueForKeypointMatching.from_pretrained(MODEL_ID, matching_threshold=MATCH_THRESHOLD)
    return model.to(device()).eval()


def match(image0: np.ndarray, image1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Matched ``(M, 2)`` points in each image, their confidence, and the forward time in ms.

    Both images must share one size: the model batches them as a single tensor.
    """
    import torch

    height, width = image0.shape[:2]
    pixels = torch.stack([to_tensor(image0), to_tensor(image1)], dim=1)  # (1, 2, 1, H, W)
    model = _model()  # loading stays out of the timed forward pass
    with torch.inference_mode():
        start = perf_counter()
        output = model(pixels)
        matches = output.matches[0, 0].cpu()  # the copy waits for the device
        elapsed_ms = (perf_counter() - start) * 1000.0
    keypoints = output.keypoints[0].cpu() * torch.tensor([width, height])  # (2, N, 2)
    valid = output.mask[0, 0].bool().cpu() & (matches > -1)
    points0 = keypoints[0][valid].round().numpy()
    points1 = keypoints[1][matches[valid]].round().numpy()
    confidence = output.matching_scores[0, 0].cpu()[valid].numpy()
    return points0, points1, confidence, elapsed_ms
