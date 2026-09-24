"""SuperPoint (DeTone et al., 2018) as the detector ``reference_superpoint``.

The Hugging Face port is wrapped behind OpenCV's ``detectAndCompute``, so
``detect_image`` times it exactly like SIFT, ORB and AKAZE. It runs at full working
resolution, no resize. torch is imported lazily: the base pipeline never needs it.
"""

from __future__ import annotations

import os
from functools import cache
from typing import Any

import cv2
import numpy as np

#: Hugging Face checkpoint.
MODEL_ID: str = "magic-leap-community/superpoint"

# Ops MPS does not implement run on the CPU instead of raising.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def to_tensor(image: np.ndarray) -> Any:
    """Grayscale uint8 ``(H, W)`` as the ``(1, 1, H, W)`` float in [0, 1] the models read."""
    import torch

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return torch.from_numpy(image).float().div(255.0)[None, None].to(device())


@cache
def _model(max_keypoints: int) -> Any:
    from transformers import SuperPointForKeypointDetection

    model = SuperPointForKeypointDetection.from_pretrained(MODEL_ID, max_keypoints=max_keypoints)
    return model.to(device()).eval()


class SuperPoint:
    """``cv2.Feature2D`` stand in: keypoints with response = score, float32 (N, 256) descriptors."""

    def __init__(self, nfeatures: int = -1) -> None:
        self.nfeatures = nfeatures  # -1 keeps every keypoint above the model threshold

    def detectAndCompute(
        self, image: np.ndarray, mask: np.ndarray | None
    ) -> tuple[list[cv2.KeyPoint], np.ndarray]:
        import torch

        height, width = image.shape[:2]
        with torch.inference_mode():
            output = _model(self.nfeatures)(to_tensor(image))
        valid = output.mask[0].bool().cpu()
        # the model returns (x, y) relative to the image size; back to pixels
        points = (output.keypoints[0].cpu()[valid] * torch.tensor([width, height])).round().numpy()
        scores = output.scores[0].cpu()[valid].numpy()
        descriptors = output.descriptors[0].cpu()[valid].numpy().astype(np.float32)
        keypoints = [
            cv2.KeyPoint(x=float(x), y=float(y), size=8.0, response=float(s))
            for (x, y), s in zip(points, scores)
        ]
        return keypoints, descriptors
