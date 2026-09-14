"""Keypoint detection and description."""

from __future__ import annotations

from .detect import detect_directory, detect_image
from .registry import BINARY_DETECTORS, DETECTORS, create_detector, is_binary, norm_type

__all__ = [
    "detect_directory",
    "detect_image",
    "BINARY_DETECTORS",
    "DETECTORS",
    "create_detector",
    "is_binary",
    "norm_type",
]
