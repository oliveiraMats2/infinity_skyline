"""Detector construction from YAML params.

One YAML block is shared by every detector, so most keys are meaningless for at
least one of them. Each constructor therefore has an explicit allowlist and
anything outside it is dropped with a warning naming the key and the detector.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Mapping

import cv2

from ..logging_setup import get_logger
from ..stitcher import LoFTRDetector, SuperPoint

logger = get_logger(__name__)

#: Keyword arguments each OpenCV constructor actually accepts.
ALLOWED_PARAMS: dict[str, frozenset[str]] = {
    "sift": frozenset(
        {"nfeatures", "nOctaveLayers", "contrastThreshold", "edgeThreshold", "sigma"}
    ),
    "orb": frozenset(
        {
            "nfeatures",
            "scaleFactor",
            "nlevels",
            "edgeThreshold",
            "firstLevel",
            "WTA_K",
            "scoreType",
            "patchSize",
            "fastThreshold",
        }
    ),
    "akaze": frozenset(
        {
            "descriptor_type",
            "descriptor_size",
            "descriptor_channels",
            "threshold",
            "nOctaves",
            "nOctaveLayers",
            "diffusivity",
        }
    ),
    "reference_superpoint": frozenset({"nfeatures"}),  # max keypoints kept by score
    "reference_loftr": frozenset(),
}

#: OpenCV class name behind each detector key.
_CV_NAMES: dict[str, str] = {"sift": "SIFT", "orb": "ORB", "akaze": "AKAZE"}


def _construct(cv_name: str, **kwargs: Any) -> cv2.Feature2D:
    """Resolve ``<cv_name>_create`` across OpenCV builds and call it."""
    contrib = getattr(cv2, "xfeatures2d", None)
    factory: Callable[..., cv2.Feature2D] | None = (
        getattr(cv2, f"{cv_name}_create", None)
        or getattr(getattr(cv2, cv_name, None), "create", None)
        or getattr(contrib, f"{cv_name}_create", None)
    )
    if factory is None:
        raise RuntimeError(
            f"OpenCV {cv2.__version__} exposes no {cv_name} constructor; "
            "install opencv-contrib-python"
        )
    try:
        return factory(**kwargs)
    except TypeError as error:
        # The allowlist matched the documented names but this build disagrees.
        raise TypeError(
            f"{cv_name} constructor rejected keywords {sorted(kwargs)}: {error}"
        ) from error


#: Factories keyed by the config ``detection.name``.
DETECTORS: dict[str, Callable[..., cv2.Feature2D]] = {
    **{key: partial(_construct, cv_name) for key, cv_name in _CV_NAMES.items()},
    # learned references with the same detectAndCompute, see stitcher/
    "reference_superpoint": SuperPoint,
    "reference_loftr": LoFTRDetector,
}

#: Detectors producing binary descriptors, so Hamming distance and uint8 buffers.
BINARY_DETECTORS: frozenset[str] = frozenset({"orb", "akaze"})


def create_detector(name: str, params: Mapping[str, Any]) -> cv2.Feature2D:
    """Build a detector, dropping params its OpenCV constructor cannot take."""
    key = name.lower()
    if key not in DETECTORS:
        raise KeyError(f"unknown detector {name!r}; available: {sorted(DETECTORS)}")

    allowed = ALLOWED_PARAMS[key]
    kwargs: dict[str, Any] = {k: v for k, v in params.items() if k in allowed}
    for dropped in sorted(set(params) - allowed):
        if key == "akaze" and dropped == "nfeatures":
            # Honest mapping of the shared cap: AKAZE has no native limit, so the
            # fairness fix happens in detect.py (truncate by response).
            logger.warning(
                "akaze: dropping 'nfeatures'=%r, AKAZE has no keypoint cap; "
                "detect.py truncates to the strongest ones by response instead",
                params[dropped],
            )
        else:
            logger.warning(
                "%s: dropping %r=%r, the %s constructor does not accept it",
                key,
                dropped,
                params[dropped],
                _CV_NAMES.get(key, key),
            )
    logger.debug("%s created with %s", _CV_NAMES.get(key, key), kwargs)
    return DETECTORS[key](**kwargs)


def is_binary(name: str) -> bool:
    """True when the descriptors are bit strings (ORB, AKAZE)."""
    return name.lower() in BINARY_DETECTORS


def norm_type(name: str) -> int:
    """Distance to compare this detector's descriptors with."""
    return cv2.NORM_HAMMING if is_binary(name) else cv2.NORM_L2
