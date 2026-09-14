"""Matcher construction from the YAML ``matching`` block.

FLANN needs a different index for each descriptor family: a KD-tree over float
descriptors (SIFT), LSH over binary ones (ORB, AKAZE). The brute force matcher
needs the matching norm for the same reason.
"""

from __future__ import annotations

from typing import Any

import cv2

from ..config import MatchingConfig
from ..logging_setup import get_logger

logger = get_logger(__name__)

#: Used when the YAML leaves the corresponding FLANN block empty.
DEFAULT_KDTREE_INDEX: dict[str, Any] = {"algorithm": 1, "trees": 5}
DEFAULT_LSH_INDEX: dict[str, Any] = {
    "algorithm": 6,
    "table_number": 6,
    "key_size": 12,
    "multi_probe_level": 1,
}


def create_matcher(config: MatchingConfig, binary: bool) -> cv2.DescriptorMatcher:
    """Build the matcher described by ``config`` for the given descriptor family."""
    if config.name == "flann":
        if binary:
            index_params = dict(config.flann.binary_descriptors) or dict(DEFAULT_LSH_INDEX)
        else:
            index_params = dict(config.flann.float_descriptors) or dict(DEFAULT_KDTREE_INDEX)
        search_params: dict[str, Any] = {"checks": config.flann.checks}
        logger.debug("flann index_params=%s search_params=%s", index_params, search_params)
        return cv2.FlannBasedMatcher(index_params, search_params)

    norm = cv2.NORM_HAMMING if binary else cv2.NORM_L2
    # crossCheck stays False: it is mutually exclusive with knnMatch(k=2), which the
    # Lowe ratio test needs. config.py already refuses a true value in the YAML.
    return cv2.BFMatcher(normType=norm, crossCheck=False)
