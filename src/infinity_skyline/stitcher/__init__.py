"""Reference methods the own pipeline is compared against, each named ``reference_<method>``.

They plug into the same detection and matching stages as SIFT, ORB, AKAZE, FLANN
and brute force. The learned ones need the optional ``learned`` extra (torch).
"""

from __future__ import annotations

from .reference_cv2_stitcher import stitch_with_opencv
from .reference_loftr import LoFTRDetector
from .reference_loftr import match as loftr_match
from .reference_superglue import match as superglue_match
from .reference_superpoint import SuperPoint

__all__ = ["LoFTRDetector", "SuperPoint", "loftr_match", "stitch_with_opencv", "superglue_match"]
