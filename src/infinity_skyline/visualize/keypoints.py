"""Stage 3.4: draw the detected keypoints over the source image."""

from __future__ import annotations

import cv2
import numpy as np

from ..config import VisualizeConfig
from ..io import KeypointRecord
from ..logging_setup import get_logger

logger = get_logger(__name__)

#: Index of the ``response`` column in :data:`io.KEYPOINT_COLUMNS`.
_RESPONSE_COLUMN: int = 4


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
