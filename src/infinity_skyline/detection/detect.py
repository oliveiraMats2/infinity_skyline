"""Timed keypoint detection and description.

Each image is processed ``warmup + repetitions`` times: the warmup runs absorb the
lazy allocations inside OpenCV, the timed runs go to ``record.times_ms``. Every run
yields the same keypoints, so only the last result is kept.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from tqdm.auto import tqdm

from ..config import Config
from ..io import KeypointRecord, keypoints_to_array, load_image, load_keypoints, save_keypoints
from ..logging_setup import get_logger
from .registry import create_detector, is_binary

logger = get_logger(__name__)


def detect_image(
    image: np.ndarray,
    name: str,
    params: Mapping[str, Any],
    *,
    repetitions: int = 1,
    warmup: int = 0,
    image_name: str = "",
    max_keypoints: int | None = None,
) -> KeypointRecord:
    """Detect and describe one image, timing ``repetitions`` runs in milliseconds."""
    detector = create_detector(name, params)

    for _ in range(max(0, warmup)):
        detector.detectAndCompute(image, None)

    times: list[float] = []
    keypoints: Sequence[cv2.KeyPoint] = ()
    descriptors: np.ndarray | None = None
    for _ in range(max(1, repetitions)):
        start = time.perf_counter_ns()
        keypoints, descriptors = detector.detectAndCompute(image, None)
        times.append((time.perf_counter_ns() - start) / 1e6)

    if descriptors is None:
        descriptors = np.zeros(
            (0, 0), dtype=np.uint8 if is_binary(name) else np.float32
        )

    n_detected: int = len(keypoints)
    truncated: bool = max_keypoints is not None and n_detected > max_keypoints
    if truncated:
        # AKAZE has no native cap: keep the strongest responses so the timing and
        # keypoint counts stay comparable with the capped detectors.
        responses = np.fromiter((k.response for k in keypoints), dtype=np.float64, count=n_detected)
        order = np.argsort(responses)[::-1][: int(max_keypoints or 0)]
        keypoints = [keypoints[int(i)] for i in order]
        descriptors = descriptors[order]
        logger.debug(
            "%s on %s: truncated %d keypoints to the %d strongest",
            name,
            image_name or "<array>",
            n_detected,
            len(keypoints),
        )

    return KeypointRecord(
        image=image_name,
        detector=name.lower(),
        keypoints=keypoints_to_array(keypoints),
        descriptors=descriptors,
        shape=(int(image.shape[0]), int(image.shape[1])),
        times_ms=np.asarray(times, dtype=np.float64),
        meta={"truncated": truncated, "n_detected": n_detected},
    )


def detect_directory(
    config: Config,
    images: Sequence[Path],
    out_dir: Path,
    *,
    detector: str | None = None,
) -> dict[str, KeypointRecord]:
    """Detect over a list of images, keyed by image stem, cached as ``.npz``."""
    name: str = (detector or config.detection.name).lower()
    # Per detector overrides exist for the benchmark; otherwise the shared block.
    params: Mapping[str, Any] = config.evaluate.per_detector_params.get(
        name, config.detection.params
    )
    max_keypoints: int | None = params.get("nfeatures")

    out_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, KeypointRecord] = {}
    for path in tqdm(images, desc=f"detect[{name}]", unit="img"):
        stem: str = path.stem
        cache_path: Path = out_dir / f"{stem}__{name}.npz"
        if config.run.cache and cache_path.exists():
            logger.debug("cache hit for %s, loading %s", stem, cache_path)
            records[stem] = load_keypoints(cache_path)
            continue
        image = load_image(
            path,
            grayscale=config.data.grayscale,
            max_dimension=config.data.max_dimension,
        )
        record = detect_image(
            image,
            name,
            params,
            repetitions=config.detection.repetitions,
            warmup=config.detection.warmup,
            image_name=stem,
            max_keypoints=max_keypoints,
        )
        save_keypoints(cache_path, record)
        records[stem] = record
    return records
