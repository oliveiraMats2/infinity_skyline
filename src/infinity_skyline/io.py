"""Loading and persistence of images, keypoints and match records.

``cv2.KeyPoint`` is not picklable in a stable way across OpenCV versions, so it is
stored as a plain ``(N, 7)`` float32 array and rebuilt on load. Everything lands in
``.npz`` with a JSON ``meta`` blob alongside the arrays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from .logging_setup import get_logger

logger = get_logger(__name__)

#: Extensions that only ``rawpy`` can decode (OpenCV and Pillow cannot).
RAW_EXTENSIONS: frozenset[str] = frozenset({".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf"})

#: Column order of the serialized keypoint array.
KEYPOINT_COLUMNS: tuple[str, ...] = ("x", "y", "size", "angle", "response", "octave", "class_id")


# --------------------------------------------------------------------------- #
# images
# --------------------------------------------------------------------------- #
def list_images(
    directory: Path, extensions: Sequence[str], *, recursive: bool = False
) -> list[Path]:
    """Sorted listing, non-recursive by default: one directory is one scene.

    ``recursive=True`` walks subdirectories as well, for the callers that mirror a
    whole tree (the PNG converter) instead of reading a single flat scene.
    """
    wanted = {e.lower() for e in extensions}
    entries = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(p for p in entries if p.is_file() and p.suffix.lower() in wanted)


def _read_raw(path: Path) -> np.ndarray:
    """Decode a camera RAW file to an 8-bit BGR array."""
    import rawpy  # imported lazily: only RAW inputs pay the import cost

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(use_camera_wb=True, output_bps=8, no_auto_bright=False)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def resize_to_max(image: np.ndarray, max_dimension: int | None) -> np.ndarray:
    """Scale so the longest side is ``max_dimension``. Never upscales."""
    if max_dimension is None:
        return image
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dimension:
        return image
    scale = max_dimension / longest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def load_image(
    path: Path,
    grayscale: bool = False,
    max_dimension: int | None = None,
) -> np.ndarray:
    """Read any supported image (RAW included) as BGR, or single-channel if asked."""
    if path.suffix.lower() in RAW_EXTENSIONS:
        image = _read_raw(path)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not decode image: {path}")
    image = resize_to_max(image, max_dimension)
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    """Write an image, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"cv2.imwrite failed for {path}")


# --------------------------------------------------------------------------- #
# keypoints
# --------------------------------------------------------------------------- #
def keypoints_to_array(keypoints: Iterable[cv2.KeyPoint]) -> np.ndarray:
    """``(N, 7)`` float32 array in :data:`KEYPOINT_COLUMNS` order."""
    rows = [
        (k.pt[0], k.pt[1], k.size, k.angle, k.response, float(k.octave), float(k.class_id))
        for k in keypoints
    ]
    if not rows:
        return np.zeros((0, len(KEYPOINT_COLUMNS)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def array_to_keypoints(array: np.ndarray) -> list[cv2.KeyPoint]:
    """Inverse of :func:`keypoints_to_array`."""
    return [
        cv2.KeyPoint(
            x=float(row[0]),
            y=float(row[1]),
            size=float(row[2]),
            angle=float(row[3]),
            response=float(row[4]),
            octave=int(row[5]),
            class_id=int(row[6]),
        )
        for row in array
    ]


@dataclass(slots=True)
class KeypointRecord:
    """Detector output for one image, plus its timing."""

    image: str
    detector: str
    keypoints: np.ndarray
    descriptors: np.ndarray
    shape: tuple[int, int]
    times_ms: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_keypoints(self) -> int:
        return int(self.keypoints.shape[0])

    @property
    def detect_time_ms(self) -> float:
        """Mean detection time over the timed repetitions."""
        return float(self.times_ms.mean()) if self.times_ms.size else float("nan")

    @property
    def detect_time_std_ms(self) -> float:
        return float(self.times_ms.std(ddof=1)) if self.times_ms.size > 1 else 0.0

    @property
    def descriptor_bytes(self) -> int:
        """Bytes per descriptor: SIFT 512, ORB 32, AKAZE 61."""
        if self.descriptors.size == 0:
            return 0
        return int(self.descriptors.shape[1] * self.descriptors.dtype.itemsize)

    def cv_keypoints(self) -> list[cv2.KeyPoint]:
        return array_to_keypoints(self.keypoints)


def save_keypoints(path: Path, record: KeypointRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        keypoints=record.keypoints,
        descriptors=record.descriptors,
        times_ms=record.times_ms,
        meta=np.array(
            json.dumps(
                {
                    "image": record.image,
                    "detector": record.detector,
                    "shape": list(record.shape),
                    **record.meta,
                }
            )
        ),
    )


def load_keypoints(path: Path) -> KeypointRecord:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        return KeypointRecord(
            image=meta.pop("image"),
            detector=meta.pop("detector"),
            keypoints=data["keypoints"],
            descriptors=data["descriptors"],
            shape=tuple(meta.pop("shape")),  # type: ignore[arg-type]
            times_ms=data["times_ms"],
            meta=meta,
        )


# --------------------------------------------------------------------------- #
# matches
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class MatchRecord:
    """Matches between an ordered pair of images, after filtering and RANSAC."""

    query: str
    train: str
    pairs: np.ndarray  # (M, 2) int32: index in query, index in train
    distances: np.ndarray  # (M,) float32
    n_raw: int = 0
    inlier_mask: np.ndarray | None = None  # (M,) bool, set by geometry.homography
    homography: np.ndarray | None = None  # (3, 3) float64
    times_ms: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_filtered(self) -> int:
        return int(self.pairs.shape[0])

    @property
    def n_inliers(self) -> int:
        return int(self.inlier_mask.sum()) if self.inlier_mask is not None else 0

    @property
    def inlier_ratio(self) -> float:
        """Headline metric: inliers over filtered matches."""
        return self.n_inliers / self.n_filtered if self.n_filtered else 0.0

    @property
    def match_time_ms(self) -> float:
        return float(self.times_ms.mean()) if self.times_ms.size else float("nan")

    def cv_matches(self) -> list[cv2.DMatch]:
        """Rebuild ``cv2.DMatch`` objects for ``cv2.drawMatches``."""
        return [
            cv2.DMatch(_queryIdx=int(q), _trainIdx=int(t), _distance=float(d))
            for (q, t), d in zip(self.pairs, self.distances)
        ]


def save_matches(path: Path, record: MatchRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "pairs": record.pairs,
        "distances": record.distances,
        "times_ms": record.times_ms,
        "meta": np.array(
            json.dumps(
                {
                    "query": record.query,
                    "train": record.train,
                    "n_raw": record.n_raw,
                    **record.meta,
                }
            )
        ),
    }
    if record.inlier_mask is not None:
        arrays["inlier_mask"] = record.inlier_mask
    if record.homography is not None:
        arrays["homography"] = record.homography
    np.savez_compressed(path, **arrays)


def load_matches(path: Path) -> MatchRecord:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        return MatchRecord(
            query=meta.pop("query"),
            train=meta.pop("train"),
            pairs=data["pairs"],
            distances=data["distances"],
            n_raw=meta.pop("n_raw", 0),
            inlier_mask=data["inlier_mask"] if "inlier_mask" in data else None,
            homography=data["homography"] if "homography" in data else None,
            times_ms=data["times_ms"],
            meta=meta,
        )


def pair_stem(query: Path | str, train: Path | str) -> str:
    """Stable filename stem for a pair artifact: ``<query>__<train>``."""
    return f"{Path(query).stem}__{Path(train).stem}"
