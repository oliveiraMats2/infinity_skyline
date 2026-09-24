"""Descriptor matching for one pair and for a whole set of images.

Only the ``knnMatch`` call is timed: the ratio test and the bookkeeping around it
are the same for every backend, so including them would blur the comparison
between FLANN and brute force.

The learned references (``reference_superglue``, ``reference_loftr``) match the image
pair itself instead of descriptors. Their matched coordinates are turned into
keypoint indices, appending the points the detector did not return, and only the
model forward pass is timed. They skip the Lowe test: they have their own thresholds.
"""

from __future__ import annotations

from functools import cache
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import cv2
import numpy as np
from tqdm.auto import tqdm

from ..config import Config
from ..io import KeypointRecord, MatchRecord, load_image, load_matches, pair_stem, save_matches
from ..logging_setup import get_logger
from ..stitcher import loftr_match, superglue_match
from .filters import lowe_ratio_test
from .registry import create_matcher

logger = get_logger(__name__)

#: Shape of an empty match table, so callers can index it without a size check.
EMPTY_PAIRS: tuple[int, int] = (0, 2)

#: Pair matchers that read the two images, keyed by the config ``matching.name``.
LEARNED_MATCHERS = {"reference_superglue": superglue_match, "reference_loftr": loftr_match}

#: A learned match this close to an existing keypoint, in pixels, reuses it.
SNAP_PX: float = 1.0


def build_pairs(names: Sequence[str], strategy: str) -> list[tuple[str, str]]:
    """Image pairs to match, in the order given by ``names``."""
    if strategy == "all_pairs":
        return list(combinations(names, 2))
    if strategy == "sequential":
        return [(a, b) for a, b in zip(names, names[1:])]
    raise ValueError(f"unknown matching strategy: {strategy!r}")


def _matcher_ready(descriptors: np.ndarray, binary: bool) -> np.ndarray:
    """Cast to the only layout the backends accept.

    FLANN KDTREE reads float32 and FLANN LSH reads contiguous uint8; NORM_HAMMING has
    the same uint8 requirement. Records loaded from ``.npz`` can arrive as a view or a
    widened dtype, so the cast is done defensively on every call.
    """
    return np.ascontiguousarray(descriptors, dtype=np.uint8 if binary else np.float32)


def _empty_record(query: KeypointRecord, train: KeypointRecord, reason: str) -> MatchRecord:
    return MatchRecord(
        query=query.image,
        train=train.image,
        pairs=np.zeros(EMPTY_PAIRS, dtype=np.int32),
        distances=np.zeros(0, dtype=np.float32),
        n_raw=0,
        times_ms=np.zeros(0, dtype=np.float64),
        meta={"skipped": reason},
    )


def keypoint_indices(record: KeypointRecord, points: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """Row of ``record.keypoints`` for each matched point, appending the ones it lacks.

    A point within ``SNAP_PX`` of an existing keypoint reuses it (SuperGlue re-detects
    the SuperPoint keypoints); any other becomes a new row with response = confidence,
    so LoFTR, which detects nothing up front, builds its keypoints from its matches.
    Mutates ``record``.
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    indices = np.full(len(points), -1, dtype=np.int32)
    if record.n_keypoints and len(points):
        nearest = cv2.BFMatcher(cv2.NORM_L2).match(points, np.ascontiguousarray(record.keypoints[:, :2]))
        for m in nearest:
            if m.distance <= SNAP_PX:
                indices[m.queryIdx] = m.trainIdx

    new = indices < 0
    rows = np.zeros((int(new.sum()), record.keypoints.shape[1]), dtype=np.float32)
    rows[:, :2] = points[new]
    rows[:, 2] = 8.0  # size
    rows[:, 3] = -1.0  # angle: none
    rows[:, 4] = np.asarray(confidence, dtype=np.float32)[new]  # response
    rows[:, 6] = -1.0  # class_id
    indices[new] = record.n_keypoints + np.arange(len(rows), dtype=np.int32)
    record.keypoints = np.vstack([record.keypoints, rows])
    return indices


def _match_learned(
    query: KeypointRecord, train: KeypointRecord, images: tuple[np.ndarray, np.ndarray], config: Config
) -> MatchRecord:
    """Run the learned pair matcher and index its points into the two records."""
    points0, points1, confidence, elapsed_ms = LEARNED_MATCHERS[config.matching.name](*images)
    pairs = np.stack(
        [keypoint_indices(query, points0, confidence), keypoint_indices(train, points1, confidence)],
        axis=1,
    )
    return MatchRecord(
        query=query.image,
        train=train.image,
        pairs=pairs.astype(np.int32),
        distances=(1.0 - confidence).astype(np.float32),
        n_raw=len(confidence),
        times_ms=np.asarray([elapsed_ms], dtype=np.float64),
        meta={"matcher": config.matching.name, "binary": False},
    )


def match_pair(
    query: KeypointRecord,
    train: KeypointRecord,
    config: Config,
    images: tuple[np.ndarray, np.ndarray] | None = None,
) -> MatchRecord:
    """knnMatch(k=2) plus the Lowe ratio test, or the learned matcher on ``images``."""
    if config.matching.name in LEARNED_MATCHERS:
        if images is None:
            raise ValueError(f"matching.name {config.matching.name!r} reads the image pair: pass images")
        return _flag_weak(_match_learned(query, train, images, config), config)

    if query.descriptors.size == 0 or train.descriptors.size == 0:
        logger.warning("no descriptors for pair %s / %s", query.image, train.image)
        return _empty_record(query, train, "no descriptors")

    # dtype is the source of truth for the descriptor family, and it avoids importing
    # the detection registry from here.
    binary: bool = query.descriptors.dtype == np.uint8
    matcher: cv2.DescriptorMatcher = create_matcher(config.matching, binary)
    query_desc = _matcher_ready(query.descriptors, binary)
    train_desc = _matcher_ready(train.descriptors, binary)

    start = perf_counter()
    knn: Sequence[Sequence[cv2.DMatch]] = matcher.knnMatch(query_desc, train_desc, k=2)
    elapsed_ms = (perf_counter() - start) * 1000.0

    good: list[cv2.DMatch] = lowe_ratio_test(knn, config.filters.lowe_ratio)
    if good:
        pairs = np.asarray([(m.queryIdx, m.trainIdx) for m in good], dtype=np.int32)
        distances = np.asarray([m.distance for m in good], dtype=np.float32)
    else:
        pairs = np.zeros(EMPTY_PAIRS, dtype=np.int32)
        distances = np.zeros(0, dtype=np.float32)

    record = MatchRecord(
        query=query.image,
        train=train.image,
        pairs=pairs,
        distances=distances,
        n_raw=len(knn),
        times_ms=np.asarray([elapsed_ms], dtype=np.float64),
        meta={
            "matcher": config.matching.name,
            "binary": bool(binary),
            "lowe_ratio": config.filters.lowe_ratio,
        },
    )
    return _flag_weak(record, config)


def _flag_weak(record: MatchRecord, config: Config) -> MatchRecord:
    if record.n_filtered < config.filters.min_matches:
        # Flagged, not dropped: geometry and the graph decide what a weak pair means.
        record.meta["below_min_matches"] = True
        logger.debug(
            "pair %s / %s kept %d matches, below min_matches=%d",
            record.query,
            record.train,
            record.n_filtered,
            config.filters.min_matches,
        )
    return record


def match_all(
    records: Mapping[str, KeypointRecord],
    config: Config,
    out_dir: Path,
    images: Mapping[str, Path] | None = None,
) -> dict[tuple[str, str], MatchRecord]:
    """Match every pair selected by ``config.matching.strategy``, with optional cache.

    ``images`` maps image stem to file and is required by the learned matchers, which
    append keypoints to ``records`` in place: the caller re-saves them afterwards.
    """
    learned: bool = config.matching.name in LEARNED_MATCHERS
    if learned and images is None:
        raise ValueError(
            f"matching.name {config.matching.name!r} reads the images: pass images= to match_all"
        )
    load = cache(
        lambda stem: load_image(images[stem], grayscale=True, max_dimension=config.data.max_dimension)
    )
    names: list[str] = list(records)
    pairs = build_pairs(names, config.matching.strategy)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[tuple[str, str], MatchRecord] = {}
    for query_name, train_name in tqdm(pairs, desc="match", unit="pair"):
        path = out_dir / f"{pair_stem(query_name, train_name)}.npz"
        if config.run.cache and path.exists():
            results[(query_name, train_name)] = load_matches(path)
            continue
        pair_images = (load(query_name), load(train_name)) if learned else None
        record = match_pair(records[query_name], records[train_name], config, pair_images)
        # Always persist: run.cache controls reuse, not whether the artifact exists.
        save_matches(path, record)
        results[(query_name, train_name)] = record
    return results
