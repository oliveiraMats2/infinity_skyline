"""Descriptor matching for one pair and for a whole set of images.

Only the ``knnMatch`` call is timed: the ratio test and the bookkeeping around it
are the same for every backend, so including them would blur the comparison
between FLANN and brute force.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import cv2
import numpy as np
from tqdm.auto import tqdm

from ..config import Config
from ..io import KeypointRecord, MatchRecord, load_matches, pair_stem, save_matches
from ..logging_setup import get_logger
from .filters import lowe_ratio_test
from .registry import create_matcher

logger = get_logger(__name__)

#: Shape of an empty match table, so callers can index it without a size check.
EMPTY_PAIRS: tuple[int, int] = (0, 2)


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


def match_pair(query: KeypointRecord, train: KeypointRecord, config: Config) -> MatchRecord:
    """knnMatch(k=2) plus the Lowe ratio test between two keypoint records."""
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
    if record.n_filtered < config.filters.min_matches:
        # Flagged, not dropped: geometry and the graph decide what a weak pair means.
        record.meta["below_min_matches"] = True
        logger.debug(
            "pair %s / %s kept %d matches, below min_matches=%d",
            query.image,
            train.image,
            record.n_filtered,
            config.filters.min_matches,
        )
    return record


def match_all(
    records: Mapping[str, KeypointRecord],
    config: Config,
    out_dir: Path,
) -> dict[tuple[str, str], MatchRecord]:
    """Match every pair selected by ``config.matching.strategy``, with optional cache."""
    names: list[str] = list(records)
    pairs = build_pairs(names, config.matching.strategy)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[tuple[str, str], MatchRecord] = {}
    for query_name, train_name in tqdm(pairs, desc="match", unit="pair"):
        path = out_dir / f"{pair_stem(query_name, train_name)}.npz"
        if config.run.cache and path.exists():
            results[(query_name, train_name)] = load_matches(path)
            continue
        record = match_pair(records[query_name], records[train_name], config)
        # Always persist: run.cache controls reuse, not whether the artifact exists.
        save_matches(path, record)
        results[(query_name, train_name)] = record
    return results
