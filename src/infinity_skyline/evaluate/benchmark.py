"""Detector x matcher benchmark over every image pair of the scene.

One run of the whole pipeline (detect, match, estimate) per combination, with the
rest of the config held fixed, producing one long format row per
(pair, detector, matcher). ``success_rate`` is not a per row quantity: it is the
fraction of pairs that got a homography, so it only appears in the aggregated
summary written to ``metrics.md``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from tqdm.auto import tqdm

from ..config import Config
from ..detection.detect import detect_directory
from ..geometry.homography import estimate_all
from ..io import KeypointRecord, MatchRecord, list_images, pair_stem
from ..logging_setup import get_logger
from ..matching.match import build_pairs, match_all
from .metrics import inlier_ratio, reprojection_rmse, spatial_dispersion

logger = get_logger(__name__)

#: Columns identifying a row of the long table.
KEY_COLUMNS: tuple[str, ...] = ("pair", "query", "train", "detector", "matcher")

#: Per row metrics, in table order. ``success_rate`` is aggregate only.
METRIC_COLUMNS: tuple[str, ...] = (
    "n_keypoints",
    "detect_time_ms",
    "detect_time_std_ms",
    "descriptor_bytes",
    "spatial_dispersion",
    "raw_matches",
    "filtered_matches",
    "match_time_ms",
    "n_inliers",
    "inlier_ratio",
    "reprojection_rmse",
)


def _mean(values: Sequence[float]) -> float:
    """Mean that ignores nan and returns nan when nothing is left."""
    clean = [v for v in values if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else float("nan")


def _variant(config: Config, detector: str, matcher: str) -> Config:
    """Copy of the config with only the detector and matcher swapped."""
    variant = config.model_copy(deep=True)
    variant.detection.name = detector  # type: ignore[assignment]
    # Per detector overrides win over the shared base params.
    variant.detection.params = {
        **config.detection.params,
        **config.evaluate.per_detector_params.get(detector, {}),
    }
    variant.matching.name = matcher  # type: ignore[assignment]
    return variant


def _row(
    query: str,
    train: str,
    record: MatchRecord,
    keypoints: Mapping[str, KeypointRecord],
    detector: str,
    matcher: str,
) -> dict[str, Any]:
    """One long format row: pair level metrics averaged over its two images."""
    pair_records = [keypoints[name] for name in (query, train) if name in keypoints]
    return {
        "pair": pair_stem(query, train),
        "query": query,
        "train": train,
        "detector": detector,
        "matcher": matcher,
        "n_keypoints": _mean([float(r.n_keypoints) for r in pair_records]),
        "detect_time_ms": _mean([r.detect_time_ms for r in pair_records]),
        "detect_time_std_ms": _mean([r.detect_time_std_ms for r in pair_records]),
        # Descriptor width is a constant of the detector, so take it, not an average.
        "descriptor_bytes": max((r.descriptor_bytes for r in pair_records), default=0),
        "spatial_dispersion": _mean(
            [spatial_dispersion(r.keypoints, r.shape) for r in pair_records]
        ),
        "raw_matches": int(record.n_raw),
        "filtered_matches": int(record.n_filtered),
        "match_time_ms": record.match_time_ms,
        "n_inliers": int(record.n_inliers),
        "inlier_ratio": inlier_ratio(record),
        "reprojection_rmse": reprojection_rmse(record),
        "success": bool(record.homography is not None),
    }


def _selected_metrics(config: Config) -> list[str]:
    """Metric columns requested by the config, defaulting to all of them."""
    wanted = list(config.evaluate.metrics)
    if not wanted:
        return list(METRIC_COLUMNS)
    unknown = [m for m in wanted if m not in METRIC_COLUMNS and m != "success_rate"]
    if unknown:
        logger.warning("ignoring unknown evaluate.metrics entries: %s", ", ".join(unknown))
    return [m for m in METRIC_COLUMNS if m in set(wanted)]


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the long table per detector x matcher, adding ``success_rate``."""
    metric_columns = [c for c in frame.columns if c in METRIC_COLUMNS]
    grouped = frame.groupby(["detector", "matcher"], as_index=False, sort=False)
    summary = grouped[metric_columns].mean()
    summary.insert(2, "pairs", grouped.size()["size"].to_numpy())
    summary.insert(3, "success_rate", grouped["success"].mean()["success"].to_numpy())
    return summary


def run_benchmark(config: Config) -> pd.DataFrame:
    """Run every detector x matcher combination and return the long metric table.

    Writes ``results/<run_id>/metrics.csv`` (the long table) and
    ``results/<run_id>/metrics.md`` (the aggregated summary) when the matching
    ``config.evaluate.export`` flags are set. Plots are the CLI's business.
    """
    run_dir: Path = config.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(config.data.input_dir, config.data.extensions)
    if not images:
        raise FileNotFoundError(f"no images with {config.data.extensions} in {config.data.input_dir}")
    pairs = build_pairs([p.stem for p in images], config.matching.strategy)
    logger.info("benchmark over %d images, %d pairs", len(images), len(pairs))

    combinations = [
        (detector, matcher)
        for detector in config.evaluate.detectors
        for matcher in config.evaluate.matchers
    ]
    rows: list[dict[str, Any]] = []
    outer = tqdm(combinations, desc="benchmark", unit="combo", leave=True)
    for detector, matcher in outer:
        outer.set_postfix(detector=detector, matcher=matcher)
        logger.info("start combination: detector=%s matcher=%s", detector, matcher)
        variant = _variant(config, detector, matcher)
        work_dir = run_dir / "benchmark" / f"{detector}_{matcher}"

        keypoints = detect_directory(variant, images, work_dir / "keypoints", detector=detector)
        matches = match_all(keypoints, variant, work_dir / "matches")
        matches = estimate_all(keypoints, matches, variant)

        combination_rows = [
            _row(query, train, matches[(query, train)], keypoints, detector, matcher)
            for query, train in pairs
            if (query, train) in matches
        ]
        rows.extend(combination_rows)
        estimated = sum(1 for row in combination_rows if row["success"])
        logger.info(
            "done combination: detector=%s matcher=%s rows=%d homographies=%d/%d",
            detector,
            matcher,
            len(combination_rows),
            estimated,
            len(pairs),
        )

    metric_columns = _selected_metrics(config)
    full = pd.DataFrame(rows, columns=list(KEY_COLUMNS) + metric_columns + ["success"])
    summary = summarize(full)
    long_table = full.drop(columns=["success"])

    if config.evaluate.export.csv:
        long_table.to_csv(run_dir / "metrics.csv", index=False)
        logger.info("wrote %s", run_dir / "metrics.csv")
    if config.evaluate.export.markdown_table:
        (run_dir / "metrics.md").write_text(summary.to_markdown(index=False), encoding="utf-8")
        logger.info("wrote %s", run_dir / "metrics.md")

    return long_table
