"""Typer entry point.

Two modes, as the project brief asks for:

* generation  (``convert``, ``detect``, ``match``, ``visualize``, ``panorama``,
  ``pipeline``): produce artifacts under ``results/<run_id>/``.
* evaluation   (``evaluate``): hold everything else fixed, sweep detectors and
  matchers, and write ``metrics.csv`` comparing them.

Every subcommand takes the same ``--config`` YAML and nothing else, so a run is
reproducible from the file alone.
"""

from __future__ import annotations

import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import typer
from tqdm.auto import tqdm

from . import __version__
from .compose import compose_panorama
from .config import Config, load_config, dump_config
from .detection.detect import detect_directory
from .evaluate.benchmark import run_benchmark
from .geometry.graph import build_graph, infer_order, largest_component, rejected_images
from .geometry.homography import estimate_all
from .io import KeypointRecord, MatchRecord, list_images, load_image, save_image
from .logging_setup import get_logger, setup_logging
from .matching.match import match_all
from .tools.convert_to_png import convert_directory
from .visualize.graph import draw_graph
from .visualize.keypoints import draw_keypoints
from .visualize.matches import draw_before_after
from .visualize.panorama import draw_deghosting_comparison, draw_progressive
from .visualize.report import plot_benchmark

app = typer.Typer(
    add_completion=False,
    help="Feature detection, matching, geometry and panorama composition benchmark.",
)

logger = get_logger(__name__)

ConfigOption = Annotated[
    Path,
    typer.Option("--config", "-c", exists=True, dir_okay=False, help="YAML config file."),
]


# --------------------------------------------------------------------------- #
# run bootstrap
# --------------------------------------------------------------------------- #
def _git_commit() -> str:
    """Short commit hash, or 'unknown' outside a git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _prepare(config_path: Path) -> tuple[Config, Path]:
    """Load the config, resolve the run directory, start logging, save provenance."""
    config = load_config(config_path)

    if config.run.id is None:
        slug = f"{config.detection.name}_{config.matching.name}"
        config.run.id = f"{datetime.now():%Y%m%d_%H%M%S}_{slug}"

    run_dir = config.run_dir()
    if run_dir.exists() and not config.run.overwrite:
        if any(run_dir.iterdir()) and not config.run.cache:
            raise typer.BadParameter(
                f"run directory {run_dir} already exists; set run.overwrite or run.cache "
                f"in the YAML, or pick another run.id"
            )
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(config.run.log_level, run_dir / "run.log")
    random.seed(config.run.seed)
    np.random.seed(config.run.seed)
    cv2.setRNGSeed(config.run.seed)

    dump_config(
        config,
        run_dir / "config.resolved.yaml",
        extra={
            "infinity_skyline_version": __version__,
            "opencv_version": cv2.__version__,
            "git_commit": _git_commit(),
            "config_source": str(config_path),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
    )
    logger.info("run %s | opencv %s | seed %d", config.run.id, cv2.__version__, config.run.seed)
    return config, run_dir


def _scene_images(config: Config) -> list[Path]:
    """Scene images, plus the intruder set when one is configured (stage 4)."""
    images = list_images(config.data.input_dir, config.data.extensions)
    if not images:
        raise typer.BadParameter(
            f"no {config.data.extensions} images in {config.data.input_dir}; "
            f"run `convert` first if the source is Canon RAW"
        )
    if config.data.intruders_dir is not None:
        intruders = list_images(config.data.intruders_dir, config.data.extensions)
        logger.info("adding %d intruder images from %s", len(intruders), config.data.intruders_dir)
        images = images + intruders
    return images


def _load_color(images: list[Path], config: Config) -> dict[str, np.ndarray]:
    """BGR images keyed by stem, at the configured working resolution."""
    return {
        path.stem: load_image(path, grayscale=False, max_dimension=config.data.max_dimension)
        for path in tqdm(images, desc="load", unit="img")
    }


def _stage_geometry(
    config: Config, run_dir: Path
) -> tuple[
    list[Path],
    dict[str, KeypointRecord],
    dict[tuple[str, str], MatchRecord],
]:
    """Detect, match and run RANSAC. Shared by match/visualize/panorama/pipeline."""
    images = _scene_images(config)
    keypoints = detect_directory(config, images, run_dir / "keypoints")
    matches = match_all(keypoints, config, run_dir / "matches")
    matches = estimate_all(keypoints, matches, config, run_dir / "matches")
    return images, keypoints, matches


# --------------------------------------------------------------------------- #
# generation mode
# --------------------------------------------------------------------------- #
@app.command()
def convert(
    config_path: ConfigOption,
    source: Annotated[
        Path, typer.Option("--source", exists=True, file_okay=False, help="RAW input folder.")
    ],
    dest: Annotated[
        Path | None, typer.Option("--dest", file_okay=False, help="PNG output folder.")
    ] = None,
) -> None:
    """Convert a scene folder (Canon RAW, JPEG, TIFF) to lossless PNG."""
    config = load_config(config_path)
    setup_logging(config.run.log_level)
    destination = dest if dest is not None else Path("data/processed") / source.name
    written = convert_directory(
        source, destination, max_dimension=config.data.max_dimension, overwrite=config.run.overwrite
    )
    logger.info("wrote %d PNG files to %s", len(written), destination)


@app.command()
def detect(config_path: ConfigOption) -> None:
    """Detect and describe keypoints for every image of the scene."""
    config, run_dir = _prepare(config_path)
    records = detect_directory(config, _scene_images(config), run_dir / "keypoints")
    total = sum(r.n_keypoints for r in records.values())
    logger.info("%d keypoints over %d images", total, len(records))


@app.command()
def match(config_path: ConfigOption) -> None:
    """Match descriptors pairwise and estimate the homography of each pair."""
    config, run_dir = _prepare(config_path)
    _, _, matches = _stage_geometry(config, run_dir)
    ok = sum(1 for m in matches.values() if m.meta.get("success"))
    logger.info("%d/%d pairs with a homography", ok, len(matches))


@app.command()
def visualize(config_path: ConfigOption) -> None:
    """Render keypoint, match and connectivity graph figures."""
    config, run_dir = _prepare(config_path)
    images, keypoints, matches = _stage_geometry(config, run_dir)
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    color = _load_color(images, config)

    if config.visualize.keypoints.enabled:
        for name, record in tqdm(keypoints.items(), desc="fig:keypoints", unit="img"):
            save_image(
                figures / f"keypoints_{name}.{config.visualize.figures.format}",
                draw_keypoints(color[name], record, config.visualize),
            )

    if config.visualize.matches.enabled:
        for (a, b), record in tqdm(matches.items(), desc="fig:matches", unit="pair"):
            if record.n_filtered == 0:
                continue
            raw, filtered = draw_before_after(
                color[a], keypoints[a], color[b], keypoints[b], record, config.visualize
            )
            stem = f"matches_{a}__{b}"
            save_image(figures / f"{stem}_raw.{config.visualize.figures.format}", raw)
            save_image(figures / f"{stem}_filtered.{config.visualize.figures.format}", filtered)

    if config.visualize.graph.enabled:
        names = sorted(keypoints)
        graph = build_graph(matches, names, config.graph)
        draw_graph(
            graph,
            rejected_images(graph, config.graph),
            config.visualize,
            figures / f"graph.{config.visualize.figures.format}",
        )
    logger.info("figures written to %s", figures)


@app.command()
def panorama(config_path: ConfigOption) -> Path:
    """Stage 3 end to end: an unordered folder in, one panorama PNG out."""
    config, run_dir = _prepare(config_path)
    images, keypoints, matches = _stage_geometry(config, run_dir)

    names = sorted(keypoints)
    graph = build_graph(matches, names, config.graph)
    rejected = rejected_images(graph, config.graph)
    if rejected:
        logger.warning("rejected as not belonging to the scene: %s", ", ".join(rejected))

    component = largest_component(graph)
    order = infer_order(graph, component)
    if len(order) < 2:
        raise typer.BadParameter(
            "fewer than two connected images; loosen graph.min_inliers or graph.min_inlier_ratio"
        )
    logger.info("inferred order (%d images): %s", len(order), " -> ".join(order))

    color = _load_color([p for p in images if p.stem in set(order)], config)
    result = compose_panorama(color, order, graph, config.compose)

    out_dir = run_dir / "panorama"
    save_image(out_dir / "panorama.png", result.panorama)
    if result.naive is not None:
        save_image(out_dir / "naive.png", result.naive)
    if result.seam_mask is not None:
        save_image(out_dir / "seam_mask.png", result.seam_mask)
    logger.info("panorama %dx%d written to %s", *result.panorama.shape[1::-1], out_dir)

    if result.naive is not None:
        draw_deghosting_comparison(
            result.panorama, result.naive, run_dir / "figures" / "deghosting.png"
        )
    return run_dir


@app.command()
def sweep(
    config_paths: Annotated[
        list[Path],
        typer.Argument(exists=True, dir_okay=False, help="Config files, one panorama each."),
    ],
) -> None:
    """Compose one panorama per config given, then report what each one produced.

    Orchestration only: every config goes through the same `panorama` command above.
    A variant that fails does not stop the rest, because some of them exist to
    demonstrate a limit rather than to succeed.
    """
    rows: list[tuple[str, str, str]] = []
    for config_path in tqdm(config_paths, desc="sweep", unit="config"):
        name = config_path.stem
        try:
            run_dir = panorama(config_path)
        except Exception as exc:  # a sweep must survive its own failures
            rows.append((name, "FALHOU", f"{type(exc).__name__}: {exc}"))
            logger.warning("%s failed: %s", name, exc)
            continue
        image = cv2.imread(str(run_dir / "panorama" / "panorama.png"))
        shape = "missing" if image is None else f"{image.shape[1]}x{image.shape[0]}"
        rows.append((name, "ok", f"{shape}  {run_dir}"))

    width = max(len(name) for name, _, _ in rows)
    typer.echo("")
    for name, status, detail in rows:
        typer.echo(f"{name:<{width}}  {status:<7}  {detail}")
    ok = sum(1 for _, status, _ in rows if status == "ok")
    typer.echo(f"\n{ok} of {len(rows)} composed a panorama.")


@app.command()
def pipeline(config_path: ConfigOption) -> None:
    """Full generation run: detect, match, graph, figures, progressive and panorama."""
    config, run_dir = _prepare(config_path)
    images, keypoints, matches = _stage_geometry(config, run_dir)

    names = sorted(keypoints)
    graph = build_graph(matches, names, config.graph)
    rejected = rejected_images(graph, config.graph)
    order = infer_order(graph, largest_component(graph))
    logger.info("order: %s | rejected: %s", " -> ".join(order), rejected or "none")

    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    color = _load_color(images, config)

    if config.visualize.keypoints.enabled:
        for name, record in tqdm(keypoints.items(), desc="fig:keypoints", unit="img"):
            save_image(figures / f"keypoints_{name}.png", draw_keypoints(color[name], record, config.visualize))
    if config.visualize.graph.enabled:
        draw_graph(graph, rejected, config.visualize, figures / "graph.png")

    if len(order) >= 2:
        scene = {name: color[name] for name in order}
        draw_progressive(scene, order, graph, config.compose, figures / "progressive")
        result = compose_panorama(scene, order, graph, config.compose)
        out_dir = run_dir / "panorama"
        save_image(out_dir / "panorama.png", result.panorama)
        if result.naive is not None:
            save_image(out_dir / "naive.png", result.naive)
            draw_deghosting_comparison(result.panorama, result.naive, figures / "deghosting.png")
        if result.seam_mask is not None:
            save_image(out_dir / "seam_mask.png", result.seam_mask)
    logger.info("pipeline finished: %s", run_dir)


# --------------------------------------------------------------------------- #
# evaluation mode
# --------------------------------------------------------------------------- #
@app.command()
def evaluate(config_path: ConfigOption) -> None:
    """Sweep detectors and matchers over the same pairs and write metrics.csv."""
    config, run_dir = _prepare(config_path)
    frame = run_benchmark(config)
    logger.info("%d benchmark rows", len(frame))
    if config.evaluate.export.plots:
        paths = plot_benchmark(frame, run_dir / "figures", config.visualize.figures.dpi)
        logger.info("%d benchmark plots written", len(paths))


if __name__ == "__main__":  # pragma: no cover
    app()
