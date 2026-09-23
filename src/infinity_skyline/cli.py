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

import hashlib
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Annotated, Mapping, Sequence

import cv2
import networkx as nx
import numpy as np
import typer
from tqdm.auto import tqdm

from . import __version__
from .compose import compose_panorama, stitch_with_opencv
from .config import Config, load_config, dump_config
from .detection.detect import detect_directory
from .evaluate.benchmark import run_benchmark
from .geometry.graph import build_graph, infer_order, largest_component, rejected_images
from .geometry.homography import estimate_all
from .io import KeypointRecord, MatchRecord, list_images, load_image, save_image
from .logging_setup import get_logger, setup_logging
from .matching.match import match_all
from .tools.convert_to_png import convert_directory
from .visualize.graph import save_graph_figures
from .visualize.keypoints import save_keypoint_figures
from .visualize.matches import save_match_figures
from .visualize.panorama import draw_progressive, save_panorama_figures
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
        # The config file names the run: one config, one results directory, and
        # re-running the same experiment lands on top of its own output instead of
        # scattering a new timestamped folder every time.
        config.run.id = config_path.stem

    run_dir = config.run_dir()
    # Re-running never blocks and never deletes: artifacts are written over one by one,
    # so a file another stage produced and this one does not is simply left alone.
    if run_dir.exists() and any(run_dir.iterdir()) and not config.run.overwrite:
        logger.debug("writing over the existing artifacts in %s", run_dir)
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


def _stitcher_reference(config: Config, images: Mapping[str, np.ndarray]) -> Path:
    """Ensure the cv2.Stitcher reference for ``images`` exists, and return its path.

    There is one file, shared by every run, under ``results/_reference/``. cv2.Stitcher
    reads none of this config, running its own features, camera estimation and bundle
    adjustment, so it depends on the frames and their resolution and on nothing else.
    None of the variation axes changes the frames, so computing it per run would stitch
    the same mosaic again for every config and store another copy of it.

    The key is the frame names, not the scene, because the frame set does move: an axis
    that changes the graph, such as a detector or a threshold, can drop an image from
    the main component, and then the reference is genuinely a different mosaic.
    """
    names = sorted(images)
    digest = hashlib.sha1(
        ("|".join(names) + f"@{config.data.max_dimension}").encode()
    ).hexdigest()[:10]
    path = (
        config.run.output_root
        / "_reference"
        / f"{config.data.input_dir.name}_{len(names)}img_{digest}.png"
    )
    if path.exists():
        logger.info("cv2.Stitcher reference already at %s", path)
        return path
    save_image(path, stitch_with_opencv([images[name] for name in names]))
    logger.info("cv2.Stitcher reference written to %s", path)
    return path


def _write_figures(
    config: Config,
    run_dir: Path,
    color: Mapping[str, np.ndarray],
    keypoints: Mapping[str, KeypointRecord],
    matches: Mapping[tuple[str, str], MatchRecord],
    graph: nx.Graph,
    rejected: Sequence[str],
) -> list[Path]:
    """Every partial figure of a generation run, in one place instead of four.

    Each module of ``visualize/`` writes into its own subdirectory and picks its own
    file names through ``config.visualize.figures.path``, so no command here decides
    where a figure lands or what extension it gets.
    """
    written: list[Path] = []
    if config.visualize.keypoints.enabled:
        written += save_keypoint_figures(color, keypoints, config.visualize, run_dir)
    if config.visualize.matches.enabled:
        written += save_match_figures(color, keypoints, matches, config.visualize, run_dir)
    if config.visualize.graph.enabled:
        written += save_graph_figures(graph, rejected, config.visualize, run_dir)
    logger.info("%d figures written under %s", len(written), run_dir / "figures")
    return written


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
    max_dimension: Annotated[
        int | None,
        typer.Option("--max-dimension", help="Longest side in px; overrides data.max_dimension."),
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Rewrite PNGs that already exist.")
    ] = False,
) -> None:
    """Convert any image folder (Canon RAW, JPEG, TIFF, PNG) to lossless PNG.

    Recurses into subdirectories and mirrors the tree under ``--dest``. The two
    flags exist so converting a one off folder never means editing the YAML.
    """
    config = load_config(config_path)
    setup_logging(config.run.log_level)
    destination = dest if dest is not None else Path("data/processed") / source.name
    written = convert_directory(
        source,
        destination,
        max_dimension=config.data.max_dimension if max_dimension is None else max_dimension,
        overwrite=overwrite or config.run.overwrite,
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
    graph = build_graph(matches, sorted(keypoints), config.graph)
    _write_figures(
        config,
        run_dir,
        _load_color(images, config),
        keypoints,
        matches,
        graph,
        rejected_images(graph, config.graph),
    )


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
        save_panorama_figures(
            result.panorama, result.naive, config.visualize.figures, run_dir
        )
    try:
        _stitcher_reference(config, color)
    except (RuntimeError, cv2.error) as exc:  # its failure is not ours
        logger.warning("cv2.Stitcher reference not produced: %s", exc)
    return run_dir


@app.command()
def sweep(
    config_paths: Annotated[
        list[Path],
        typer.Argument(exists=True, dir_okay=False, help="Config files, one panorama each."),
    ],
) -> None:
    """Run every config given end to end, then report what each one produced.

    Orchestration only: every config goes through the same `pipeline` command below,
    so each run gets its partial figures (keypoints, matches, graph, panorama) as well
    as the mosaic. Use `panorama` directly when the mosaic is all you want: it is the
    same composition without the figures, and much faster over many configs.

    A variant that fails does not stop the rest, because some of them exist to
    demonstrate a limit rather than to succeed.
    """
    rows: list[tuple[str, str, str]] = []
    for config_path in tqdm(config_paths, desc="sweep", unit="config"):
        name = config_path.stem
        try:
            run_dir = pipeline(config_path)
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
def baseline(config_path: ConfigOption) -> Path:
    """Compose the same scene with cv2.Stitcher, as a reference to evaluate against.

    Shares nothing with our pipeline: OpenCV runs its own features, camera estimation,
    bundle adjustment and wave correction. A mosaic that is gapless and internally
    consistent can still be wrong, and this is what shows it.

    The result is one file under `results/_reference/`, not one per run, because it
    does not depend on this config. The log line says where it landed. Compare it
    against the `panorama.png` of whichever run you are judging.
    """
    config, run_dir = _prepare(config_path)
    color = _load_color(_scene_images(config), config)
    _stitcher_reference(config, color)
    return run_dir


@app.command()
def pipeline(config_path: ConfigOption) -> Path:
    """Full generation run: detect, match, graph, figures, progressive and panorama."""
    config, run_dir = _prepare(config_path)
    images, keypoints, matches = _stage_geometry(config, run_dir)

    names = sorted(keypoints)
    graph = build_graph(matches, names, config.graph)
    rejected = rejected_images(graph, config.graph)
    order = infer_order(graph, largest_component(graph))
    logger.info("order: %s | rejected: %s", " -> ".join(order), rejected or "none")

    color = _load_color(images, config)
    _write_figures(config, run_dir, color, keypoints, matches, graph, rejected)

    if len(order) >= 2:
        scene = {name: color[name] for name in order}
        draw_progressive(scene, order, graph, config.compose, config.visualize.figures, run_dir)
        result = compose_panorama(scene, order, graph, config.compose)
        out_dir = run_dir / "panorama"
        save_image(out_dir / "panorama.png", result.panorama)
        if result.naive is not None:
            save_image(out_dir / "naive.png", result.naive)
            save_panorama_figures(
                result.panorama, result.naive, config.visualize.figures, run_dir
            )
        if result.seam_mask is not None:
            save_image(out_dir / "seam_mask.png", result.seam_mask)
        try:
            _stitcher_reference(config, scene)
        except (RuntimeError, cv2.error) as exc:  # its failure is not ours
            logger.warning("cv2.Stitcher reference not produced: %s", exc)
    logger.info("pipeline finished: %s", run_dir)
    return run_dir


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
        paths = plot_benchmark(frame, config.visualize.figures, run_dir)
        logger.info("%d benchmark plots written", len(paths))


if __name__ == "__main__":  # pragma: no cover
    app()
