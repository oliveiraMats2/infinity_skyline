# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP - Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   IC - Institute of Computing            ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀   Computer Science Department              ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · IC · 2026                    ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝

"""Pydantic schemas for the YAML configuration.

Every run is fully described by one config object, so an invalid YAML fails here
with a readable message instead of exploding halfway through a 30 minute run.
Configs compose through the ``extends`` key (recursive merge, child wins).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DetectorName = Literal["sift", "orb", "akaze", "reference_superpoint", "reference_loftr"]
MatcherName = Literal["flann", "brute_force", "reference_superglue", "reference_loftr"]
BlendMethod = Literal["multiband", "feather", "linear"]


class _Base(BaseModel):
    """Forbid unknown keys: a typo in the YAML is an error, not a silent no-op."""

    model_config = ConfigDict(extra="forbid")


class RunConfig(_Base):
    id: str | None = None
    output_root: Path = Path("results")
    artifacts_root: Path = Path("artifacts")
    seed: int = 42
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    overwrite: bool = False
    cache: bool = True


class DataConfig(_Base):
    input_dir: Path = Path("data/processed/360_graus_images")
    intruders_dir: Path | None = None
    extensions: list[str] = Field(default_factory=lambda: [".png"])
    grayscale: bool = True
    max_dimension: int | None = 1600

    @field_validator("extensions")
    @classmethod
    def _normalize(cls, value: list[str]) -> list[str]:
        return [e if e.startswith(".") else f".{e}" for e in (s.lower() for s in value)]


class DetectionConfig(_Base):
    name: DetectorName = "sift"
    params: dict[str, Any] = Field(default_factory=dict)
    repetitions: int = Field(default=10, ge=1)
    warmup: int = Field(default=2, ge=0)


class FlannConfig(_Base):
    float_descriptors: dict[str, Any] = Field(default_factory=dict)
    binary_descriptors: dict[str, Any] = Field(default_factory=dict)
    checks: int = 50


class BruteForceConfig(_Base):
    cross_check: bool = False

    @field_validator("cross_check")
    @classmethod
    def _knn_incompatible(cls, value: bool) -> bool:
        if value:
            raise ValueError(
                "brute_force.cross_check must be false: it is mutually exclusive "
                "with knnMatch(k=2), which the Lowe ratio test requires."
            )
        return value


class MatchingConfig(_Base):
    name: MatcherName = "flann"
    flann: FlannConfig = Field(default_factory=FlannConfig)
    brute_force: BruteForceConfig = Field(default_factory=BruteForceConfig)
    strategy: Literal["all_pairs", "sequential"] = "all_pairs"


class FiltersConfig(_Base):
    lowe_ratio: float = Field(default=0.75, gt=0.0, lt=1.0)
    min_matches: int = Field(default=10, ge=4)


class RansacConfig(_Base):
    method: Literal["RANSAC", "USAC_MAGSAC"] = "USAC_MAGSAC"
    reproj_threshold: float = Field(default=3.0, gt=0.0)
    max_iters: int = Field(default=5000, ge=1)
    confidence: float = Field(default=0.995, gt=0.0, lt=1.0)


class GeometryConfig(_Base):
    model: Literal["homography", "fundamental"] = "homography"
    ransac: RansacConfig = Field(default_factory=RansacConfig)


class GraphConfig(_Base):
    min_inliers: int = Field(default=30, ge=4)
    min_inlier_ratio: float = Field(default=0.25, ge=0.0, le=1.0)
    build_order: bool = True
    reject_isolated: bool = True


class SeamConfig(_Base):
    finder: Literal["graphcut", "dp_color", "none"] = "graphcut"
    scale: float = Field(default=0.25, gt=0.0, le=1.0)


class BlendConfig(_Base):
    method: BlendMethod | Literal["none"] = "multiband"
    bands: int = Field(default=5, ge=1)
    sharpness: float = Field(default=0.02, gt=0.0)
    # One extra panorama per method listed, blended from the same seams, so the
    # techniques are compared on identical geometry. Empty = only `method`.
    compare: list[BlendMethod] = Field(
        default_factory=lambda: ["feather", "linear", "multiband"]
    )


class ComposeConfig(_Base):
    reference: Literal["center", "first"] | int = "center"
    projection: Literal["planar", "cylindrical", "spherical"] = "planar"
    # The focal decides how far apart the frames sit on the cylinder. Estimating it
    # from the homographies is ill conditioned under small rotations, so state it when
    # the camera is known: focal_mm and sensor_width_mm are converted against the
    # working image width. focal_px wins over both. All null falls back to estimation.
    focal_px: float | None = Field(default=None, gt=0.0)
    focal_mm: float | None = Field(default=None, gt=0.0)
    sensor_width_mm: float | None = Field(default=None, gt=0.0)
    # Levels the horizon by sending the accumulated roll back to zero. Only the non
    # planar projections build rotations, so it does nothing under "planar".
    wave_correct: Literal["horiz", "vert", "none"] = "horiz"
    # Refine every rotation and the focal jointly over all graph edges instead of
    # trusting the pairwise chain alone; closes the loop of a 360 degree sweep.
    bundle_adjust: bool = False
    canvas_max: int = Field(default=12000, ge=1)
    # Pixel budget of the canvas. Past it the config fails with a clear error instead
    # of the operating system killing the process, and the whole sweep, for memory.
    canvas_max_megapixels: float = Field(default=40.0, gt=0.0)
    seam: SeamConfig = Field(default_factory=SeamConfig)
    blend: BlendConfig = Field(default_factory=BlendConfig)
    exposure: Literal["none", "gain", "gain_blocks"] = "gain_blocks"
    save_naive: bool = True
    # Also save panorama_equirectangular.jpg: the mosaic on a 2:1 canvas with GPano
    # XMP, which 360 viewers open in 3D. Needs a cylindrical or spherical projection.
    save_equirectangular: bool = True


class ExportConfig(_Base):
    csv: bool = True
    markdown_table: bool = True
    plots: bool = True


class EvaluateConfig(_Base):
    detectors: list[DetectorName] = Field(default_factory=lambda: ["sift", "orb", "akaze"])
    matchers: list[MatcherName] = Field(default_factory=lambda: ["flann", "brute_force"])
    metrics: list[str] = Field(default_factory=list)
    per_detector_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    export: ExportConfig = Field(default_factory=ExportConfig)


class VisualizeKeypoints(_Base):
    enabled: bool = True
    rich: bool = True
    max_draw: int = 500
    color: tuple[int, int, int] = (0, 255, 0)


class VisualizeMatches(_Base):
    enabled: bool = True
    before_after: bool = True
    max_lines: int = 100
    inliers_only: bool = False


class VisualizeGraph(_Base):
    enabled: bool = True
    layout: Literal["spring", "circular", "kamada_kawai"] = "spring"
    annotate_weights: bool = True


class FiguresConfig(_Base):
    dpi: int = Field(default=150, ge=1)
    format: str = "png"

    def path(self, run_dir: Path, module: str, name: str) -> Path:
        """The one place that decides where a figure goes and what it is called.

        ``results/<run_id>/figures/<module>/<name>.<format>``, one subdirectory per
        module of ``visualize/``. The parent is created here so no caller has to.
        """
        directory = run_dir / "figures" / module
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{name}.{self.format}"


class VisualizeProgressive(_Base):
    # The progressive steps show the alignment, not the seam: without graph cut and
    # exposure each step is a plain overlay, and the final panorama keeps both.
    fast: bool = True


class VisualizeConfig(_Base):
    keypoints: VisualizeKeypoints = Field(default_factory=VisualizeKeypoints)
    matches: VisualizeMatches = Field(default_factory=VisualizeMatches)
    graph: VisualizeGraph = Field(default_factory=VisualizeGraph)
    progressive: VisualizeProgressive = Field(default_factory=VisualizeProgressive)
    figures: FiguresConfig = Field(default_factory=FiguresConfig)


class Config(_Base):
    run: RunConfig = Field(default_factory=RunConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    compose: ComposeConfig = Field(default_factory=ComposeConfig)
    evaluate: EvaluateConfig = Field(default_factory=EvaluateConfig)
    visualize: VisualizeConfig = Field(default_factory=VisualizeConfig)

    @model_validator(mode="after")
    def _learned_pairs(self) -> "Config":
        detector, matcher = self.detection.name, self.matching.name
        if matcher == "reference_superglue" and detector != "reference_superpoint":
            raise ValueError("matching.name reference_superglue needs detection.name reference_superpoint")
        if (matcher == "reference_loftr") != (detector == "reference_loftr"):
            raise ValueError(
                "reference_loftr is detector free: set it as both detection.name and matching.name"
            )
        return self

    @model_validator(mode="after")
    def _binary_descriptors_need_lsh(self) -> "Config":
        if self.geometry.model == "fundamental" and self.compose.seam.finder != "none":
            raise ValueError(
                "compose requires geometry.model == 'homography'; a fundamental "
                "matrix does not define the warp needed to build the mosaic."
            )
        return self

    def run_dir(self) -> Path:
        """``results/<run_id>/`` for this config. Caller creates it."""
        assert self.run.id is not None, "run.id must be resolved before run_dir()"
        return self.run.output_root / self.run.id

    def artifact_path(self, name: str) -> Path:
        """``artifacts/<run_id>/<name>.csv``: the one place every table is written to."""
        assert self.run.id is not None, "run.id must be resolved before artifact_path()"
        directory = self.run.artifacts_root / self.run.id
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{name}.csv"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; scalars and lists from ``override`` win outright."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_raw(path: Path, _seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """Read a YAML config, resolving ``extends`` relative to the file itself."""
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"circular 'extends' chain at {path}")
    with path.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}
    parent_ref = data.pop("extends", None)
    if parent_ref is None:
        return data
    parent = load_raw(path.parent / str(parent_ref), _seen | {path})
    return _deep_merge(parent, data)


def load_config(path: Path | str, override: dict[str, Any] | None = None) -> Config:
    """Parse and validate a YAML config into a :class:`Config`.

    ``override`` is merged on top the same way a child merges over ``extends``, so a
    sweep can point every config at another scene without editing the files.
    """
    return Config.model_validate(_deep_merge(load_raw(Path(path)), override or {}))


def dump_config(config: Config, path: Path, extra: dict[str, Any] | None = None) -> None:
    """Write the resolved config plus provenance (OpenCV version, git commit)."""
    payload: dict[str, Any] = yaml.safe_load(config.model_dump_json())
    if extra:
        payload["_provenance"] = extra
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
