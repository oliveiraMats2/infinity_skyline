# infinity-skyline

A benchmark and panorama pipeline over skyline photo sets. It detects local features
(SIFT, ORB, AKAZE), matches them across images, estimates pairwise homographies with
RANSAC, builds a connectivity graph of the scene, infers the sweep order, and composes
the images into a single mosaic with exposure compensation, seam finding and multi-band
blending.

Every run is described by one YAML file validated against pydantic schemas, so a typo
fails immediately instead of halfway through a long run. Artifacts land in
`results/<run_id>/`.

## Installation

```bash
uv venv .venv --prompt infinity_skyline && source .venv/bin/activate
uv sync --extra dev
```

Requires Python 3.11 or newer. `--extra dev` adds `pytest`; plain `uv sync` installs
only the runtime dependencies and will *remove* `pytest` if it is already there.

`opencv-contrib-python` is pinned below 5.0 on purpose: the 5.x Python bindings dropped
`AKAZE`, `KAZE` and `BRISK`, and AKAZE is one of the three detectors under comparison.

## Input data

The source images are Canon RAW `.CR2` files. Neither OpenCV nor Pillow can decode
them, which is why `rawpy` is a hard dependency: it is used by `io.load_image` and by
the `convert` subcommand, which writes lossless PNG copies that the rest of the
pipeline reads.

One directory is one scene, and the pipeline runs over one scene at a time:

| Directory | Files | What it is |
|---|---|---|
| `data/high_bright/` | 8 `.CR2` | The same view, over-exposed. **The default scene** |
| `data/low_bright/` | 7 `.CR2` | The same view, under-exposed |
| `data/360_graus_images/` | 25 `.CR2` | The panoramic 360 degree sweep |

`data.input_dir` points at `high_bright`. Swap it for another row to change scene.

`low_bright` and `high_bright` are exposure brackets of the same view, and that is
exactly why `compose.exposure: gain_blocks` matters: when frames of one mosaic arrive
at different brightness levels, a single global gain cannot fix vignetting or the
brightness variation inside a frame, while block-wise gain compensation can.

Pointing `data.intruders_dir` at a different scene directory injects foreign images
into the run and exercises the rejection stage: they should fall outside the largest
connected component of the graph.

`convert` is the one subcommand that does not take its input path from the config.
`data.input_dir` names the PNG directory that every other stage reads, so it points at
the *output* of the conversion, not at the RAW source. The source is passed explicitly:

```bash
python main.py convert --config configs/default.yaml --source data/high_bright
```

`--source` is required and takes the scene directory holding the `.CR2` files.
`--dest` is optional and defaults to `data/processed/<source directory name>`, which is
the layout the rest of the pipeline expects. `run.overwrite` in the config decides
whether an already converted PNG is rewritten or skipped.

## CLI

All subcommands take a config. The examples use the base config; swap in
`configs/experiments/sift_vs_orb_vs_akaze.yaml` to run the detector comparison.

`convert` additionally needs `--source`, for the reason given above. Every other
subcommand is fully described by its config.

```bash
python main.py convert   --config configs/default.yaml --source data/high_bright
python main.py detect    --config configs/default.yaml   # keypoints and descriptors per image
python main.py match     --config configs/default.yaml   # descriptor matching plus Lowe ratio test
python main.py evaluate  --config configs/default.yaml   # detector x matcher benchmark, metrics.csv
python main.py visualize --config configs/default.yaml   # keypoint, match and graph figures
python main.py pipeline  --config configs/default.yaml   # detect, match, geometry, graph end to end
python main.py panorama  --config configs/default.yaml   # warp, seam, blend, final mosaic
```

### First run, from RAW to a mosaic

```bash
python main.py convert  --config configs/default.yaml --source data/high_bright
# data.input_dir already points at data/processed/high_bright, then
python main.py pipeline --config configs/default.yaml
```

`compose.projection` defaults to `cylindrical`, which is what every scene here needs:
a planar mosaic projects each frame onto the tangent plane of the reference view, and
that plane diverges as the sweep widens. In planar, `high_bright` asks for a 25615x13310
canvas and `360_graus_images` for 45297x10649, and the `compose.canvas_max` guard stops
the run. Switch to `planar` only for a narrow sector, where it is exact and cheaper.

In a non planar projection the focal decides how far apart the frames sit on the
cylinder, and recovering it from the homographies is ill conditioned under a small
rotation: on this scene the estimate came out at 3471 px against the 1291 px the EXIF
implies, and the mosaic composed with black gaps between the frames. So declare the
camera instead, which is what the default config does:

```yaml
compose:
  focal_mm: 18.0          # from the EXIF of the sources
  sensor_width_mm: 22.3   # APS-C, Canon EOS Rebel T5i
```

They are converted against the working image width, so changing `data.max_dimension`
needs no edit here. `compose.focal_px` overrides both. Leave all three null and the
pipeline falls back to estimating, warning when the estimates disagree by more than a
factor of two. Item 6 of `future_works.md` has the measurements.

## Sweeping one axis at a time

`configs/experiments/` holds one config per variant. Each is a **full copy** of
`default.yaml`, comments included, with a single axis edited and a header saying which,
so two copies sharing a prefix differ in exactly one line and nothing is inherited from
anywhere. The prefix names the axis:

| prefix | axis | variants |
|---|---|---|
| `det_` | `detection.name` | sift, orb, akaze |
| `match_` | `matching.name` | flann, brute_force |
| `strategy_` | `matching.strategy` | all_pairs, sequential |
| `ransac_` | `geometry.ransac.method` | plain, magsac |
| `graph_` | `graph.min_inliers` and `min_inlier_ratio` | loose, default, strict |
| `proj_` | `compose.projection` | planar, cylindrical, spherical |
| `ref_` | `compose.reference` | center, first, index |
| `focal_` | where the focal comes from | exif, estimated |
| `wave_` | `compose.wave_correct` | horiz, vert, none |

Each sets its own `run.id` to its own name, so a variant lands in
`results/<config name>/` and two panoramas of the same prefix can be opened side by side.

Run one like any other config, or hand several to `sweep`:

```bash
python main.py panorama --config configs/experiments/proj_spherical.yaml
python main.py sweep configs/experiments/proj_*.yaml     # one axis
python main.py sweep configs/experiments/*.yaml          # all of them
```

`sweep` is orchestration only: every config goes through the same `panorama` command, and
it prints the canvas size and run directory of each. A variant that fails does not stop
the rest, which matters because two of them exist to demonstrate a limit rather than to
succeed: `proj_planar` is expected to abort on these scenes, since a planar mosaic
diverges as the sweep widens, and `focal_estimated` is expected to produce a mosaic with
gaps, since recovering the focal from the homographies is ill conditioned here.

Note that the first two axes are also what `evaluate` mode sweeps, but `evaluate` stops at
`metrics.csv` and never composes a panorama. Use these configs when the mosaic itself is
the subject.

## Two modes of operation

**Generation mode** (`detect`, `match`, `visualize`, `pipeline`, `panorama`) runs a
single configured detector and matcher and produces artifacts: keypoint and descriptor
files, match records with their homographies and inlier masks, the connectivity graph,
the figures, and the final panorama. This is the mode you use to actually build a
mosaic out of a scene.

**Evaluate mode** (`evaluate`) ignores `detection.name` and `matching.name` and sweeps
the matrix in `evaluate.detectors` x `evaluate.matchers` instead, timing each
combination over `detection.repetitions` runs and collecting every metric listed in
`evaluate.metrics` into a long-format `metrics.csv`. This is the mode you use to
compare detectors, not to produce a picture.

For the comparison to be honest, `nfeatures` must be the same for all three detectors,
otherwise the measured difference is a keypoint budget difference, not a detector
quality difference. Changing `data.max_dimension` invalidates any earlier comparison,
since keypoint counts, detection time and reprojection error all scale with resolution.

## Output: `results/<run_id>/`

- the resolved config, with provenance (OpenCV version, git commit)
- the run log
- `keypoints/` with one `.npz` per image and detector
- `matches/` with one `.npz` per image pair, including homography and inlier mask
- `figures/` with keypoint overlays, before and after match panels, the connectivity
  graph, the progressive stitching steps and the deghosting comparison
- `metrics.csv` (and `metrics.md` when `evaluate.export.markdown_table` is on), plus
  the benchmark plots
- the final panorama, and the naive average alongside it when `compose.save_naive` is on

Caching is on by default (`run.cache`), so re-running a later stage reuses the `.npz`
artifacts of the earlier ones.

## Configs

- `configs/default.yaml` is the base config; every field carries the schema default.
- `configs/experiments/sift_vs_orb_vs_akaze.yaml` extends it and overrides only what the
  experiment changes (`extends` merges recursively, the child wins).
- `configs/detectors.yaml` is not a config: it is a reusable set of per-detector
  parameter presets to copy into `evaluate.per_detector_params`.

## Tests

```bash
uv sync --extra dev   # if pytest is not installed yet
pytest
```
