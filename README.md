# infinity-skyline

A benchmark and panorama pipeline over skyline photo sets. It detects local features
(SIFT, ORB, AKAZE), matches them across images, estimates pairwise homographies with
RANSAC, builds a connectivity graph of the scene, infers the sweep order, and composes
the images into a single mosaic with exposure compensation, seam finding and multi-band
blending.

One YAML describes a whole run, validated against pydantic schemas, so a typo fails
immediately instead of halfway through. Artifacts land in `results/<run_id>/`.

## Installation

```bash
uv venv .venv --prompt infinity_skyline && source .venv/bin/activate
uv sync --extra dev
```

Python 3.11 or newer. `--extra dev` adds `pytest`. `opencv-contrib-python` is pinned
below 5.0 because the 5.x bindings dropped AKAZE, one of the three detectors compared.

## Data

One directory is one scene, and the pipeline runs over one scene at a time.

| Directory | Files | What it is |
|---|---|---|
| `data/high_bright/` | 8 `.CR2` | The same view, over-exposed. **The default scene** |
| `data/low_bright/` | 7 `.CR2` | The same view, under-exposed |
| `data/360_graus_images/` | 25 `.CR2` | The panoramic 360 degree sweep |

`data.input_dir` points at the PNG output of `convert`, not at the RAW source. Pointing
`data.intruders_dir` at another scene injects foreign images and exercises the rejection
stage: they should fall outside the largest connected component.

## CLI

Every subcommand takes a config. Only `convert` also needs `--source`, because its input
is the RAW folder rather than `data.input_dir`.

```bash
python main.py convert   --config configs/default.yaml --source data/high_bright
python main.py detect    --config configs/default.yaml   # keypoints and descriptors per image
python main.py match     --config configs/default.yaml   # matching plus Lowe ratio test
python main.py evaluate  --config configs/default.yaml   # detector x matcher benchmark, metrics.csv
python main.py visualize --config configs/default.yaml   # keypoint, match and graph figures
python main.py pipeline  --config configs/default.yaml   # the above end to end, every figure, and the mosaic
python main.py panorama  --config configs/default.yaml   # warp, seam, blend, mosaic only, no figures
python main.py baseline  --config configs/default.yaml   # the same scene through cv2.Stitcher
```

Run every variation axis at once, one full `pipeline` per config, into its own
`results/<run_id>/`. A config that fails does not stop the rest:

```bash
python main.py sweep configs/default.yaml configs/experiments/*.yaml
```

### convert

Recursive, and the source subfolder tree is mirrored under `--dest` (default
`data/processed/<source name>`), so it takes any image folder. Accepts every RAW format
`io.RAW_EXTENSIONS` lists, plus JPEG, TIFF and PNG. A file it cannot read is logged and
counted, never fatal, and two sources colliding on one output name are reported instead
of one overwriting the other.

```bash
python main.py convert --config configs/default.yaml \
    --source ~/photos/any_folder --max-dimension 2400 --overwrite
```

`--max-dimension` falls back to `data.max_dimension`, `--overwrite` to `run.overwrite`.
Without `--overwrite` an existing PNG is skipped silently, and since `resize_to_max`
never upscales, a stale PNG smaller than `data.max_dimension` makes the whole pipeline
run at the lower resolution without saying so. Pass it whenever `max_dimension` changed.

## Output: `results/<run_id>/`

The resolved config with provenance, the run log, `keypoints/` and `matches/` as `.npz`,
the panorama, and `metrics.csv` plus `metrics.md` from `evaluate`. Figures go under
`figures/`, one subdirectory per module of `visualize/`:

| Subdirectory | Contents |
|---|---|
| `keypoints/` | one overlay per image, plus response and scale histograms and the spatial density map |
| `matches/` | the before and after panel per pair, plus inlier against outlier distance and the inlier matrix |
| `graph/` | the node and edge drawing, plus the connectivity matrix and the edge weight distribution |
| `panorama/` | the progressive steps, the deghosting comparison and the difference map against the naive average |
| `report/` | the `evaluate` benchmark charts |

`visualize.figures.format` and `dpi` apply to all of them. A figure whose metric column
is missing from `metrics.csv` is skipped with a warning instead of failing the run.

Caching is on by default (`run.cache`), so re-running a later stage reuses the `.npz` of
the earlier ones.

## Modes

**Generation** (`detect`, `match`, `visualize`, `pipeline`, `panorama`) runs the one
configured detector and matcher and builds a mosaic.

**Evaluate** ignores `detection.name` and `matching.name` and sweeps
`evaluate.detectors` x `evaluate.matchers` into a long-format `metrics.csv`. For the
comparison to be honest, `nfeatures` must be equal across detectors, otherwise the
measured difference is a keypoint budget difference. Changing `data.max_dimension`
invalidates any earlier comparison.

`panorama`, `pipeline` and `sweep` also ensure a `cv2.Stitcher` mosaic of the same frames
exists under `results/_reference/`, one file shared by every run, as an independent
reference: our own output can only tell us whether it is self consistent, not whether it
is right. `baseline` produces only that.

## Configs

- `configs/default.yaml` is the base; every field carries its schema default.
- `configs/experiments/` holds full copies of it, never `extends`, one per variation
  axis, each differing in the one line its header names.
- `configs/detectors.yaml` is not a config but a set of per-detector presets to copy into
  `evaluate.per_detector_params`.

## Tests

```bash
pytest
```
