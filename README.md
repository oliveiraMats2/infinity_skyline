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
`--extra learned` adds torch, kornia and transformers for the `reference_*` methods;
their weights download on first use.

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
python main.py evaluate  --config configs/default.yaml   # detector x matcher benchmark table
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

To run the same axes over another scene or into another folder, without editing the
configs, list them in a sweep file whose `override` block is merged over each one (see
`configs/sweep.yaml`):

```bash
python main.py sweep --config configs/sweep.yaml
```

The phone scenes are HEIC. `convert` reads them as it reads RAW, so each scene is one
`convert` into `data/processed/<scene>` and one `sweep`, which writes to
`~/<scene>/results` and `~/<scene>/artifacts`. Replace `~` with the full path on Windows:

```
python main.py convert --config configs/default.yaml --source ~/plasma_180_degree --dest data/processed/plasma_180_degree
python main.py convert --config configs/default.yaml --source ~/plasma_23_sep_plano --dest data/processed/plasma_23_sep_plano
python main.py convert --config configs/default.yaml --source ~/360_plasma_images --dest data/processed/360_plasma_images

python main.py sweep --config configs/sweep_plasma_180_degree.yaml    # flat panorama, report axes
python main.py sweep --config configs/sweep_plasma_23_sep_plano.yaml  # flat panorama, report axes
python main.py sweep --config configs/sweep_360_plasma_images.yaml    # 360, bundle adjustment on
```

These sweep files set `resume: true`: stopping a sweep and running the same command
again skips only the configs that finished (`results/<config>/pipeline.done`, written as
the last step) and redoes the interrupted one in full. Delete that file to redo a config.

### convert

Recursive, and the source subfolder tree is mirrored under `--dest` (default
`data/processed/<source name>`), so it takes any image folder. Accepts every RAW format
`io.RAW_EXTENSIONS` lists, plus HEIC, JPEG, TIFF and PNG. A file it cannot read is logged and
counted, never fatal, and two sources colliding on one output name are reported instead
of one overwriting the other.

```bash
python main.py convert --config configs/default.yaml \
    --source ~/photos/any_folder --max-dimension 1280 --overwrite
```

Every PNG is at most 1920x1080 (1080x1920 portrait), aspect kept; `--max-dimension`
only lowers the long side further. `--overwrite` falls back to `run.overwrite`. Without
it an existing PNG is skipped silently, so pass it to redo a folder converted at another
size. The pipeline still reads at `data.max_dimension` on top of that.

## Output: `results/<run_id>/`

The resolved config with provenance, the run log, `keypoints/` and `matches/` as `.npz`,
and `panorama/`: `panorama.png` (`compose.blend.method`), one `panorama_<method>.png` per
method in `compose.blend.compare`, `naive.png`, `seam_mask.png` and, for 360 degree sweeps
(`compose.save_equirectangular`, on in the 360 sweep only), `panorama_equirectangular.jpg`
(2:1 with GPano XMP, opens in 3D in a 360 viewer). Figures go under
`figures/`, one subdirectory per module of `visualize/`:

| Subdirectory | Contents |
|---|---|
| `keypoints/` | one overlay and one image, R, corners, keypoints panel per image, plus response and scale histograms and the spatial density map |
| `matches/` | the before and after panel per pair, plus inlier against outlier distance and the inlier matrix |
| `graph/` | the node and edge drawing, plus the connectivity matrix and the edge weight distribution |
| `panorama/` | the progressive steps, the deghosting comparison and the difference map against the naive average |
| `report/` | the `evaluate` benchmark charts |

`visualize.figures.format` and `dpi` apply to all of them. A figure whose metric column
is missing from the benchmark table is skipped with a warning instead of failing the run.

Caching is on by default (`run.cache`), so re-running a later stage reuses the `.npz` of
the earlier ones.

## Tables: `artifacts/<run_id>/`

Every table is a CSV named `<method>_<what>`, the method being `<detector>_<matcher>` or
the `reference_*` name: `_matching_metrics` (inlier rate and reprojection error of each
pair the mosaic uses), `_graph_connectivity`, `_graph_order` (with the rejected images),
`<blend>_blending_metrics` (one per technique, `naive` included) and
`evaluate_benchmark_metrics` from `evaluate`.

## Modes

**Generation** (`detect`, `match`, `visualize`, `pipeline`, `panorama`) runs the one
configured detector and matcher and builds a mosaic.

**Evaluate** ignores `detection.name` and `matching.name` and sweeps
`evaluate.detectors` x `evaluate.matchers` into a long-format table. For the
comparison to be honest, `nfeatures` must be equal across detectors, otherwise the
measured difference is a keypoint budget difference. Changing `data.max_dimension`
invalidates any earlier comparison.

`panorama`, `pipeline` and `sweep` also ensure a `cv2.Stitcher` mosaic of the same frames
exists under `results/stitcher/` (`reference_cv2_stitcher_*`), one file shared by every run, as an independent
reference: our own output can only tell us whether it is self consistent, not whether it
is right. `baseline` produces only that.

## Configs

- `configs/default.yaml` is the base; every field carries its schema default.
- `configs/experiments/` holds full copies of it, never `extends`, one per variation
  axis, each differing in the one line its header names.
- `reference_superpoint`, `reference_superglue` and `reference_loftr` run the learned
  methods inside the pipeline in place of our detector and matcher; `bundle_on`/`bundle_off`
  toggle `compose.bundle_adjust`; `scene_360` is the full 360 degree sweep.
- `configs/detectors.yaml` is not a config but a set of per-detector presets to copy into
  `evaluate.per_detector_params`.

## Tests

```bash
pytest
```
