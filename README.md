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
uv venv .venv --prompt infinity_skyline && source .venv/bin/activate && uv sync
```

Requires Python 3.11 or newer.

## Input data

The source images are Canon RAW `.CR2` files. Neither OpenCV nor Pillow can decode
them, which is why `rawpy` is a hard dependency: it is used by `io.load_image` and by
the `convert` subcommand, which writes lossless PNG copies that the rest of the
pipeline reads.

One directory is one scene, and the pipeline runs over one scene at a time:

| Directory | Files | What it is |
|---|---|---|
| `data/360_graus_images/` | 26 `.CR2` | The panoramic 360 degree sweep, the main scene |
| `data/low_bright/` | 8 `.CR2` | The same view, under-exposed |
| `data/high_bright/` | 8 `.CR2` | The same view, over-exposed |

`low_bright` and `high_bright` are exposure brackets of the same view, and that is
exactly why `compose.exposure: gain_blocks` matters: when frames of one mosaic arrive
at different brightness levels, a single global gain cannot fix vignetting or the
brightness variation inside a frame, while block-wise gain compensation can.

Pointing `data.intruders_dir` at a different scene directory injects foreign images
into the run and exercises the rejection stage: they should fall outside the largest
connected component of the graph.

`convert` reads `data/<scene>/*.CR2` and writes `data/processed/<scene>/*.png`.

## CLI

All subcommands take a config. The examples use the base config; swap in
`configs/experiments/sift_vs_orb_vs_akaze.yaml` to run the detector comparison.

```bash
python main.py convert   --config configs/default.yaml   # RAW to PNG under data/processed/
python main.py detect    --config configs/default.yaml   # keypoints and descriptors per image
python main.py match     --config configs/default.yaml   # descriptor matching plus Lowe ratio test
python main.py evaluate  --config configs/default.yaml   # detector x matcher benchmark, metrics.csv
python main.py visualize --config configs/default.yaml   # keypoint, match and graph figures
python main.py pipeline  --config configs/default.yaml   # detect, match, geometry, graph end to end
python main.py panorama  --config configs/default.yaml   # warp, seam, blend, final mosaic
```

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
pytest
```
