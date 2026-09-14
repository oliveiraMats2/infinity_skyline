# Future works

Open decisions and known limits, found while running the pipeline end to end on the
real `.CR2` sets. Nothing here is a crash: the code runs and the test suite passes.
These are places where the implemented behaviour and the specification disagree, or
where the method has a ceiling worth naming before someone trips on it.

Evidence below comes from three runs kept under `results/`: `smoke` (the 25 image
360 degree sweep), `smoke_bench` (the detector comparison) and `smoke_vis`.

---

## 1. `compose.projection: planar` cannot stitch a 360 degree sweep

**Status:** open decision. The default ships `planar`, as specified.

**What happens.** On `data/processed/360_graus_images` (25 images, a full rotation)
the planar path aborts:

```
ValueError: the canvas would be 45297x10649 px, past compose.canvas_max=12000;
'IMG_0562' alone warps to 45297x8052 px, so its homography is degenerate
```

That is the `canvas_max` guard doing its job, not a bug. A planar mosaic projects
every frame onto the tangent plane of the reference view, and that plane diverges as
the view angle approaches 90 degrees from the reference. At a full rotation the
homography is genuinely singular, so no threshold tuning helps.

Switching to `compose.projection: cylindrical` composes the same 25 images into a
5622x2398 mosaic, with a focal of 1458.9 px estimated from 58 edge homographies.

**Options.**

* Change the default to `cylindrical`. Safe for wide sweeps, mildly wasteful for the
  small 3 to 5 frame sets where planar is exact and cheaper.
* Keep `planar` and let the guard teach the user. The error already names the
  offending image, but it does not say "try cylindrical".
* Pick the projection from the measured angular span of the graph, and warn when
  overriding. More code, no user decision needed.

The cheapest honest improvement, whatever is decided: extend the `canvas_max` error
message to suggest `compose.projection: cylindrical` when the estimated span is wide.

---

## 2. The 360 degree loop does not close (no bundle adjustment)

**Status:** known method limit. Out of the specified scope of `compose.py`.

**What happens.** Global homographies are built by chaining pairwise homographies
along the graph path to the reference frame (`chain_homographies`). Every pairwise
estimate carries a small error, and chaining multiplies them, so the error grows
along the path. Over a full rotation the last frames land visibly off: in
`results/smoke/panorama/panorama.png` the two frames furthest from the reference are
flung to the top left and bottom right corners, and black gaps open between groups.

This is inherent to chained pairwise estimation. Nothing in the current pipeline
knows that the first and last frame of a 360 degree sweep should coincide.

**Fix.** The standard OpenCV route, roughly what `stitching_detailed.py` does:

1. `cv2.detail_HomographyBasedEstimator` to get initial camera parameters, or derive
   them from the existing `_focals_from_homography` plus `_rotation_from_homography`,
   which `compose.py` already has.
2. `cv2.detail_BundleAdjusterRay` (or `BundleAdjusterReproj`) to refine all cameras
   jointly against every inlier correspondence, which distributes the drift around
   the loop instead of piling it at one end.
3. `cv2.detail.waveCorrect(rotations, cv2.detail.WAVE_CORRECT_HORIZ)` to straighten
   the residual roll, which is what makes the horizon bow in a long sweep.

The inputs the bundle adjuster wants (`ImageFeatures`, `MatchesInfo`) are exactly
what `KeypointRecord` and `MatchRecord` already hold, so this is an adapter plus the
three calls, not a rewrite. It belongs between `chain_homographies` and the warping
stage in `compose_panorama`.

**Cost.** Bundle adjustment over 25 images is seconds, not minutes. The real cost is
the adapter and the fallback path when it fails to converge.

---

## 3. `detection.repetitions` slows generation runs for a measurement nobody reads

**Status:** open decision. Current behaviour follows the specification literally.

**What happens.** `detection.repetitions: 10` and `warmup: 2` mean every image is
detected 12 times, and only the last result is kept. The other 11 exist solely to
fill `KeypointRecord.times_ms`, which feeds `detect_time_ms` in the benchmark table.

That is correct and necessary in `evaluate` mode. In `detect`, `match`, `visualize`,
`panorama` and `pipeline` the timing is never read, so those commands pay 12x the
detection cost for nothing. On the 25 image set at 800 px that is a few seconds; at
the configured `max_dimension: 1600` with `nfeatures: 5000` it is the dominant cost
of a generation run.

**Options.**

* Have the generation commands pass `repetitions=1, warmup=0` regardless of config.
  Simplest, and the timing they would have written is not used by anything.
* Add a `run.mode` field so the config states the intent, and let `detect_directory`
  read it. More explicit, one more field to keep in sync.
* Leave it. The config is the single source of truth, and a user who wants a fast
  generation run sets `repetitions: 1` themselves. This is the status quo.

Note that `run.cache: true` already removes the cost on a second run against the same
`run.id`, so this only bites on first runs and on fresh run ids.

---

## 4. `graph.reject_isolated` rejects more than its name and comment claim

**Status:** deliberate divergence from the specification, with a visible side effect.

**Specified:** `reject_isolated: true  # nodes with no valid edge are marked intruders`,
that is, degree zero.

**Implemented:** everything outside the largest connected component.

**Why the divergence.** The stated goal of stage 4 is rejecting an intruder image
from a different scene. `data/raw/intruders/` is a directory, plural: two or more
images of that other scene match each other perfectly well, form their own component
with degree greater than zero, and a degree zero rule would let every one of them
through. The largest component rule catches them.

**The side effect, observed.** In `results/smoke_vis`, `low_bright` has 7 images that
split into a 3 node component (`0541`, `0543`, `0545`), a second 3 node component
(`0546`, `0549`, `0550`) and one isolated node (`0553`). The second component is not
an intruder, it is a legitimate part of the same scene whose overlap fell under
`graph.min_inliers: 30`. The figure labels all four as rejected.

**Options.**

* Rename the field to `keep_largest_component` and fix the comment. Honest, and a
  breaking change to every existing config.
* Report the two cases separately: genuinely isolated nodes (degree zero) as
  intruders, and smaller components as "disconnected fragment, not composed". They
  need different wording in the figure and different handling in the log, because one
  means "wrong scene" and the other means "loosen your thresholds".
* Keep it and just fix the YAML comment to describe what the code does.

The second option is the one that would have helped here: seeing "fragment" instead
of "intruder" points straight at `min_inliers`, which is the actual cause.

---

## 5. `evaluate` silently ignores `data.intruders_dir`

**Status:** defensible behaviour, undocumented.

**What happens.** `cli._scene_images` appends `intruders_dir` to the image list for
every generation command. `evaluate.run_benchmark` builds its own list from
`data.input_dir` only, so the field has no effect in evaluation mode. In
`results/smoke_bench` the config pointed `intruders_dir` at the 25 image set and the
benchmark still produced 21 pairs, that is 7 choose 2, the scene alone.

**Why it is arguably right.** The benchmark compares detectors on pairs that share a
view. Adding 25 unrelated images turns 21 pairs into 496, of which roughly 475 are
guaranteed non matches. They would dominate every mean in `metrics.csv` and measure
nothing about detector quality on real overlaps.

**Why it is still wrong as it stands.** A config field that is read in one mode and
dropped without a word in another is a trap.

**Options.**

* Log a warning in `run_benchmark` when `intruders_dir` is set, saying it is ignored
  and why. One line, removes the trap.
* Include intruders and add an `is_intruder_pair` column, so rejection quality per
  detector becomes measurable. Genuinely interesting for the article: which detector
  produces fewest false homographies between unrelated scenes. Costs 24x the pairs.
* Reject the config at validation time when `intruders_dir` is set and the command is
  `evaluate`. Cleanest, but `config.py` does not know which command is running.

---

## Smaller observations

**`infer_order` transposes locally on densely connected graphs.** On the 25 image
sweep the inferred order came out
`0558 -> 0559 -> 0556 -> 0557 -> 0554 -> 0578 -> ...`: consecutive pairs are swapped
against the capture order. The graph has 68 edges over 25 nodes, so it is not the
chain the diameter path heuristic assumes, and the leftover nodes get inserted next
to their strongest neighbour without a global direction. This does not affect the
mosaic, because `chain_homographies` walks `nx.shortest_path` on the graph rather
than the order list, but it does affect the progressive figures of stage 5.4 and any
claim the report makes about recovering capture order. A spectral ordering (Fiedler
vector of the graph Laplacian) would be the small fix.

**`cv2.detail.focalsFromHomography` is unusable from Python.** Its focal values are
C++ output references that the binding does not return, so the call always yields
`None`. `compose._focals_from_homography` reimplements the closed form from
`autocalib.cpp` in numpy. Worth knowing before someone "simplifies" it back to the
OpenCV call.

**OpenCV must stay below 5.0.** `opencv-contrib-python` 5.x dropped `AKAZE`, `KAZE`
and `BRISK` from the Python bindings, and AKAZE is one of the three detectors under
comparison. `pyproject.toml` pins `>=4.9,<5` for this reason.
