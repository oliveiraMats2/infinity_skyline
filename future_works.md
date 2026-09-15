# Future works

Open decisions and known limits, found while running the pipeline end to end on the
real `.CR2` sets. Nothing here is a crash: the code runs and the test suite passes.
These are places where the implemented behaviour and the specification disagree, or
where the method has a ceiling worth naming before someone trips on it.

Evidence below comes from three runs kept under `results/`: `smoke` (the 25 image
360 degree sweep), `smoke_bench` (the detector comparison) and `smoke_vis`.

---

## 1. Planar projection cannot stitch a wide sweep

**Status:** resolved for the default config, one loose end left.

The default is now `compose.projection: cylindrical` with `compose.canvas_max: 20000`,
because both scenes tried so far are wide sweeps and neither composes in planar. Planar
remains the right choice for a narrow sector, where it is exact and cheaper.

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
5622x2398 mosaic, with a focal of 1458.9 px estimated from 58 edge homographies. The
default scene `high_bright` behaves the same way: 25615x13310 px in planar, 12341x1706
in cylindrical.

**The loose end: the `canvas_max` error misdiagnoses.** It always blames the widest
single tile as "degenerate", which is right when a homography really has blown up and
wrong when the mosaic is simply wide. On the cylindrical `high_bright` run it reported

```
the canvas would be 12341x1706 px, past compose.canvas_max=12000;
'IMG_0555' alone warps to 1640x1145 px, so its homography is degenerate
```

A tile of 1640x1145 is not degenerate, it is one ordinary frame. The message should
compare the worst tile against the source frame size and say "no single image is
degenerate, the mosaic is legitimately this wide, raise compose.canvas_max" when they
are comparable, keeping the current wording only when one tile actually dominates the
canvas. It should also suggest `cylindrical` when the run is planar.

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

## 6. The estimated focal drives the cylindrical layout, and it is unreliable

**Status:** worked around for a known camera, still open when the camera is unknown.
`compose.focal_mm` and `compose.sensor_width_mm` (or `compose.focal_px`) now state the
focal, and the default config declares the 18 mm lens and the 22.3 mm APS-C sensor of
these sources, which closes every gap. The estimator is still what runs when they are
null, and it is still unreliable; it now warns when its estimates disagree by more than
a factor of two. This section supersedes an earlier version that blamed the capture and
the graph thresholds. That diagnosis was wrong, and how it was wrong is worth
recording.

**What the layout depends on.** In a non planar projection, `compose.py` builds the
intrinsics `K` from a single focal estimated by `_estimate_focal`, the median of
`_focals_from_homography` over the graph edges. `K` then decides the rotation extracted
from each homography and the arc length each frame occupies on the cylinder. A wrong
focal does not fail loudly, it spaces the frames wrongly.

**The estimate is wrong by 2.7x on the default scene.** Ground truth from the EXIF of
the source files: Canon EOS Rebel T5i, APS-C sensor 22.3 mm wide, lens at 18 mm. At the
configured `max_dimension: 1600` that is

```
f = 18.0 / 22.3 * 1600 = 1291 px      horizontal field of view 63.6 degrees
```

What the pipeline computed, edge by edge:

| edge | inliers | focal estimates |
|---|---:|---|
| `0540 -> 0542` | 385 | 3825 |
| `0540 -> 0555` | 1384 | none |
| `0542 -> 0544` | 244 | none |
| `0542 -> 0555` | 441 | 1037, 3118 |
| `0544 -> 0547` | 89 | 6713 |
| `0547 -> 0548` | 173 | none |
| `0547 -> 0551` | 80 | none |
| `0548 -> 0551` | 409 | none |
| `0551 -> 0552` | 84 | none |

Six of the nine edges yield nothing, and the four values that survive span 1037 to
6713, a factor of 6.5. Their median, 3471 px, is 2.69 times the truth and implies a
26 degree field of view for a lens that sees 63.6.

**Why the closed form fails here.** `_focals_from_homography` solves for the focal from
the perspective terms `h[6]` and `h[7]`. A rotation about the camera centre with a small
angle leaves those terms near zero, so the system is ill conditioned: the denominators
`h[6]*h[7]` and `(h[7]-h[6])*(h[7]+h[6])` approach zero and the candidate focal squares
either go negative, which is the `none` rows, or blow up, which is the 6713. This is a
property of the formula, not a coding error, and it is why OpenCV's own stitcher feeds
`estimateFocal` many pairs and then runs bundle adjustment over the result.

**Measured effect, and the proof it is the cause.** Recomposing the same graph, the same
matches and the same images, changing only the focal:

| focal | canvas | empty columns |
|---|---|---|
| 3471 px, estimated | 12341x1706 | 1110 px, 9.0 percent of the width |
| 1291 px, from EXIF | 4674x1098 | 0 px |

With the EXIF focal the frames overlap, the railing runs continuously across the whole
mosaic and there are no black gaps at all.

**Fix, in increasing order of effort.**

* ~~Let the config state the focal~~. **Done.** `compose.focal_mm` plus
  `compose.sensor_width_mm`, converted against the working image width, or a direct
  `compose.focal_px`. All null keeps the estimate. This solves it for any scene shot on
  known glass, at the cost of the user having to know the glass.
* Carry the EXIF across the conversion. `convert` reads the RAW, where the focal lives,
  and writes PNG, where it does not. Writing a small sidecar JSON per scene at convert
  time would make the focal available without the user typing it.
* Sanity filter the estimates: drop any candidate outside a plausible band and warn when
  the surviving spread is wide. Note this alone would not have saved this run, since
  3825 sits inside any reasonable band. It is a guard, not a fix.
* Bundle adjustment, item 2 above, refines the focal jointly with the rotations and is
  the principled answer. It also needs a starting focal that is not off by 2.7x.

**What the earlier wrong diagnosis was, and why.** A previous version of this section
measured the x translation implied by each edge homography, found values like 3282 and
4064 px against a 1600 px frame, called them geometrically impossible, and concluded
that repetitive structure on the railing was fabricating false edges. Two errors: those
numbers came from chaining homographies in the *planar* frame, which is not the path a
cylindrical run takes, and the pairs in question sit two positions apart in the sweep,
where a displacement of two to three frame widths is exactly what is expected rather
than impossible. The lesson is to measure the composition from `seam_mask`, which
records the source of every pixel that actually landed on the canvas, and not from a
re-derivation that may not match the code path under test.

---

## 7. Sign of the chained homography flipped the end frames (fixed)

**Status:** fixed. Recorded because the failure mode is invisible and easy to reintroduce.

A homography and its negation are the same projective transform, but `cv2.findHomography`
normalizes `H[2,2]` to 1, and that flips the sign of the whole matrix whenever the true
`[2,2]` is negative, which happens once a frame is far enough in angle from the reference.
The determinant then goes negative. `_rotation_from_homography` orthonormalizes
`K^-1 H^-1 K` with an SVD and has a `det < 0` branch that multiplies by
`diag([1, 1, -1])`, so a negative determinant sends it down the reflection path and it
returns a rotation wrong by more than 100 degrees. Nothing raises, nothing warns: the
frame is simply pasted into the mosaic crooked.

Measured on the default scene, reference `IMG_0547`, roll and pitch per frame:

| frame | hops | det(H) | roll before | roll after |
|---|---:|---:|---:|---:|
| `0540` | 3 | **-80.5** | -123.5 | **6.5** |
| `0555` | 3 | **-509** | -76.7 | **5.2** |
| `0542` | 2 | +74.0 | 3.2 | 3.2 |
| `0544` | 1 | +2.7 | 1.3 | 1.3 |
| `0548` | 1 | +0.75 | -1.0 | -1.0 |
| `0551` | 1 | +0.74 | -1.9 | -2.0 |
| `0552` | 2 | +3.7 | -29.0 | **-16.7** |

Exactly the two frames with a negative determinant were the crooked ones. The fix is one
guard in `_rotation_from_homography`: negate the homography when its determinant is
negative, before extracting the rotation.

`0552` improved for a second reason, covered next.

**Chains now route over the maximum spanning tree.** `chain_homographies` used
`nx.shortest_path`, which counts hops and ignores edge weight, so `0552` reached the
reference through the two weakest edges of the whole graph, 84 and 80 inliers. Error
multiplies along a chain, and one weak shortcut costs more than an extra strong hop.
Routing over `nx.maximum_spanning_tree` sends it through 84, 409 and 173 instead, one
hop longer and far stronger, which took its spurious roll from 29 degrees to 17.

**Net effect on the horizon**, measured by fitting a line to the sky to ground boundary
across the mosaic:

| | tilt | horizon flatness (std) |
|---|---:|---:|
| before | -0.73 deg | 86.7 px |
| after | **+0.09 deg** | **77.2 px** |

**Wave correction.** `cv2.detail.waveCorrect` is wired up as `compose.wave_correct` and
defaults to `horiz`. An earlier measurement said it hurt and it was defaulted off; that
measurement was taken while the rotation convention bug below was still present, so it
was comparing two wrong layouts. Re-measured after that fix, it helps, though modestly:
+0.62 deg of tilt against +0.81 without it, and a flatter horizon, 91.0 px against 94.6.
The gain stays small because the rotations come from chained homographies rather than
from bundle adjustment. Revisit when item 2 lands.

**What is still left.** `0552` keeps 17 degrees of spurious roll and `0540` keeps 6.5,
and both grow with distance from the reference. That residue is accumulated chain error
and it is what bundle adjustment exists to remove. Item 2 is still the real fix.

---

## 8. The rotation handed to the warper was inverted, and mirrored the sweep

**Status:** fixed. This is the one a reader is most likely to reintroduce.

`_rotation_from_homography` derived the camera rotation correctly. With the reference at
identity the homography is `H = K R^-1 K^-1`, so the camera rotation is `R = K^-1 H^-1 K`,
and that is what the function returned. The problem is that this is not the convention
`cv2.PyRotationWarper` wants: it turns an image point into a world ray with `R * K^-1`,
which is the inverse. Handed the camera rotation, the warper laid every frame out on the
wrong side of the reference and the whole sweep came back in reverse order.

It is a silent failure. The mosaic still has zero gaps, the seams still line up and the
railing still runs continuously, because reversing every frame consistently is
self consistent. Only comparing against the world shows it.

**How it was caught.** `cv2.Stitcher_create(cv2.Stitcher_PANORAMA)` over the same eight
images, as an independent ground truth. It puts the dish antenna of `IMG_0540` on the
left and the blue tower on the right. Ours had them the other way round. The capture
direction agrees with OpenCV: the edge homography `0540 -> 0542` has `dx = -1200 px`, so
content moves left as the sequence advances, so the camera pans right, so `0540` belongs
at the left end.

| | left to right |
|---|---|
| capture order | 0540 0542 0544 0547 0548 0551 0552 0555 |
| before the fix | 0552 0551 0548 0547 0544 0542 0555 0540 |
| after the fix | **0540 0555 0542 0544 0547 0548 0551 0552** |

`0555` sits beside `0540` because they are the same viewpoint shot twice, which is right.

**The fix** is one term: `K^-1 H K` instead of `K^-1 H^-1 K`. Note that orthonormalizing
the inverse and inverting the orthonormalized matrix agree here, since the polar factor
of `M^-1` is the transpose of the polar factor of `M`, so transposing the result works
equally well.

**Worth keeping as a habit:** a stitching result that looks plausible is not evidence.
Both the focal defect of item 6 and this one produced mosaics that were internally
consistent and wrong. This check now lives in the pipeline as `main.py baseline`, and
`pipeline` writes `opencv_stitcher.png` beside `panorama.png` on every run. It costs
about a second and a half on this scene.

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
