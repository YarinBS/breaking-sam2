# Metrics: definitions, calculation, and interpretation

This document explains the mathematics behind every number that the benchmark
reports. Sections 1–3 match the code in `src/metrics.py`, and Section 4
matches `reproduce_results.py`. The thresholds come from
`configs/benchmark.yaml`.

## Notation

For one video frame $t$:

- $G_t$ is the ground-truth (GT) mask: the set of pixels that belong to the target.
- $P_t$ is the predicted mask: the set of pixels that the pipeline labels as the target.
- $|A|$ is the number of pixels in the set $A$.
- $\mathcal{T}$ is the set of **annotated** frames. The code allows sparse GT
  (the protocol planned every 5th–10th frame), but all four datasets of the
  benchmark annotate every frame, so here $\mathcal{T}$ contains every frame.

All masks are binary. For indexed masks (DAVIS, M3-VOS, VOST, VOT),
`--object-id k` keeps only the pixels with value $k$. Without it, every
nonzero pixel counts as foreground.

**Important rule:** a frame without an annotation file is *skipped*. It is never
treated as an empty GT mask. Every per-frame metric below is computed only for
$t \in \mathcal{T}$.

---

## 1. Per-frame metrics

These are saved for every annotated frame in `metrics/frame_metrics.csv`.

### 1.1 Region similarity `J` (Jaccard index / IoU)

$$
J_t = \frac{|G_t \cap P_t|}{|G_t \cup P_t|}, \qquad
J_t = 1 \text{ if } G_t = P_t = \varnothing .
$$

**Meaning.** $J$ is the share of pixels that are correct, out of all pixels that
either mask calls "target". It measures how well the two regions overlap.

**Example.** Two $4\times4$ squares that are shifted by 2 pixels in each
direction overlap in a $2\times2$ area:
$|G\cap P| = 4$, $|G\cup P| = 16 + 16 - 4 = 28$, so $J = 4/28 \approx 0.14$.
A small shift already gives a very low score.

**Interpretation.**

| $J$ | Typical reading |
| --- | --- |
| $\ge 0.8$ | Good segmentation; errors are mostly at the edges. |
| $0.5$ – $0.8$ | Target is found, but a large part is missing or extra. |
| $< 0.5$ | Counted as a failed frame by the failure rule (Section 3). |
| $0$ | No overlap: the target is lost or a different object is tracked. |

- $J$ is **stricter for small and thin objects**. A shift of a few pixels
  destroys most of the overlap of a thin stream of liquid, while a large object
  hardly changes. Compare $J$ between clips with this in mind.
- If the target leaves the frame ($G_t = \varnothing$), then $J_t = 1$ only when
  the prediction is also empty. Any remaining prediction gives $J_t = 0$. This
  is how "false positives that stay after the target disappears" show up in the
  data.

### 1.2 Boundary accuracy `F`

$F$ checks whether the *contours* of the two masks lie in the same place,
allowing a small distance error.

**Step 1 — boundary maps.** A pixel is a boundary pixel if its mask value is
different from its east, south, or south-east neighbour (the DAVIS
`seg2bmap` rule). Call the boundary maps $\partial G_t$ and $\partial P_t$.
Mask edges that touch the image border are not counted as boundary.

**Step 2 — tolerance radius.** With `boundary_tolerance` $\tau = 0.008$ and an
image of height $H$ and width $W$:

$$
r = \max\!\left(1,\; \left\lceil \tau \sqrt{H^2 + W^2} \right\rceil\right)
\text{ pixels}.
$$

For DAVIS 480p
($854\times480$), $r = \lceil 0.008 \cdot 979.7 \rceil = 8$ px. For 1080p,
$r = 18$ px. Because $r$ grows with image size, $F$ gives comparable results
across resolutions.

**Step 3 — precision and recall.** Let $D_r(\cdot)$ be dilation with a disk of
radius $r$ ("everything within $r$ pixels of the boundary").

$$
\text{Precision}_t = \frac{|\partial P_t \cap D_r(\partial G_t)|}{|\partial P_t|},
\qquad
\text{Recall}_t = \frac{|\partial G_t \cap D_r(\partial P_t)|}{|\partial G_t|}
$$

$$
F_t = \frac{2\,\text{Precision}_t\,\text{Recall}_t}{\text{Precision}_t + \text{Recall}_t}
$$

Special cases: if both boundaries are empty, $F_t = 1$. If only one is empty,
$F_t = 0$.

**Meaning.**

- *Precision*: how much of the predicted contour is close to a real contour.
  Low precision means the prediction has extra edges (background leaking in,
  noise blobs).
- *Recall*: how much of the real contour is found. Low recall means parts of
  the object outline are missing (for example, a thin part cut off).
- $F$ is their **harmonic mean**, so it is low whenever *either* one is low.
  Example: precision 0.9 and recall 0.6 give $F = 2\cdot0.54/1.5 = 0.72$, which
  is lower than the simple average 0.75.

**Interpretation.** $F$ is sensitive to *shape*, while $J$ is sensitive to
*area*. A blob that covers the right place but has a smooth, wrong outline can
have a good $J$ and a poor $F$. This happens often with non-solid targets such
as smoke, flames, or pouring liquid. When the mask lags behind a changing
contour, $F$ usually drops before $J$ does.

### 1.3 Combined score `J&F`

$$
(J\&F)_t = \frac{J_t + F_t}{2}
$$

This is the standard single number in DAVIS-style papers. Use it for headline
tables, but look at $J$ and $F$ separately when you explain *why* a clip fails.

### 1.4 Signed area error

$$
\text{AreaErr}_t = \frac{|P_t| - |G_t|}{|G_t|}, \qquad
\text{undefined (empty) if } |G_t| = 0 .
$$

**Meaning and interpretation.**

- $> 0$: the prediction is too large (over-segmentation, background takeover).
  $+1.0$ means the predicted mask has twice the true area.
- $< 0$: the prediction is too small (the target is shrinking or being lost).
  $-1.0$ means the prediction is empty.
- $\approx 0$ with a low $J$: the area is right but the mask is in the **wrong
  place** (identity switch or drift to a similar object).

The sign tells you the *direction* of the error, which $J$ cannot tell you.

### 1.5 Fragmentation

$$
\text{Fragments}_t = \text{number of 8-connected components in } P_t
$$

**Meaning.** How many separate pieces the predicted mask has. Note that a
normal single-object mask gives **1** (not 0), and an empty prediction gives 0.

**Interpretation.** Values above 1 mean that the mask has split. This is not
always an error: a pouring liquid or a torn piece of dough can really split into
several parts. Compare with the GT by eye (or count GT components) before you
call it "mask fragmentation". Stray small blobs far from the target also raise
this number.

### 1.6 Temporal IoU (prediction stability)

$$
\text{TIoU}_t = \frac{|P_{t-1} \cap P_t|}{|P_{t-1} \cup P_t|}
$$

It is computed at annotated frames $t$, using the prediction of the frame just
before ($t-1$), when that prediction file exists. It does **not** use GT.

**Interpretation.**

- High TIoU means the mask changes smoothly from frame to frame.
- Low TIoU means flicker or a sudden jump. But fast real motion also lowers
  TIoU, so compare it with the clip's motion.
- **High TIoU with falling $J$** is an important pattern: the mask is stable but
  wrong. The model keeps its old memory while the object changes (boundary lag,
  "stuck" mask).
- **Low TIoU with falling $J$**: the model is unstable and loses the target
  suddenly.

---

## 2. Per-clip summary

`metrics/summary_metrics.json` aggregates the per-frame rows. Let the annotated
frames be ordered as $t_0 < t_1 < \dots < t_{N-1}$. By the protocol,
$t_0 = 0$ and $G_0$ is not empty.

| Field | Formula | Meaning |
| --- | --- | --- |
| `initialization_j`, `_f`, `_j_and_f` | value at $t_0$ | Quality of the first-frame mask. Low here means the **initialization** failed (bad box, or Grounding DINO chose the wrong object), not propagation. |
| `mean_j`, `mean_f`, `mean_j_and_f` | $\frac{1}{N}\sum_{i} J_{t_i}$ (etc.) | Average quality over all annotated frames. |
| `propagation_mean_j` | $\frac{1}{N-1}\sum_{i\ge1} J_{t_i}$ | Average $J$ **without** frame 0, so only tracking quality. |
| `worst_j` | $\min_i J_{t_i}$ | The single worst moment of the clip. |
| `mean_absolute_area_error` | mean of $\lvert\text{AreaErr}_t\rvert$ over frames with $\lvert G_t\rvert>0$ | Typical size error, in either direction. |
| `mean_fragments` | mean of $\text{Fragments}_t$ | Typical number of mask pieces. |
| `mean_temporal_iou` | mean of defined $\text{TIoU}_t$ | Typical frame-to-frame stability. |

**Why keep initialization and propagation separate.** For `grounded_sam2`, a
Grounding DINO miss produces empty masks for every frame. Then $J_{t_0}=0$ and
the whole clip fails, but SAM 2 never had a chance to track. Report such clips
as *detection failures*, not as SAM 2 propagation failures. Compare
`propagation_mean_j` only between runs whose initialization succeeded.

**Caveat about means.** The mean is over *annotated* frames. In this benchmark
every frame is annotated, so it is a plain frame average. With sparse GT that
is denser around the event, the event period would get more weight.

---

## 3. Temporal failure analysis

The mean hides *when* a clip breaks. The failure rule turns the $J$ curve into
an event time. The rule was fixed before the experiments
(`failure_iou = 0.5`, `failure_consecutive = 2`).

### 3.1 First failure

Walk through the annotated observations $(t_i, J_{t_i})$ in time order. The
first failure is

$$
t_{\text{fail}} = \min\{\, t_i \;:\; J_{t_i} < 0.5 \text{ and } J_{t_{i+1}} < 0.5 \,\}
$$

(in general: the first frame of the first run of $k=2$ **consecutive annotated
observations** that are strictly below the threshold).

- Requiring two observations stops one noisy frame from counting as a failure.
- If no such run exists, `first_failure_frame` is `null` and
  `failure_censored` is `true`: the clip "survived" until its end.

### 3.2 Recovery

After a failure, recovery is the first frame of the first run of 2 consecutive
annotated observations with $J \ge 0.5$ (the search starts after
$t_{\text{fail}}$). Recovery is only defined if a failure happened.

### 3.3 Example

| Frame | 0 | 5 | 10 | 15 | 20 |
| --- | --- | --- | --- | --- | --- |
| $J$ | 0.9 | **0.4** | **0.3** | 0.8 | 0.9 |

Frames 5 and 10 are both below 0.5, so $t_{\text{fail}} = 5$. Frames 15 and 20
are both at or above 0.5, so `recovery_frame` $= 15$. `worst_j` $= 0.3$.

### 3.4 How to interpret failure times

- **Time to failure** answers research question (i): which disruption breaks
  SAM 2 earliest. Align clips by their disruption event time and compare
  $t_{\text{fail}} - t_{\text{event}}$ (in seconds, using the clip's fps).
- **Abrupt vs. gradual** (question ii): look at the continuous $J$ curve around
  $t_{\text{fail}}$. A sharp drop together with low TIoU is abrupt; a slow
  decline together with high TIoU is gradual drift.
- **Recoverable** (question ii): a recovery frame means SAM 2's memory found the
  target again. No recovery means the error was permanent.
- **Censored clips** did not fail. Do not replace their failure time with the
  clip length and then average; that would make the model look worse than it
  is. Report the number of censored clips, or use a survival method such as a
  Kaplan–Meier curve if you aggregate failure times.
- **Sampling.** Here every frame is annotated, so "two consecutive
  observations" are two consecutive frames at the clip's frame rate (VOST is
  sampled at 5 fps, so a failure shorter than 0.2 s can still be missed). With
  GT only every 10th frame, the failure would have to last about 10–20 frames,
  and a short failure *between* two annotated frames could be missed.
- The binary rule is only a summary. Always keep and show the continuous curves
  (`j_curve.png`, `f_curve.png`, `j_and_f_curve.png`).

---

## 4. Across clips (`reproduce_results.py`)

These numbers combine the per-clip files of all runs into the report's tables
and figures. Let $t_{\text{ev}}$ be the clip's event frame (0 for controls,
which have no event) and fps its frame rate.

| Number | Definition |
| --- | --- |
| Time to failure $\Delta$ | $(t_{\text{fail}} - t_{\text{ev}})/\text{fps}$ seconds. It is negative when the clip fails before the event. |
| Initialization failure | A `grounded_sam2` run with no detection, or with $J_{t_0} < 0.5$. |
| Fail | Share of clips with a first failure. For `grounded_sam2` only correctly initialized clips count, so a detector error is not counted twice. |
| Median $\Delta$ | Median over the clips that failed; censored clips are left out. |
| Recovered | Share of failed clips with a recovery frame. |
| Abrupt | Share of failed clips whose temporal IoU at $t_{\text{fail}}$ is below 0.5; the others drifted gradually. |
| FP persistence | Per clip, the share of annotated frames with empty GT where the prediction is not empty; then the mean over clips that have such frames. |
| `J&F`, worst `J`, area error, fragments, temporal IoU | Mean of the per-clip summary values over **all** clips, including initialization failures (end-to-end score). |
| Initialization vs. propagation | On clips where `grounded_sam2` initialized correctly: mean of `J&F(grounded_sam2) − J&F(sam2)`. |
| Prompt sensitivity | For `sam2_shift10` and `sam2_scale120`: mean `J&F` change against the verified box, the number of clips whose failure status changes (fails ↔ censored), and the mean change of $\Delta$ on clips that fail in both runs. |
| `J&F` across the video (Fig. 2) | Per category and pipeline, each clip's `J&F` is averaged in ten bins of position in the clip (frame / clip length); the curve is the mean and the band the interquartile range over clips. |
| Survival curve (Fig. 3) | Kaplan–Meier estimate of the share of `sam2` clips that have not failed yet, against time since the event. A censored clip stays at risk until its last annotated frame. |
| `J` vs. event (`j_vs_event.pdf`, not in the report) | Per category, each `sam2` clip's `J` is averaged in 0.5 s bins of time since the event; the curve is the median and interquartile range over clips, drawn where at least 2 clips have data. |

---

## 5. Reading the metrics together

| Pattern | Likely failure mode |
| --- | --- |
| Low $J$ at frame 0 | Initialization failure (wrong box or wrong detection). |
| $J\to0$, AreaErr $\to -1$ | Target loss: the mask disappears. |
| Low $J$, AreaErr $\gg 0$ | Background takeover or over-segmentation. |
| Low $J$, AreaErr $\approx 0$ | Identity switch / mask on a similar object. |
| Good $J$, low $F$, high TIoU | Boundary lag: the mask does not follow shape change. |
| Fragments $> 1$ while GT is one piece | Mask fragmentation. |
| $J = 0$ on frames where GT is empty | False-positive persistence after the target leaves. |

These patterns are hints for the human-authored qualitative labels, not
automatic labels. Borderline cases need a second reviewer, as the protocol
says.
