# Figures: how to read them

`python reproduce_results.py` draws four figures into `figures/`
(`plot_examples` and `plot_figures` in `reproduce_results.py`):

| File | Report | Runs used | Question it answers |
| --- | --- | --- | --- |
| `examples.pdf` | Fig. 1 | `sam2` frames + GT | What does each challenge category look like? |
| `jf_progress.pdf` | Fig. 2 | `sam2` and `grounded_sam2` | How does quality change from the prompted frame to the end of the clip? |
| `survival.pdf` | Fig. 3 | `sam2` | What share of clips has not failed yet *t* seconds after the event? |
| `j_vs_event.pdf` | not in the report | `sam2` | How does `J` change around the event? |

Figures that use only `sam2` (manually verified first-frame box) contain no
Grounding DINO initialization failures, so every drop in them is a SAM 2
*propagation* failure. Categories: topology/state (14 clips),
scale/viewpoint (14), illumination (10), control (14). The results these
figures show are discussed in the report (Sec. 3); this file explains how to
read them.

## Fig. 1 — `examples.pdf`: one example clip per category

Each column is one clip: frame 0 (top, where SAM 2 is prompted) and a later
frame after the event (bottom), with the GT outline in green and the time in
seconds. The clips and frames (`0109_break_glass_1` frame 59, `rallye` 39,
`bag` 240, `blackswan` 49) were picked by eye. The figure only illustrates the
categories and does not enter any number.

## Fig. 2 — `jf_progress.pdf`: `J&F` across the video

**What is plotted.** The x-axis is the position in the clip: frame index
divided by the clip length, in ten 10% bins (0% = prompted frame, 100% = last
frame). For each clip, `J&F` is averaged inside each bin. Then, per category
and bin, the **line** is the mean over clips and the **band** is the
interquartile range (IQR, 25th to 75th percentile). Ticks at the bottom mark
the event position of each clip (controls have no event). Left: `sam2`.
Right: `grounded_sam2`, including initialization failures (all-empty masks
give `J&F = 0`).

**Why position instead of time.** Clip lengths differ a lot (DAVIS 1.5–4.1 s,
VOST up to 35 s). With position on the x-axis, every clip counts in every bin;
in an event-aligned plot (see `j_vs_event.pdf`) the late bins rest on a few
long clips.

**How to read it.**

- The first bin shows the initialization quality. With the verified box all
  categories start at a similar level, so later differences come from
  propagation.
- A drop after a category's ticks suggests that the event hurts tracking.
- The gap between left and right in the first bin comes from Grounding DINO
  initialization failures; compare it with the gap at the end to see whether
  text initialization also changes propagation.

## Fig. 3 — `survival.pdf`: survival until the first failure

**What is plotted.** A Kaplan–Meier survival curve per category, for `sam2`.
The x-axis is time since the event, `(frame − t_ev) / fps` in seconds
(`t_ev` from `configs/clips.csv`; controls use `t_ev = 0`, so for them it is
time since the first frame). The y-axis ("share not yet failed") is the
estimated share of clips that have **not** reached their first failure at
time *t*. The failure time of a clip is the pre-registered first failure: the
first of two consecutive annotated frames with `J < 0.5`.

**Why Kaplan–Meier.** Many clips never fail before they end. These clips are
**censored**: we only know that they survived *at least* until their last
frame. Dropping them would make SAM 2 look worse; counting the clip end as a
failure would be wrong too. In Kaplan–Meier, a censored clip counts as "at
risk" until its last annotated frame and then leaves the risk set without
causing a drop. At each failure time the curve is multiplied by
`1 − failures / clips still at risk`.

**How to read it.**

- The curve starts at 1.0 and is a step function: every downward step is one
  (or more) failures.
- The **height of a step** depends on how many clips are still at risk. Early
  steps are small (many clips at risk); late steps are big (few clips left).
- A curve that **drops early and low** means the disruption breaks SAM 2 early
  and often. A curve that **stays at 1.0** means no clip failed.
- The x-value where a curve crosses 0.5 is the Kaplan–Meier **median time to
  failure**.
- A step at a **negative** time is a failure *before* the annotated event.

**Late steps rest on few clips.** DAVIS clips are at most 4.1 s long, so after
about 4 s only the long VOT, M3-VOS, and VOST clips are still followed:

- **Topology/state** (12 of 14 fail): the first step at −1.5 s is
  `0030_cut_celery_2`, which fails before `t_ev` because the celery is almost
  hidden by the hand. The last step to 0 at 17.6 s is `3316_peel_onion`, the
  only clip still at risk; the two censored clips (`0158_melt_glue_block_1`,
  `0239_exhaust_smoke_1`) left the risk set at 9.7 and 10.3 s. So "0%
  survival" at the end is a small-sample effect, not proof that every
  topology clip eventually fails.
- **Illumination** (5 of 10 fail): the five censored clips end by 3.2 s, so
  the last two steps (`fernando` at 4.7 s, `shaking` at 6.5 s) rest on two
  clips and one clip.
- **Scale/viewpoint** (1 of 14 fails): `helicopter` fails at 9.5 s, when only
  it and `graduate` are still at risk, so the curve drops to 0.5.
- **Control** (0 of 14 fail): flat at 1.0.

## `j_vs_event.pdf`: `J` aligned by the event (not in the report)

**What is plotted.** Same x-axis as Fig. 3 (time since the event). For each
`sam2` clip, `J` is averaged in 0.5 s bins from −5 to 20 s. Then, per
category and bin, the **line** is the **median** over clips and the **band**
is the IQR; a bin is drawn only when **at least 2 clips** have data there, so
a single clip is never shown as a category trend. The dashed line at
`J = 0.5` is the failure threshold. The median is used because `J` is bounded
in [0, 1] and failed clips fall near 0, which would pull a mean down strongly.

**How to read it.** A line that stays high after `x = 0` means the event did
not hurt SAM 2; a line that drops shows how fast it breaks. A wide band means
the clips in a category behave very differently. The median crossing
`J = 0.5` is only a rough visual sign: the pre-registered rule needs two
consecutive annotated frames per clip, and Fig. 3 uses that rule. Line length
depends on clip length, so the late part rests on the few long clips.

## Small samples

With 10–14 clips per category, each step and each IQR band is uncertain. The
figures describe this benchmark; they are not precise estimates for all
videos of each type.
