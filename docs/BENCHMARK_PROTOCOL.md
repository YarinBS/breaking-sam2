# Benchmark protocol

Select 15–30 clips before model inference. Include topology/state,
scale/viewpoint, illumination, and ordinary-motion control clips. Each clip is
5–10 seconds and records its dataset/source, license, target, fixed prompt,
frame rate, disruption category, and disruption event time.

Annotate the target every 5th–10th frame and more densely around the event.
Keep a small dense subset. Frame zero must have a non-empty mask and a manually
verified `xyxy` box. Missing annotations are omitted; they are never converted
to empty ground truth.

Run both conditions with fixed settings:

1. `sam2`: manually verified first-frame box → SAM 2 propagation.
2. `grounded_sam2`: fixed text prompt → Grounding DINO box → SAM 2 propagation.

Report Grounding DINO misses separately from SAM 2 propagation failures. Save
continuous `J`, `F`, `J&F`, signed area error, fragmentation, and temporal IoU.
The primary failure is the first of two consecutive annotated observations
with `J < 0.5`; clips without such a run are censored. Report recovery only
after a qualifying failure. Qualitative labels are human-authored, with a
second reviewer for borderline cases.

Pass each clip's mask directory with `--annotations`. For indexed DAVIS masks,
also pass the target's positive `--object-id`. Preserve `frame_metrics.csv` for
continuous analysis and use one `summary_metrics.json` row per clip/pipeline in
the paper's aggregate tables.

## How the repository applies this protocol

The rules above were fixed before inference and are kept unchanged. This
section records how they were carried out, including where the final
benchmark differs from the plan.

- `python reproduce_results.py` runs the whole protocol: it reads the 52
  clips from `configs/clips.csv`, runs both conditions (plus two perturbed
  boxes for prompt sensitivity), and prints the aggregate tables.
- **Deviation — clip count:** the first 26 clips (14 topology, 5 scale,
  3 illumination, 4 control) were fixed before inference. To balance the
  categories, 26 more scale, illumination, and control clips (DAVIS and
  VOT-ST2022) were added later, with the same selection rules and before any
  model was run on them, giving 52 clips (14 / 14 / 10 / 14). See
  `docs/DATA.md`.
- **Deviation — clip length:** the chosen clips are 1.5–4.1 s (DAVIS),
  2.7–28 s (VOT), 10–18 s (M3-VOS), and 10–35 s (VOST), not 5–10 s. The
  report mentions this as a limitation. Two VOT clips end before their first
  shot cut (`frames` in `configs/clips.csv`).
- **Deviation — annotation:** no own annotation was made. M3-VOS, VOST, VOT,
  and DAVIS already annotate every frame at their frame rate (30, 5, 30, and
  24 fps),
  so there is no separate dense subset. The code still skips missing
  annotations, as required.
- The verified box is the tight box around the frame-0 GT mask, checked by
  eye. The event frame `t_ev` is stored per clip; controls have none.
- A `grounded_sam2` run counts as an initialization failure when Grounding
  DINO finds nothing or `J < 0.5` on frame 0; such runs are left out of the
  propagation statistics.
