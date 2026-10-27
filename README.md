# Breaking SAM 2

An inference-only stress-test benchmark that measures *when*, *how*, and
*under which transformation* SAM 2 video object segmentation fails: topology
and state change (splitting, pouring, melting, smoke), extreme scale and
viewpoint change, and illumination change. Two conditions are compared:

1. `sam2`: a manually verified frame-0 box → SAM 2 propagation.
2. `grounded_sam2`: a fixed text prompt → Grounding DINO box → SAM 2
   propagation.

A Grounding DINO miss or wrong box is an *initialization* failure and is
reported separately from SAM 2 *propagation* failures. The report is
[`report.pdf`](report.pdf) (source `report.tex`).

## Code map

| File | Functions | Purpose |
| --- | --- | --- |
| `reproduce_results.py` | `download_missing`, `run_all`, `print_report`, `plot_figures`, `plot_examples`, … | Downloads the 52 clips, runs every condition, prints the report's numbers, and draws its figures. |
| `src/main.py` | `main` | Command line for one condition on one clip. |
| `src/inference.py` | `run_pipeline`, `write_comparison_video` | One run: frames → initialization → SAM 2 → masks, metrics, `run.json`, comparison video. |
| `src/model.py` | `sam2_pipeline`, `grounding_dino_pipeline` | The two models, one function each. |
| `src/metrics.py` | `intersection_over_union`, `boundary_f_measure`, `evaluate_predictions` | Mask metrics, the failure rule, and the `metrics/` files. |
| `src/utils.py` | `load_config`, `list_frame_paths`, `read_binary_mask` | The helpers that several modules share. |
| `configs/benchmark.yaml` | | Model IDs, checkpoint URL, and all thresholds. |
| `configs/clips.csv` | | The fixed clip list: box, prompt, frame rate, event frame. |

One run follows this path:
`main` → `run_pipeline` → (`grounding_dino_pipeline` →) `sam2_pipeline` →
`evaluate_predictions` → `write_comparison_video`.

## Installation

Use Python 3.10 or newer. Install a CUDA build of PyTorch that matches your
driver first, then the rest:

```bash
python -m venv .venv
source .venv/bin/activate
# The default PyPI wheel targets CUDA 13, which needs a newer NVIDIA driver.
# Pick the index that matches the "CUDA Version" shown by nvidia-smi
# (our A100 server: driver 560 / CUDA 12.6 -> cu126).
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

The second `pip install` builds SAM 2 from source and takes about 15 minutes.
Tested with Python 3.12 and 3.13.

Run every command from the repository root. On first use, the code downloads
the SAM 2.1 Hiera B+ checkpoint to `checkpoints/sam2.1_hiera_base_plus.pt` and
Grounding DINO to `checkpoints/huggingface/`. URLs and locations are set in
`configs/benchmark.yaml`; `checkpoints/` is ignored by Git.

## Reproduce the results

One command downloads any missing data, runs every condition on all 52 clips,
and prints every number used in the report:

```bash
python reproduce_results.py
```

For each clip it runs four conditions into
`outputs/reproduce/<clip>__<condition>/`: `sam2` (verified box),
`grounded_sam2` (text prompt), and two `sam2` runs with a changed box for the
prompt-sensitivity question (`sam2_shift10`: moved by 10% of the box size;
`sam2_scale120`: 20% larger around the center). Finished runs are reused, so
an interrupted job can be restarted. It then prints Table 1 (per category:
`J&F`, failure share, median time from event to failure, Grounding DINO
initialization failures), recovery, abrupt vs. gradual failures, area error,
fragmentation, temporal IoU, false-positive persistence, and prompt
sensitivity. It also writes `outputs/reproduce/results.csv` and the figures
`figures/examples.pdf` (report Fig. 1), `figures/jf_progress.pdf` (Fig. 2),
`figures/survival.pdf` (Fig. 3), and `figures/j_vs_event.pdf` (not in the
report). [`docs/FIGURES.md`](docs/FIGURES.md) explains how to read them.

The runs take about 80 minutes on one H200 GPU, plus the downloads. The report
was made on an A100. On another GPU the numbers can differ slightly: on an
H200, with the same inputs and software, category means changed by at most
0.2 `J&F` points, and all failure counts and median times stayed the same.

## Data

The benchmark uses 14 topology/state clips from M3-VOS and VOST, and 14
scale, 10 illumination, and 14 control clips from DAVIS 2017 and VOT-ST2022. To download only the
data:

```bash
python reproduce_results.py --download-only
```

DAVIS comes as one ~800 MB archive; M3-VOS, VOST, and VOT are downloaded clip
by clip. [`docs/DATA.md`](docs/DATA.md) lists every clip, explains why it was
chosen, and describes the folder layout under `data/` (ignored by Git).

## Run one clip

Text prompt (Grounded-SAM 2) on a frame folder, with ground truth:

```bash
python -m src.main \
  --input data/davis/extracted/DAVIS/JPEGImages/480p/blackswan \
  --annotations data/davis/extracted/DAVIS/Annotations/480p/blackswan \
  --object-id 1 \
  --fps 24 \
  --pipeline grounded_sam2 \
  --prompt "black swan" \
  --output-dir outputs/blackswan-grounded
```

Verified box (SAM 2) on a video file, without ground truth:

```bash
python -m src.main \
  --input input.mp4 \
  --pipeline sam2 \
  --box 100 80 400 350 \
  --output-dir outputs/manual-box
```

- `--input` is a video file or a folder of frames with numeric names
  (`00000.jpg`, `00001.png`, …). `--fps` sets the frame rate of a folder
  (default 30; DAVIS 24, M3-VOS 30, VOST 5, VOT 30); a video uses its own.
- `--output-dir` must not exist yet, so every run needs a new one. Keep
  `sam2` and `grounded_sam2` runs in separate directories.
- `--annotations` is a folder of numeric ground-truth masks. Frame 0 must have
  a non-empty mask. DAVIS, M3-VOS, VOST, and VOT masks are indexed (0 =
  background, one positive ID per object); `--object-id` selects the target.

Each run writes `frames/` (the input as numeric JPEGs), `predictions/` (one
binary PNG mask per frame), `run.json` (settings and detections), and
`comparison_<input name>.mp4`. The video shows the frame and the prediction,
plus the ground truth as a middle panel when `--annotations` is given.

Annotated runs also write `metrics/`:

- `frame_metrics.csv`: continuous per-frame `J`, `F`, `J&F`, signed area
  error, fragments, and temporal IoU. Frames without ground truth are
  skipped, never treated as empty masks.
- `summary_metrics.json`: initialization quality, mean and worst scores, and
  the first failure / recovery frames. The failure rule is the first of two
  consecutive annotated frames with `J < 0.5`.
- `j_curve.png`, `f_curve.png`, `j_and_f_curve.png`.

[`docs/METRICS.md`](docs/METRICS.md) explains every metric. To evaluate saved
predictions again without running a model:

```python
from src.metrics import evaluate_predictions
from src.utils import load_config

evaluate_predictions(
    "data/davis/extracted/DAVIS/Annotations/480p/blackswan",
    "outputs/blackswan-grounded/predictions",
    "outputs/blackswan-grounded/metrics-again",
    load_config("configs/benchmark.yaml")["thresholds"],
    object_id=1,
)
```
