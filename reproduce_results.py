"""Reproduce the results reported in ``report.pdf``.

Usage (from the repository root, with the virtual environment active)::

    python reproduce_results.py                  # download, run, and report
    python reproduce_results.py --download-only  # only fetch missing clips

The clip list, boxes, prompts, and event frames come from ``configs/clips.csv``
so the benchmark stays fixed before any model is run. The script:

1. downloads every clip whose frames or masks are missing locally
   (DAVIS as the full archive, M3-VOS, VOST, and VOT clip by clip);
2. runs both pipelines on every clip, plus two ``sam2`` runs with a perturbed
   box for the prompt-sensitivity question, reusing finished runs so an
   interrupted job can be restarted;
3. prints Table 1 and the numbers used in the result paragraphs, and writes
   ``results.csv``, the example-frame figure, and the three temporal figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.inference import run_pipeline
from src.utils import load_config, read_binary_mask


CLIPS_CSV = Path("configs/clips.csv")
CONFIG = Path("configs/benchmark.yaml")
DATA = Path("data")
PIPELINES = ("sam2", "grounded_sam2")
# RQ (iii): two sam2 runs with a changed verified box; see run_all.
PERTURBATIONS = ("sam2_shift10", "sam2_scale120")
CATEGORIES = ("topology", "scale", "illumination", "control")
DAVIS_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
M3VOS_REPO = "Lijiaxin0111/M3_VOS"
VOST_URL = "https://tri-ml-public.s3.amazonaws.com/datasets/VOST.zip"
VOT_URL = "https://data.votchallenge.net/vot2022/sts/"
# A failure counts as abrupt when the mask changes a lot between two frames
# (low temporal IoU at the failure frame); otherwise the mask drifted slowly.
ABRUPT_TEMPORAL_IOU = 0.5


def load_clips(path: Path = CLIPS_CSV) -> list[dict[str, Any]]:
    """Read the fixed benchmark clip list.

    Args:
        path: CSV with one row per clip.

    Returns:
        Clip dictionaries with typed ``object_id``, ``box``, ``fps``, and
        ``t_ev``, plus the local ``frames_dir`` and ``annotations_dir``. Clips
        without an event (controls) get ``t_ev = 0``, so their failure times
        are measured from the first frame.

    Used by:
        :func:`main` and unit tests.
    """

    # Where each dataset keeps its frames and masks under data/.
    roots = {
        "davis": (DATA / "davis/extracted/DAVIS/JPEGImages/480p",
                  DATA / "davis/extracted/DAVIS/Annotations/480p"),
        "m3vos": (DATA / "m3vos/data/JPEGImages", DATA / "m3vos/data/Annotations"),
        "vost": (DATA / "vost_clips/JPEGImages", DATA / "vost_clips/Annotations"),
        "vot": (DATA / "vot/JPEGImages", DATA / "vot/Annotations"),
    }
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["object_id"] = int(row["object_id"])
        row["box"] = [float(value) for value in row["box"].split()]
        row["fps"] = float(row["fps"])
        row["t_ev"] = int(row["t_ev"]) if row["t_ev"] else 0
        frames_root, annotations_root = roots[row["dataset"]]
        row["frames_dir"] = frames_root / row["clip_id"]
        row["annotations_dir"] = annotations_root / row["clip_id"]
    return rows


def is_downloaded(clip: Mapping[str, Any]) -> bool:
    """Check that a clip has frames and one mask per frame.

    Comparing the two counts catches downloads that stopped half-way.

    Args:
        clip: Row from :func:`load_clips`.

    Returns:
        ``True`` when both directories exist and hold the same number of files.

    Used by:
        :func:`download_missing`, before and after downloading.
    """

    frames_dir, annotations_dir = clip["frames_dir"], clip["annotations_dir"]
    if not frames_dir.is_dir() or not annotations_dir.is_dir():
        return False
    frames = len(list(frames_dir.glob("*.jpg")))
    return frames > 0 and frames == len(list(annotations_dir.glob("*.png")))


def download_missing(clips: Sequence[Mapping[str, Any]]) -> None:
    """Download every clip that is not complete on disk.

    DAVIS comes as one ~800 MB archive. M3-VOS, VOST, and VOT are downloaded
    clip by clip, because their full archives are large (``VOST.zip`` is
    54 GB). Files that already exist are skipped, so an interrupted download
    continues.

    Args:
        clips: Rows from :func:`load_clips`.

    Used by:
        :func:`main`.
    """

    missing = [clip for clip in clips if not is_downloaded(clip)]
    by_dataset = {name: [c["clip_id"] for c in missing if c["dataset"] == name]
                  for name in ("davis", "m3vos", "vost", "vot")}

    if by_dataset["davis"]:
        import torch

        print("Downloading DAVIS 2017 (full archive)")
        archive = DATA / "davis/DAVIS-2017-trainval-480p.zip"
        if not archive.exists():
            archive.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(DAVIS_URL, str(archive))
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(DATA / "davis/extracted")

    if by_dataset["m3vos"]:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import HfHubHTTPError

        print(f"Downloading M3-VOS clips: {', '.join(by_dataset['m3vos'])}")
        # Every frame is one HuggingFace request, and anonymous users are
        # limited to 1000 requests per 5 minutes, so a rate-limit error waits
        # and retries.
        for clip_id in by_dataset["m3vos"]:
            while True:
                try:
                    snapshot_download(
                        M3VOS_REPO,
                        repo_type="dataset",
                        allow_patterns=[f"data/JPEGImages/{clip_id}/*",
                                        f"data/Annotations/{clip_id}/*"],
                        local_dir=DATA / "m3vos",
                    )
                    break
                except HfHubHTTPError as error:
                    if error.response is None or error.response.status_code != 429:
                        raise
                    print(f"HuggingFace rate limit on {clip_id}; retrying in 5 minutes")
                    time.sleep(300)

    if by_dataset["vost"]:
        from remotezip import RemoteZip

        print(f"Downloading VOST clips: {', '.join(by_dataset['vost'])}")
        # remotezip reads only the needed files from the remote zip with HTTP
        # range requests, over one connection (parallel SSL connections failed
        # on our server). VOST names frames by their 30 fps index (frame00000,
        # frame00006, ...), but predictions and GT are matched by consecutive
        # numeric names, so files are renamed to 00000, 00001, ...
        with RemoteZip(VOST_URL) as archive:
            names = archive.namelist()
            for clip_id in by_dataset["vost"]:
                for kind, suffix in (("JPEGImages", ".jpg"), ("Annotations", ".png")):
                    members = sorted(
                        name for name in names
                        if name.startswith(f"VOST/{kind}/{clip_id}/") and name.endswith(suffix)
                    )
                    destination = DATA / "vost_clips" / kind / clip_id
                    destination.mkdir(parents=True, exist_ok=True)
                    for index, member in enumerate(members):
                        target = destination / f"{index:05d}{suffix}"
                        if not target.exists():
                            target.write_bytes(archive.read(member))
                print(f"VOST {clip_id}: {len(members)} frames")

    if by_dataset["vot"]:
        import io
        import urllib.request

        import cv2

        print(f"Downloading VOT clips: {', '.join(by_dataset['vot'])}")
        with urllib.request.urlopen(VOT_URL + "description.json") as response:
            sequences = {s["name"]: s for s in json.load(response)["sequences"]}
        for clip in (c for c in clips if c["clip_id"] in by_dataset["vot"]):
            sequence = sequences[clip["clip_id"]]
            # Some VOT clips are cut before their first shot cut (the `frames`
            # column), because a cut is a different disruption than the one
            # the clip was chosen for.
            count = int(clip["frames"]) if clip["frames"] else sequence["length"]
            clip["frames_dir"].mkdir(parents=True, exist_ok=True)
            clip["annotations_dir"].mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(VOT_URL + sequence["channels"]["color"]["url"]) as response:
                bundle = zipfile.ZipFile(io.BytesIO(response.read()))
            # VOT names frames 00000001.jpg, ...; like VOST they are renamed to
            # 00000, 00001, ... so that masks match frames by numeric name.
            members = sorted(name for name in bundle.namelist() if name.endswith(".jpg"))
            for index, member in enumerate(members[:count]):
                (clip["frames_dir"] / f"{index:05d}.jpg").write_bytes(bundle.read(member))
            with urllib.request.urlopen(VOT_URL + sequence["annotations"]["url"]) as response:
                lines = zipfile.ZipFile(io.BytesIO(response.read())).read("groundtruth.txt")
            # Each line is one mask "m<x>,<y>,<w>,<h>,<run lengths>": run
            # lengths alternate background/object pixels, row by row, inside
            # the offset box. Masks are saved with value 1 (object_id 1).
            width, height = sequence["width"], sequence["height"]
            for index, line in enumerate(lines.decode().splitlines()[:count]):
                x, y, w, h, *runs = (int(v) for v in line[1:].split(",") if v)
                mask = np.zeros((height, width), np.uint8)
                patch = np.repeat(np.arange(len(runs)) % 2, runs).astype(np.uint8)
                patch = np.pad(patch, (0, w * h - patch.size)).reshape(h, w)
                mask[y:y + h, x:x + w] = patch[:height - y, :width - x]
                cv2.imwrite(str(clip["annotations_dir"] / f"{index:05d}.png"), mask)
            print(f"VOT {clip['clip_id']}: {count} frames")

    still_missing = [clip["clip_id"] for clip in clips if not is_downloaded(clip)]
    if still_missing:
        raise RuntimeError(f"Clips are still incomplete: {', '.join(still_missing)}")


def run_all(
    clips: Sequence[Mapping[str, Any]],
    output_dir: Path,
    config: Mapping[str, Any],
) -> None:
    """Run every condition on every clip, skipping runs that already finished.

    Args:
        clips: Rows from :func:`load_clips`.
        output_dir: Parent directory; each run goes to
            ``<clip_id>__<condition>/``.
        config: Loaded benchmark configuration.

    Used by:
        :func:`main` and unit tests.
    """

    for clip in clips:
        # RQ (iii): the verified box moved right/down by 10% of its size, and
        # enlarged by 20% around its center. sam2_pipeline clips it to the image.
        x1, y1, x2, y2 = clip["box"]
        dx, dy = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
        conditions = {
            "sam2": ("sam2", clip["box"]),
            "grounded_sam2": ("grounded_sam2", clip["box"]),
            "sam2_shift10": ("sam2", [x1 + dx, y1 + dy, x2 + dx, y2 + dy]),
            "sam2_scale120": ("sam2", [x1 - dx, y1 - dy, x2 + dx, y2 + dy]),
        }
        for condition, (pipeline, box) in conditions.items():
            run_dir = output_dir / f"{clip['clip_id']}__{condition}"
            if (run_dir / "run.json").exists():
                continue
            if run_dir.exists():
                raise RuntimeError(
                    f"{run_dir} is an unfinished run; delete it and start again"
                )
            print(f"Running {condition} on {clip['clip_id']}")
            run_pipeline(
                input_path=clip["frames_dir"],
                output_dir=run_dir,
                config=config,
                pipeline=pipeline,
                prompt=clip["prompt"],
                box=box,
                frames_per_second=clip["fps"],
                annotations_dir=clip["annotations_dir"],
                object_id=clip["object_id"],
            )


def clip_result(clip: Mapping[str, Any], run_dir: Path, condition: str) -> dict[str, Any]:
    """Collect the per-clip numbers used in the report from one finished run.

    Args:
        clip: Row from :func:`load_clips`.
        run_dir: Output directory written by :func:`src.inference.run_pipeline`.
        condition: Name of the condition, stored as ``pipeline`` in the row.

    Returns:
        One flat result row. ``delta_s`` is the time from the event to the
        first failure in seconds (``None`` when censored). ``init_failure`` is
        only defined for ``grounded_sam2``: no detection, or ``J < 0.5`` on
        frame 0.

    Used by:
        :func:`main` and unit tests.
    """

    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (run_dir / "metrics/summary_metrics.json").read_text(encoding="utf-8")
    )
    with (run_dir / "metrics/frame_metrics.csv").open(encoding="utf-8") as handle:
        frames = list(csv.DictReader(handle))
    frame_by_index = {int(row["frame_index"]): row for row in frames}
    failure = summary["first_failure_frame"]
    init_failure = None
    if run["pipeline"] == "grounded_sam2":
        init_failure = not run["detections"] or summary["initialization_j"] < 0.5

    abrupt = None
    if failure is not None and frame_by_index[failure]["temporal_iou"] not in ("", None):
        abrupt = float(frame_by_index[failure]["temporal_iou"]) < ABRUPT_TEMPORAL_IOU

    # False-positive persistence: how often a mask remains after the target
    # has left the frame (annotated frames with empty GT).
    absent = [row for row in frames if row["target_present"] == "False"]
    fp_persistence = (
        float(np.mean([int(row["predicted_area"]) > 0 for row in absent]))
        if absent else None
    )
    last_frame = max(frame_by_index)
    return {
        "clip_id": clip["clip_id"],
        "category": clip["category"],
        "pipeline": condition,
        "fps": clip["fps"],
        "t_ev": clip["t_ev"],
        "mean_j": summary["mean_j"],
        "mean_f": summary["mean_f"],
        "mean_j_and_f": summary["mean_j_and_f"],
        "worst_j": summary["worst_j"],
        "initialization_j": summary["initialization_j"],
        "mean_absolute_area_error": summary["mean_absolute_area_error"],
        "mean_fragments": summary["mean_fragments"],
        "mean_temporal_iou": summary["mean_temporal_iou"],
        "failure_frame": failure,
        "delta_s": None if failure is None else (failure - clip["t_ev"]) / clip["fps"],
        "followup_s": (last_frame - clip["t_ev"]) / clip["fps"],
        "recovered": summary["recovered"],
        "abrupt": abrupt,
        "fp_persistence": fp_persistence,
        "init_failure": init_failure,
        "j_curve": [(int(row["frame_index"]), float(row["j"])) for row in frames],
        "jf_curve": [(int(row["frame_index"]), float(row["j_and_f"])) for row in frames],
        "n_frames": last_frame + 1,
    }


def mean_or_none(values: Sequence[float | None]) -> float | None:
    """Average the defined values, or return ``None`` when there are none.

    Used by:
        :func:`summarize`.
    """

    defined = [value for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def summarize(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate result rows of one category and pipeline.

    For ``grounded_sam2``, failure statistics use only correctly initialized
    clips, so a detector miss is reported as an initialization failure and not
    also as a propagation failure. ``J&F`` uses all clips (end-to-end score).

    Args:
        results: Rows from :func:`clip_result`.

    Returns:
        Clip count, mean ``J&F``, failure share, median ``delta_s``, recovery
        share, abrupt share, initialization-failure share, and diagnostics.

    Used by:
        :func:`main` and unit tests.
    """

    propagated = [row for row in results if not row["init_failure"]]
    failed = [row for row in propagated if row["failure_frame"] is not None]
    init = [row["init_failure"] for row in results if row["init_failure"] is not None]
    return {
        "clips": len(results),
        "j_and_f": mean_or_none([row["mean_j_and_f"] for row in results]),
        "fail_share": len(failed) / len(propagated) if propagated else None,
        "median_delta_s": (
            float(np.median([row["delta_s"] for row in failed])) if failed else None
        ),
        "recovery_share": (
            float(np.mean([row["recovered"] for row in failed])) if failed else None
        ),
        "abrupt_share": mean_or_none(
            [float(row["abrupt"]) for row in failed if row["abrupt"] is not None]
        ),
        "init_failure_share": float(np.mean(init)) if init else None,
        "worst_j": mean_or_none([row["worst_j"] for row in results]),
        "area_error": mean_or_none([row["mean_absolute_area_error"] for row in results]),
        "fragments": mean_or_none([row["mean_fragments"] for row in results]),
        "temporal_iou": mean_or_none([row["mean_temporal_iou"] for row in results]),
        "fp_persistence": mean_or_none([row["fp_persistence"] for row in results]),
    }


def kaplan_meier(
    durations: Sequence[float], observed: Sequence[bool]
) -> tuple[list[float], list[float]]:
    """Estimate the share of clips that have not failed yet over time.

    Kaplan--Meier handles censored clips (no failure before the clip ends)
    correctly, instead of treating the clip end as a failure time.

    Args:
        durations: Time to failure, or to the end of the clip when censored.
        observed: ``True`` for a failure, ``False`` for a censored clip.

    Returns:
        Step times and survival values, starting at ``1.0`` at the event
        (``t = 0``) or at the earliest failure if it happened before the event.

    Used by:
        :func:`plot_figures` and unit tests.
    """

    pairs = sorted(zip(durations, observed))
    times, survival = [min(0.0, *durations)], [1.0]
    current = 1.0
    for time_point in sorted({d for d, event in pairs if event}):
        failures = sum(1 for d, event in pairs if event and d == time_point)
        at_risk = sum(1 for d, _ in pairs if d >= time_point)
        current *= 1 - failures / at_risk
        times.append(time_point)
        survival.append(current)
    return times, survival


def plot_figures(results: Sequence[Mapping[str, Any]], figures_dir: Path) -> None:
    """Draw the temporal figures of the report.

    ``jf_progress.pdf`` shows the mean and IQR of ``J&F`` against the position
    in the video (10% bins), per category, for both pipelines; ticks at the
    bottom mark each clip's event. ``j_vs_event.pdf`` shows the median and IQR
    of ``J`` against time since the event, per category, using 0.5 s bins.
    ``survival.pdf`` shows the Kaplan--Meier curve of time from the event to
    the first failure. The last two use only the ``sam2`` (verified box) runs.

    Args:
        results: Rows from :func:`clip_result`.
        figures_dir: Output directory.

    Used by:
        :func:`main`.
    """

    figures_dir.mkdir(parents=True, exist_ok=True)
    sam2 = [row for row in results if row["pipeline"] == "sam2"]

    # Normalizing time to the clip length lets every clip count in every bin,
    # unlike the event-aligned plot, whose tails rest on the few long clips.
    edges = np.linspace(0, 1, 11)
    figure, axes = plt.subplots(1, 2, figsize=(7, 2.4), sharey=True)
    for axis, pipeline in zip(axes, PIPELINES):
        for k, category in enumerate(CATEGORIES):
            rows = [row for row in results
                    if row["pipeline"] == pipeline and row["category"] == category]
            per_clip = np.full((len(rows), len(edges) - 1), np.nan)
            for i, row in enumerate(rows):
                progress = np.array([f / row["n_frames"] for f, _ in row["jf_curve"]])
                scores = np.array([jf for _, jf in row["jf_curve"]])
                slot = np.minimum(np.digitize(progress, edges) - 1, len(edges) - 2)
                for b in range(len(edges) - 1):
                    if np.any(slot == b):
                        per_clip[i, b] = scores[slot == b].mean()
            centers = 100 * (edges[:-1] + edges[1:]) / 2
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                q1, q3 = np.nanpercentile(per_clip, [25, 75], axis=0)
                mean = np.nanmean(per_clip, axis=0)
            line, = axis.plot(centers, mean, marker="o", markersize=2.5, label=category)
            axis.fill_between(centers, q1, q3, alpha=0.15, color=line.get_color())
            if category != "control":
                events = [100 * row["t_ev"] / row["n_frames"] for row in rows]
                axis.plot(events, np.full(len(events), 0.03 + 0.04 * k), "|",
                          color=line.get_color(), markersize=5)
        axis.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        axis.set(xlabel="position in video (%)", xlim=(0, 100), ylim=(0, 1),
                 title="SAM 2 (verified box)" if pipeline == "sam2" else "Grounded-SAM 2 (text)")
        axis.title.set_fontsize(8)
    axes[0].set_ylabel("J&F")
    axes[0].legend(fontsize=6, loc="lower left", bbox_to_anchor=(0, 0.15))
    figure.tight_layout()
    figure.savefig(figures_dir / "jf_progress.pdf")
    plt.close(figure)
    bins = np.arange(-5.0, 20.5, 0.5)

    figure, axis = plt.subplots(figsize=(5, 3.2))
    for category in CATEGORIES:
        rows = [row for row in sam2 if row["category"] == category]
        per_clip = np.full((len(rows), len(bins) - 1), np.nan)
        for i, row in enumerate(rows):
            times = np.array([(f - row["t_ev"]) / row["fps"] for f, _ in row["j_curve"]])
            scores = np.array([j for _, j in row["j_curve"]])
            slot = np.digitize(times, bins) - 1
            for b in range(len(bins) - 1):
                if np.any(slot == b):
                    per_clip[i, b] = scores[slot == b].mean()
        if not rows:
            continue
        valid = np.sum(~np.isnan(per_clip), axis=0) >= 2
        centers = (bins[:-1] + bins[1:]) / 2
        # Bins without any clip give all-NaN columns; they are masked by `valid`.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            q1, median, q3 = np.nanpercentile(per_clip, [25, 50, 75], axis=0)
        axis.plot(centers[valid], median[valid], label=category)
        axis.fill_between(centers[valid], q1[valid], q3[valid], alpha=0.2)
    axis.axvline(0, color="gray", linestyle=":")
    axis.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    axis.set(xlabel="time since event (s)", ylabel="J", ylim=(0, 1))
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(figures_dir / "j_vs_event.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5, 3.2))
    for category in CATEGORIES:
        rows = [row for row in sam2 if row["category"] == category]
        if not rows:
            continue
        durations = [row["delta_s"] if row["delta_s"] is not None else row["followup_s"]
                     for row in rows]
        times, survival = kaplan_meier(durations, [row["delta_s"] is not None for row in rows])
        times.append(max(durations))
        survival.append(survival[-1])
        axis.step(times, survival, where="post", label=category)
    axis.axvline(0, color="gray", linestyle=":")
    axis.set(xlabel="time since event (s)", ylabel="share not yet failed", ylim=(0, 1.05))
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(figures_dir / "survival.pdf")
    plt.close(figure)


def plot_examples(clips: Sequence[Mapping[str, Any]], run_root: Path, figures_dir: Path) -> None:
    """Draw Fig. 1 of the report: one example clip per category.

    Each column shows frame 0 and one later frame with the GT outline, so the
    reader sees how the object changes. The clips and frames were picked by
    eye from contact sheets; they only illustrate the categories and do not
    enter any number.

    Args:
        clips: Rows from :func:`load_clips`.
        run_root: Directory with the finished ``sam2`` runs (frames and the
            GT path in ``run.json``).
        figures_dir: Output directory.

    Used by:
        :func:`main`.
    """

    # (category, clip, later frame): shattering, approach, darkness, no change.
    examples = [("topology", "0109_break_glass_1", 59), ("scale", "rallye", 39),
                ("illumination", "bag", 240), ("control", "blackswan", 49)]
    by_id = {clip["clip_id"]: clip for clip in clips}
    figure, axes = plt.subplots(2, len(examples), figsize=(7, 2.3))
    for column, (category, clip_id, later) in enumerate(examples):
        clip, run_dir = by_id[clip_id], run_root / f"{clip_id}__sam2"
        annotations = Path(json.loads((run_dir / "run.json").read_text())["annotations"])
        # GT names differ per dataset (00000.png vs 0000000.png); match by number.
        masks = {int(path.stem): path for path in annotations.glob("*.png")}
        for row, frame in enumerate((0, later)):
            axis = axes[row, column]
            axis.imshow(plt.imread(run_dir / "frames" / f"{frame:05d}.jpg"), aspect="auto")
            axis.contour(read_binary_mask(masks[frame], clip["object_id"]),
                         levels=[0.5], colors="lime", linewidths=0.8)
            axis.text(0.02, 0.96, f"{frame / clip['fps']:.1f} s", transform=axis.transAxes,
                      va="top", fontsize=7, color="white",
                      bbox={"facecolor": "black", "alpha": 0.6, "pad": 1, "linewidth": 0})
            axis.set_axis_off()
        axes[0, column].set_title(f"{category}: {clip_id}", fontsize=7)
    figure.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.92,
                           wspace=0.02, hspace=0.03)
    figure.savefig(figures_dir / "examples.pdf", dpi=200)
    plt.close(figure)


def fmt(value: float | None, scale: float = 1.0, digits: int = 1, suffix: str = "") -> str:
    """Format a number for the printed tables, with ``-`` for missing values.

    Used by:
        :func:`print_report`.
    """

    return "-" if value is None else f"{value * scale:.{digits}f}{suffix}"


def print_report(results: Sequence[Mapping[str, Any]]) -> None:
    """Print Table 1 and the numbers used in the result paragraphs.

    Args:
        results: Rows from :func:`clip_result`.

    Used by:
        :func:`main`.
    """

    groups = [(category, [r for r in results if r["category"] == category])
              for category in CATEGORIES] + [("all", list(results))]

    print("\nTable 1: J&F x100, Fail = share of (initialized) clips with a first failure,")
    print("median Delta = event -> failure (s), Init = Grounded-SAM 2 initialization failures")
    print(f"{'category':<13}{'clips':>6} | {'sam2 J&F':>9}{'Fail':>6}{'Delta':>7} |"
          f" {'gsam2 J&F':>9}{'Fail':>6}{'Delta':>7}{'Init':>6}")
    for name, rows in groups:
        s = summarize([r for r in rows if r["pipeline"] == "sam2"])
        g = summarize([r for r in rows if r["pipeline"] == "grounded_sam2"])
        print(f"{name:<13}{s['clips']:>6} | {fmt(s['j_and_f'], 100):>9}"
              f"{fmt(s['fail_share'], 100, 0, '%'):>6}{fmt(s['median_delta_s']):>7} |"
              f" {fmt(g['j_and_f'], 100):>9}{fmt(g['fail_share'], 100, 0, '%'):>6}"
              f"{fmt(g['median_delta_s']):>7}{fmt(g['init_failure_share'], 100, 0, '%'):>6}")

    print("\nFailure behavior and diagnostics (sam2 / grounded_sam2):")
    print(f"{'category':<13}{'recovered':>12}{'abrupt':>12}{'worst J':>12}"
          f"{'|area err|':>12}{'fragments':>12}{'temp. IoU':>12}{'FP persist':>12}")
    for name, rows in groups:
        s = summarize([r for r in rows if r["pipeline"] == "sam2"])
        g = summarize([r for r in rows if r["pipeline"] == "grounded_sam2"])
        cells = [
            f"{fmt(s[k], *a)}/{fmt(g[k], *a)}"
            for k, a in (("recovery_share", (100, 0, "%")), ("abrupt_share", (100, 0, "%")),
                         ("worst_j", (1, 2)), ("area_error", (1, 2)),
                         ("fragments", (1, 1)), ("temporal_iou", (1, 2)),
                         ("fp_persistence", (100, 0, "%")))
        ]
        print(f"{name:<13}" + "".join(f"{cell:>12}" for cell in cells))

    # RQ iv: compare the two pipelines only where text initialization worked.
    sam2 = {r["clip_id"]: r for r in results if r["pipeline"] == "sam2"}
    grounded = [r for r in results if r["pipeline"] == "grounded_sam2"]
    init_failed = [r["clip_id"] for r in grounded if r["init_failure"]]
    paired = [r["mean_j_and_f"] - sam2[r["clip_id"]]["mean_j_and_f"]
              for r in grounded if not r["init_failure"] and r["clip_id"] in sam2]
    print(f"\nGrounding DINO initialization failures: {len(init_failed)} of {len(grounded)}"
          f" clips ({', '.join(init_failed) or 'none'})")
    print(f"Correctly initialized clips: mean J&F(grounded_sam2) - J&F(sam2) = "
          f"{fmt(mean_or_none(paired), 100)} over {len(paired)} clips")

    # RQ iii: how much a small change of the verified box changes the result.
    print("\nPrompt sensitivity (verified box -> perturbed box, sam2):")
    for condition in PERTURBATIONS:
        rows = [r for r in results if r["pipeline"] == condition and r["clip_id"] in sam2]
        diffs = [r["mean_j_and_f"] - sam2[r["clip_id"]]["mean_j_and_f"] for r in rows]
        flipped = [r["clip_id"] for r in rows
                   if (r["failure_frame"] is None) != (sam2[r["clip_id"]]["failure_frame"] is None)]
        shifts = [(r["delta_s"] - sam2[r["clip_id"]]["delta_s"]) for r in rows
                  if r["delta_s"] is not None and sam2[r["clip_id"]]["delta_s"] is not None]
        print(f"  {condition:<14} mean delta J&F = {fmt(mean_or_none(diffs), 100)}, "
              f"failure status changed on {len(flipped)} of {len(rows)} clips, "
              f"mean change of failure time = {fmt(mean_or_none(shifts))} s")

    print("\nPer clip (J&F x100, first failure frame, Delta s, recovered):")
    main_rows = [r for r in results if r["pipeline"] in PIPELINES]
    for r in sorted(main_rows, key=lambda r: (CATEGORIES.index(r["category"]), r["clip_id"])):
        print(f"  {r['category']:<13}{r['clip_id']:<26}{r['pipeline']:<15}"
              f"{fmt(r['mean_j_and_f'], 100):>6}  fail={r['failure_frame']!s:<5}"
              f"delta={fmt(r['delta_s']):<6}recovered={r['recovered']}"
              + ("  INIT FAILURE" if r["init_failure"] else ""))


def main() -> None:
    """Download data, run both pipelines, and print the report's numbers."""

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reproduce"))
    parser.add_argument("--download-only", action="store_true")
    arguments = parser.parse_args()

    clips = load_clips()
    download_missing(clips)
    if arguments.download_only:
        return
    run_all(clips, arguments.output_dir, load_config(CONFIG))

    results = [
        clip_result(clip, arguments.output_dir / f"{clip['clip_id']}__{condition}", condition)
        for clip in clips for condition in PIPELINES + PERTURBATIONS
    ]
    print_report(results)
    table = arguments.output_dir / "results.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        columns = [key for key in results[0] if not key.endswith("_curve")]
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    plot_figures(results, Path("figures"))
    plot_examples(clips, arguments.output_dir, Path("figures"))
    print(f"\nWrote {table} and figures/examples.pdf, jf_progress.pdf, j_vs_event.pdf, survival.pdf")


if __name__ == "__main__":
    main()
