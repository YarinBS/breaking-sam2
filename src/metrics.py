"""Mask metrics and the per-clip evaluation written to ``metrics/``.

Definitions follow ``docs/METRICS.md``; thresholds come from
``configs/benchmark.yaml``.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .utils import list_frame_paths, read_binary_mask


def intersection_over_union(first: np.ndarray, second: np.ndarray) -> float:
    """Compute the Jaccard index of two boolean masks.

    Args:
        first: Boolean mask.
        second: Boolean mask with the same shape.

    Returns:
        Intersection divided by union. Two empty masks score ``1.0`` (DAVIS
        convention), so a correctly empty prediction is not punished.
    """

    union = np.count_nonzero(first | second)
    return 1.0 if union == 0 else float(np.count_nonzero(first & second) / union)


def boundary_f_measure(annotation: np.ndarray, prediction: np.ndarray, tolerance: float) -> float:
    """Compute DAVIS boundary accuracy ``F``.

    Args:
        annotation: Boolean ground-truth mask.
        prediction: Boolean predicted mask with the same shape.
        tolerance: Matching radius as a fraction of the image diagonal, so the
            score is comparable across resolutions.

    Returns:
        The harmonic mean of boundary precision and recall in ``[0, 1]``.
    """

    radius = max(int(math.ceil(tolerance * np.linalg.norm(annotation.shape))), 1)
    ground_boundary = _mask_boundary(annotation)
    predicted_boundary = _mask_boundary(prediction)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    disk = ((xx * xx + yy * yy) <= radius * radius).astype(np.uint8)
    ground_dilated = cv2.dilate(ground_boundary.astype(np.uint8), disk).astype(bool)
    predicted_dilated = cv2.dilate(predicted_boundary.astype(np.uint8), disk).astype(bool)

    ground_count = np.count_nonzero(ground_boundary)
    predicted_count = np.count_nonzero(predicted_boundary)
    if ground_count == 0 or predicted_count == 0:
        return 1.0 if ground_count == predicted_count else 0.0
    precision = np.count_nonzero(predicted_boundary & ground_dilated) / predicted_count
    recall = np.count_nonzero(ground_boundary & predicted_dilated) / ground_count
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    """Convert a boolean mask to the one-pixel DAVIS boundary map (``seg2bmap``).

    Args:
        mask: Boolean mask.

    Returns:
        A boolean boundary image.
    """

    east = np.zeros_like(mask)
    south = np.zeros_like(mask)
    southeast = np.zeros_like(mask)
    east[:, :-1] = mask[:, 1:]
    south[:-1, :] = mask[1:, :]
    southeast[:-1, :-1] = mask[1:, 1:]
    boundary = (mask ^ east) | (mask ^ south) | (mask ^ southeast)
    boundary[-1, :] = mask[-1, :] ^ east[-1, :]
    boundary[:, -1] = mask[:, -1] ^ south[:, -1]
    boundary[-1, -1] = False
    return boundary


def _first_run(
    observations: Sequence[tuple[int, float]],
    is_hit: Callable[[float], bool],
    consecutive: int,
) -> int | None:
    """Find the first run of ``consecutive`` observations that satisfy a test.

    Failure and recovery are the same rule with opposite tests, so one loop
    serves both.

    Args:
        observations: Ordered ``(frame_index, J)`` pairs of annotated frames.
        is_hit: Test applied to each ``J``.
        consecutive: Required run length.

    Returns:
        The first frame of the first qualifying run, or ``None``.
    """

    run: list[int] = []
    for frame_index, score in observations:
        if is_hit(score):
            run.append(frame_index)
            if len(run) == consecutive:
                return run[0]
        else:
            run.clear()
    return None


def evaluate_predictions(
    annotations_dir: Path | str,
    predictions_dir: Path | str,
    output_dir: Path | str,
    thresholds: Mapping[str, Any],
    object_id: int | None = None,
) -> dict[str, Any]:
    """Evaluate one run against its ground truth and save the results.

    Writes ``frame_metrics.csv`` (continuous per-frame values),
    ``summary_metrics.json`` (clip summary with the failure rule applied), and
    ``j_curve.png`` / ``f_curve.png`` / ``j_and_f_curve.png``.

    Args:
        annotations_dir: Numeric ground-truth masks. They may be sparse; a
            frame without a GT file is skipped, never treated as empty GT.
        predictions_dir: Numeric predicted masks, matched to GT by name.
        output_dir: New ``metrics/`` directory.
        thresholds: The ``thresholds`` mapping of the benchmark config.
        object_id: Value to isolate in indexed ground-truth masks.

    Returns:
        The clip summary that is written to ``summary_metrics.json``.
    """

    annotation_paths = list_frame_paths(annotations_dir)
    if not annotation_paths:
        raise ValueError(f"No annotation masks found in {annotations_dir}")

    # Per-frame metrics, only on annotated frames.
    predictions = Path(predictions_dir)
    rows: list[dict[str, Any]] = []
    for annotation_path in annotation_paths:
        frame_index = int(annotation_path.stem)
        annotation = read_binary_mask(annotation_path, object_id)
        prediction = read_binary_mask(predictions / f"{frame_index:05d}.png")
        previous_path = predictions / f"{frame_index - 1:05d}.png"
        has_previous = frame_index > 0 and previous_path.exists()
        previous = read_binary_mask(previous_path) if has_previous else None
        j_score = intersection_over_union(annotation, prediction)
        f_score = boundary_f_measure(annotation, prediction, thresholds["boundary_tolerance"])
        ground_area = int(np.count_nonzero(annotation))
        predicted_area = int(np.count_nonzero(prediction))
        label_count, _ = cv2.connectedComponents(prediction.astype(np.uint8), connectivity=8)
        rows.append(
            {
                "frame_index": frame_index,
                "j": j_score,
                "f": f_score,
                "j_and_f": (j_score + f_score) / 2,
                "signed_area_error": (
                    None if ground_area == 0 else (predicted_area - ground_area) / ground_area
                ),
                "fragments": max(label_count - 1, 0),
                "temporal_iou": (
                    None if previous is None else intersection_over_union(previous, prediction)
                ),
                "ground_truth_area": ground_area,
                "predicted_area": predicted_area,
                "target_present": ground_area > 0,
            }
        )

    # Clip summary: the pre-registered failure rule on the J observations.
    threshold = thresholds["failure_iou"]
    consecutive = thresholds["failure_consecutive"]
    observations = [(row["frame_index"], row["j"]) for row in rows]
    failure = _first_run(observations, lambda j: j < threshold, consecutive)
    recovery = (
        None
        if failure is None
        else _first_run(
            [(frame, j) for frame, j in observations if frame > failure],
            lambda j: j >= threshold,
            consecutive,
        )
    )
    area_errors = [
        abs(row["signed_area_error"]) for row in rows if row["signed_area_error"] is not None
    ]
    temporal_scores = [row["temporal_iou"] for row in rows if row["temporal_iou"] is not None]
    propagation = rows[1:]
    summary = {
        "frames": len(rows),
        "initialization_j": rows[0]["j"],
        "initialization_f": rows[0]["f"],
        "initialization_j_and_f": rows[0]["j_and_f"],
        "mean_j": float(np.mean([row["j"] for row in rows])),
        "mean_f": float(np.mean([row["f"] for row in rows])),
        "mean_j_and_f": float(np.mean([row["j_and_f"] for row in rows])),
        "worst_j": min(row["j"] for row in rows),
        "mean_absolute_area_error": float(np.mean(area_errors)) if area_errors else None,
        "mean_fragments": float(np.mean([row["fragments"] for row in rows])),
        "mean_temporal_iou": float(np.mean(temporal_scores)) if temporal_scores else None,
        "propagation_mean_j": (
            float(np.mean([row["j"] for row in propagation])) if propagation else None
        ),
        "first_failure_frame": failure,
        "failure_censored": failure is None,
        "recovery_frame": recovery,
        "recovered": recovery is not None,
        "failure_threshold": threshold,
        "failure_consecutive": consecutive,
    }

    # Save the table, the summary, and one curve per continuous score.
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "frame_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for metric in ("j", "f", "j_and_f"):
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot([row["frame_index"] for row in rows], [row[metric] for row in rows], marker="o")
        axis.axhline(threshold, color="gray", linestyle=":", label="failure threshold")
        axis.set(xlabel="frame", ylabel=metric.upper(), ylim=(0, 1))
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
        figure.tight_layout()
        figure.savefig(output / f"{metric}_curve.png", dpi=160)
        plt.close(figure)
    return summary
