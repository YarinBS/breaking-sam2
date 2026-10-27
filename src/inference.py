"""Run one benchmark condition on one clip and save everything it produces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .metrics import evaluate_predictions
from .model import grounding_dino_pipeline, sam2_pipeline
from .utils import list_frame_paths, read_binary_mask


def run_pipeline(
    input_path: Path | str,
    output_dir: Path | str,
    config: Mapping[str, Any],
    pipeline: str,
    prompt: str | None = None,
    box: Sequence[float] | None = None,
    frames_per_second: float = 30.0,
    annotations_dir: Path | str | None = None,
    object_id: int | None = None,
) -> Path:
    """Segment a clip with one condition and save frames, masks, and results.

    The output directory receives ``frames/``, ``predictions/``, ``run.json``,
    ``comparison_<input name>.mp4``, and, for annotated runs, ``metrics/``.

    Args:
        input_path: Directory of numeric image frames, or a video file.
        output_dir: New directory; an existing one is refused so that runs are
            never mixed.
        config: Loaded benchmark configuration.
        pipeline: ``sam2`` (verified box) or ``grounded_sam2`` (text prompt →
            Grounding DINO box).
        prompt: Target description, required by ``grounded_sam2``.
        box: Frame-0 ``xyxy`` box, required by ``sam2``.
        frames_per_second: Frame rate of a frame-directory input. Video files
            use their own frame rate.
        annotations_dir: Optional directory of numeric ground-truth masks.
        object_id: Value to isolate in indexed ground-truth masks.

    Returns:
        The path of the comparison video.
    """

    if pipeline not in ("sam2", "grounded_sam2"):
        raise ValueError("pipeline must be 'sam2' or 'grounded_sam2'")
    if pipeline == "sam2" and box is None:
        raise ValueError("box is required for the sam2 pipeline")
    if pipeline == "grounded_sam2" and not prompt:
        raise ValueError("prompt is required for the grounded_sam2 pipeline")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)

    # SAM 2 reads only folders of zero-based JPEGs, so every input is rewritten
    # as frames/00000.jpg, 00001.jpg, ...
    source = Path(input_path)
    frames_dir = output / "frames"
    frames_dir.mkdir()
    if source.is_dir():
        fps = float(frames_per_second)
        for index, path in enumerate(list_frame_paths(source)):
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            cv2.imwrite(str(frames_dir / f"{index:05d}.jpg"), frame)
    else:
        capture = cv2.VideoCapture(str(source))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            cv2.imwrite(str(frames_dir / f"{index:05d}.jpg"), frame)
            index += 1
        capture.release()
    frame_paths = list_frame_paths(frames_dir)
    if not frame_paths:
        raise ValueError(f"No frames could be read from {source}")

    # Initialization. A Grounding DINO miss is an observed initialization
    # failure, kept apart from SAM 2 propagation: it gets explicit empty masks.
    detections: list[dict[str, Any]] = []
    if pipeline == "grounded_sam2":
        detections = grounding_dino_pipeline(frame_paths[0], prompt, config)
        box = detections[0]["box_xyxy"] if detections else None
    if box is None:
        height, width = cv2.imread(str(frame_paths[0])).shape[:2]
        empty = np.zeros((height, width), dtype=bool)
        predictions = {index: empty for index in range(len(frame_paths))}
    else:
        predictions = sam2_pipeline(frames_dir, box, config)

    predictions_dir = output / "predictions"
    predictions_dir.mkdir()
    for frame_index, mask in predictions.items():
        cv2.imwrite(str(predictions_dir / f"{frame_index:05d}.png"), mask.astype(np.uint8) * 255)
    if annotations_dir is not None:
        evaluate_predictions(
            annotations_dir, predictions_dir, output / "metrics", config["thresholds"], object_id
        )

    run = {
        "input": str(source.resolve()),
        "input_type": "frames" if source.is_dir() else "video",
        "pipeline": pipeline,
        "prompt": prompt,
        "box_xyxy": None if box is None else list(box),
        "detections": detections,
        "fps": fps,
        "frame_count": len(predictions),
        "annotations": None if annotations_dir is None else str(Path(annotations_dir).resolve()),
        "object_id": object_id,
        "config": dict(config),
    }
    (output / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return write_comparison_video(
        frame_paths,
        predictions,
        output / f"comparison_{source.stem}.mp4",
        fps,
        annotations_dir,
        object_id,
    )


def write_comparison_video(
    frame_paths: Sequence[Path],
    predictions: Mapping[int, np.ndarray],
    output_path: Path,
    fps: float,
    annotations_dir: Path | str | None = None,
    object_id: int | None = None,
) -> Path:
    """Render the frame, the ground truth (if any), and the prediction side by side.

    Args:
        frame_paths: Numeric source frames in order.
        predictions: Boolean mask for every frame index.
        output_path: Destination MP4 file.
        fps: Output frame rate.
        annotations_dir: Optional numeric ground-truth masks. When given, the
            video has three panels instead of two; frames without GT say so.
        object_id: Value to isolate in indexed ground-truth masks.

    Returns:
        The written video path.
    """

    annotations = (
        {int(path.stem): path for path in list_frame_paths(annotations_dir)}
        if annotations_dir is not None
        else {}
    )
    height, width = cv2.imread(str(frame_paths[0])).shape[:2]
    panel_count = 3 if annotations_dir is not None else 2
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * panel_count, height)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not initialize the video writer")
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            frame_index = int(frame_path.stem)
            panels = [_render_panel(frame, "Original")]
            if annotations_dir is not None:
                annotation_path = annotations.get(frame_index)
                if annotation_path is None:
                    panels.append(_render_panel(frame, "Ground truth unavailable"))
                else:
                    annotation = read_binary_mask(annotation_path, object_id)
                    panels.append(_render_panel(frame, "Ground truth", annotation, (80, 220, 80)))
            prediction = predictions[frame_index]
            panels.append(_render_panel(frame, "Prediction", prediction, (0, 180, 255)))
            writer.write(np.concatenate(panels, axis=1))
    finally:
        writer.release()
    return output_path


def _render_panel(
    frame: np.ndarray,
    label: str,
    mask: np.ndarray | None = None,
    color: tuple[int, int, int] = (0, 180, 255),
) -> np.ndarray:
    """Create one labeled video panel with an optional tinted mask.

    Args:
        frame: Source BGR image.
        label: Short panel heading.
        mask: Optional boolean mask to tint.
        color: BGR tint color.

    Returns:
        A copy of the frame with the overlay and heading.
    """

    panel = frame.copy()
    if mask is not None:
        blended = cv2.addWeighted(frame, 0.55, np.full_like(frame, color), 0.45, 0)
        panel[mask] = blended[mask]
    cv2.rectangle(panel, (0, 0), (min(panel.shape[1], 240), 30), (0, 0, 0), -1)
    cv2.putText(
        panel, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )
    return panel
