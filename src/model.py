"""The two models of the benchmark: SAM 2 and Grounding DINO.

Each model is one plain function. The optional model arguments let tests inject
fakes (and callers reuse loaded weights); by default the configured model is
loaded, and downloaded on first use.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


def sam2_pipeline(
    frames_dir: Path | str,
    box: Sequence[float],
    config: Mapping[str, Any],
    predictor: Any | None = None,
) -> dict[int, np.ndarray]:
    """Segment one boxed object on frame 0 and track it through the video.

    Args:
        frames_dir: Directory of zero-based, numerically named JPEG frames.
        box: Frame-0 object box ``(x1, y1, x2, y2)`` in pixels. It is clipped
            to the image, because perturbed boxes can leave it.
        config: Loaded benchmark configuration.
        predictor: Optional already-loaded SAM 2 video predictor.

    Returns:
        A dictionary mapping every frame index to a boolean mask.
    """

    import torch

    # Device choice is automatic; bf16 autocast is the SAM 2 default on CUDA.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if predictor is None:
        from sam2.build_sam import build_sam2_video_predictor

        models = config["models"]
        checkpoint = Path(models["sam2_checkpoint"])
        if not checkpoint.is_file():
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(models["sam2_checkpoint_url"], str(checkpoint))
        predictor = build_sam2_video_predictor(
            config_file=models["sam2_config"],
            ckpt_path=str(checkpoint),
            device=device,
            vos_optimized=False,
        )

    state = predictor.init_state(
        video_path=str(frames_dir),
        offload_video_to_cpu=config["runtime"]["offload_video_to_cpu"],
    )
    width, height = state["video_width"], state["video_height"]
    x1, y1, x2, y2 = np.clip(np.asarray(box, dtype=float), 0, [width, height, width, height])
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Box is empty after clipping to the image: {box}")

    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    )
    threshold = config["thresholds"]["mask_logit"]
    predictions: dict[int, np.ndarray] = {}
    with torch.inference_mode(), autocast:
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            box=np.asarray([x1, y1, x2, y2], dtype=np.float32),
        )
        # Propagation also yields frame 0, so every frame comes from here.
        for frame_index, _, logits in predictor.propagate_in_video(state):
            predictions[int(frame_index)] = (logits[0, 0] > threshold).cpu().numpy()
    return predictions


def grounding_dino_pipeline(
    image_path: Path | str,
    prompt: str,
    config: Mapping[str, Any],
    processor: Any | None = None,
    model: Any | None = None,
) -> list[dict[str, Any]]:
    """Detect objects that match a text prompt in one image.

    Args:
        image_path: Image to search (frame 0 of a clip).
        prompt: Text describing the target object.
        config: Loaded benchmark configuration.
        processor: Optional already-loaded Hugging Face processor.
        model: Optional already-loaded Grounding DINO model.

    Returns:
        Score-sorted detections, each with ``box_xyxy``, ``score``, and
        ``label``. Overlapping boxes are removed with NMS. An empty list is a
        detector miss.
    """

    import torch
    from torchvision.ops import nms

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        model_id = config["models"]["grounding_dino_model_id"]
        cache_dir = config["models"]["grounding_dino_cache_dir"]
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, cache_dir=cache_dir
        ).to(device)
        model.eval()

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    # Grounding DINO separates text phrases with periods, so end with one.
    text = prompt.strip().rstrip(".") + "."
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    thresholds = config["thresholds"]
    result = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs.get("input_ids"),
        threshold=thresholds["box"],
        text_threshold=thresholds["text"],
        target_sizes=[(image.height, image.width)],
    )[0]
    boxes = result["boxes"].detach().cpu()
    scores = result["scores"].detach().cpu()
    if len(boxes) == 0:
        return []
    kept = nms(boxes, scores, thresholds["nms"]).tolist()
    labels = result.get("text_labels", result.get("labels"))
    return [
        {
            "box_xyxy": boxes[index].tolist(),
            "score": float(scores[index]),
            "label": str(labels[index]),
        }
        for index in kept
    ]
