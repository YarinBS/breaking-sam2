"""The few helpers that more than one module needs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


def load_config(path: Path | str) -> dict[str, Any]:
    """Load the YAML benchmark configuration.

    Local model paths are resolved relative to the configuration file, so the
    same config works from any working directory.

    Args:
        path: YAML configuration file.

    Returns:
        The configuration dictionary with absolute local model paths.
    """

    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    models = config["models"]
    for key in ("sam2_checkpoint", "grounding_dino_cache_dir"):
        models[key] = str((config_path.parent / models[key]).resolve())
    return config


def list_frame_paths(directory: Path | str) -> list[Path]:
    """List image files in numeric frame order.

    Frames and masks are matched by their numeric file name, so a name that is
    not a number is an error rather than something to guess around.

    Args:
        directory: Directory containing frames named with numeric stems.

    Returns:
        Sorted JPEG and PNG paths.
    """

    paths = [
        path
        for path in Path(directory).iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    try:
        return sorted(paths, key=lambda path: int(path.stem))
    except ValueError as error:
        raise ValueError(f"Frame names must be numeric in {directory}") from error


def read_binary_mask(path: Path | str, object_id: int | None = None) -> np.ndarray:
    """Read a binary or indexed mask image as a 2-D boolean array.

    Args:
        path: Mask image path.
        object_id: Value to isolate in an indexed mask (DAVIS, M3-VOS,
            VOST, and VOT store one positive ID per object). Without it, every nonzero
            pixel is foreground.

    Returns:
        A two-dimensional boolean mask.
    """

    with Image.open(path) as image:
        array = np.asarray(image)
    if object_id is not None:
        return array == object_id
    return np.any(array != 0, axis=2) if array.ndim == 3 else array != 0
