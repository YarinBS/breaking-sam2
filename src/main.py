"""Command line: run one condition on one clip (``python -m src.main --help``)."""

from __future__ import annotations

import argparse
from pathlib import Path

from .inference import run_pipeline
from .utils import load_config


def main() -> None:
    """Parse the arguments, run the pipeline, and print the comparison video path."""

    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="Segment one object in a video or a directory of numeric frames.",
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="Video file or directory of numeric image frames")
    parser.add_argument("--pipeline", choices=("sam2", "grounded_sam2"), default="grounded_sam2")
    parser.add_argument("--prompt", help="Target description for grounded_sam2")
    parser.add_argument("--box", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"),
                        help="Frame-0 box for sam2")
    parser.add_argument("--output-dir", required=True, type=Path, help="New output directory")
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Frame rate of a directory input (default: 30)")
    parser.add_argument("--annotations", type=Path,
                        help="Directory of numeric ground-truth masks")
    parser.add_argument("--object-id", type=int,
                        help="Object value to isolate in indexed ground-truth masks")
    arguments = parser.parse_args()

    print(
        run_pipeline(
            input_path=arguments.input,
            output_dir=arguments.output_dir,
            config=load_config(arguments.config),
            pipeline=arguments.pipeline,
            prompt=arguments.prompt,
            box=arguments.box,
            frames_per_second=arguments.fps,
            annotations_dir=arguments.annotations,
            object_id=arguments.object_id,
        )
    )


if __name__ == "__main__":
    main()
