"""Quick smoke test for the trained ball detector."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO ball inference on a video")
    parser.add_argument("--model", default="models/best_100.pt")
    parser.add_argument("--source", default="input_videos/input_video.webm")
    parser.add_argument("--conf", type=float, default=0.2)
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not Path(args.source).exists():
        raise FileNotFoundError(f"Source not found: {args.source}")

    model = YOLO(args.model)
    model.predict(args.source, conf=args.conf, save=True)


if __name__ == "__main__":
    main()
