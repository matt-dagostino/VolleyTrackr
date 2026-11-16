from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def read_video(
    video_path: str,
    max_frames: Optional[int] = None,
) -> Tuple[List[np.ndarray], float]:
    """
    Load frames from a video file.

    Returns:
        frames: list of BGR frames
        fps: source frames-per-second (falls back to 24.0)
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-3:
        fps = 24.0

    frames: List[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break

    capture.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    return frames, fps


def save_video(
    output_video_frames: List[np.ndarray],
    output_video_path: str,
    fps: float = 24.0,
) -> None:
    if not output_video_frames:
        raise ValueError("Cannot save empty frame list")

    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = output_video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open VideoWriter for {output_video_path}")

    for frame in output_video_frames:
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)

    writer.release()
    print(f"Video saved as '{output_path}' ({len(output_video_frames)} frames @ {fps:.2f} FPS)")
