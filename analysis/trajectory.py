"""Ball trajectory extraction, smoothing, and landing heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils.bbox_utils import get_centre_of_bbox, get_bottom_centre_of_bbox

Point = Tuple[float, float]


@dataclass
class TrajectoryPoint:
    frame_idx: int
    center: Optional[Point]
    bottom: Optional[Point]
    interpolated: bool = False


class BallTrajectoryAnalyzer:
    """Builds a screen-space trajectory from per-frame ball detections."""

    def __init__(self, ball_detections: Sequence[dict], history: int = 30):
        self.history = history
        self.points: List[TrajectoryPoint] = self._build_points(ball_detections)
        self.centers: List[Optional[Point]] = [p.center for p in self.points]
        self.velocities = self._compute_velocities(self.centers)

    @staticmethod
    def _build_points(ball_detections: Sequence[dict]) -> List[TrajectoryPoint]:
        points: List[TrajectoryPoint] = []
        for idx, detection in enumerate(ball_detections):
            bbox = detection.get(1) if isinstance(detection, dict) else None
            if bbox is None or len(bbox) < 4 or any(v is None or (isinstance(v, float) and np.isnan(v)) for v in bbox):
                points.append(TrajectoryPoint(idx, None, None, interpolated=False))
                continue
            center = get_centre_of_bbox(bbox)
            bottom = get_bottom_centre_of_bbox(bbox)
            # Interpolation fills gaps with float coords; treat non-original as interpolated
            # Caller can pass a parallel flag; here we infer from presence of floats mid-gap later.
            points.append(
                TrajectoryPoint(
                    frame_idx=idx,
                    center=(float(center[0]), float(center[1])),
                    bottom=(float(bottom[0]), float(bottom[1])),
                    interpolated=False,
                )
            )
        return points

    @staticmethod
    def _compute_velocities(centers: Sequence[Optional[Point]]) -> List[Optional[Point]]:
        velocities: List[Optional[Point]] = [None] * len(centers)
        for i in range(1, len(centers)):
            if centers[i] is None or centers[i - 1] is None:
                continue
            velocities[i] = (
                centers[i][0] - centers[i - 1][0],
                centers[i][1] - centers[i - 1][1],
            )
        return velocities

    def get_center(self, frame_idx: int) -> Optional[Point]:
        if 0 <= frame_idx < len(self.centers):
            return self.centers[frame_idx]
        return None

    def get_path_window(self, frame_idx: int) -> List[Point]:
        start = max(0, frame_idx - self.history + 1)
        path = [c for c in self.centers[start : frame_idx + 1] if c is not None]
        return path

    def find_bounce_frames(self, velocity_threshold: float = 2.5) -> List[int]:
        """
        Detect bounce-like frames where vertical velocity flips from downward (+)
        to upward (-) in image coordinates (y increases downward).
        """
        bounces: List[int] = []
        for i in range(2, len(self.velocities)):
            prev_v = self.velocities[i - 1]
            curr_v = self.velocities[i]
            if prev_v is None or curr_v is None:
                continue
            if prev_v[1] > velocity_threshold and curr_v[1] < -velocity_threshold * 0.5:
                bounces.append(i)
        return bounces

    def draw_on_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        color: Tuple[int, int, int] = (0, 140, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        path = self.get_path_window(frame_idx)
        if len(path) < 2:
            return frame

        pts = np.array(path, dtype=np.int32).reshape((-1, 1, 2))
        overlay = frame.copy()
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            segment_color = tuple(int(c * alpha) for c in color)
            cv2.line(
                overlay,
                tuple(pts[i - 1][0]),
                tuple(pts[i][0]),
                segment_color,
                thickness,
                cv2.LINE_AA,
            )
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        tip = path[-1]
        cv2.circle(frame, (int(tip[0]), int(tip[1])), 4, color, -1, cv2.LINE_AA)
        return frame
