"""Court polygon helpers, in/out judgment, and video↔mini-court homography."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

Point = Tuple[float, float]
KeypointsLike = Union[np.ndarray, Sequence]


class CourtGeometry:
    """Spatial model of a volleyball court defined by four corner clicks."""

    def __init__(self, keypoints: KeypointsLike):
        self.polygon = self._normalize_keypoints(keypoints)
        if len(self.polygon) < 3:
            raise ValueError("Court geometry requires at least 3 corner points")
        self._homography_to_dst: Optional[np.ndarray] = None
        self._homography_to_src: Optional[np.ndarray] = None

    @staticmethod
    def _normalize_keypoints(keypoints: KeypointsLike) -> np.ndarray:
        arr = np.asarray(keypoints, dtype=np.float32)
        arr = arr.reshape(-1, 2)
        return arr

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "CourtGeometry":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        points = data["keypoints"] if isinstance(data, dict) else data
        return cls(points)

    def to_json(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"keypoints": self.polygon.tolist()}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def as_opencv_polygon(self) -> np.ndarray:
        return self.polygon.reshape((-1, 1, 2)).astype(np.int32)

    def contains(self, point: Point, measure_dist: bool = False) -> Union[bool, float]:
        """Return True if point is inside/on the court polygon."""
        result = cv2.pointPolygonTest(
            self.polygon.astype(np.float32),
            (float(point[0]), float(point[1])),
            measure_dist,
        )
        if measure_dist:
            return float(result)
        return result >= 0

    def in_out_label(self, point: Optional[Point]) -> str:
        if point is None:
            return "UNKNOWN"
        return "IN" if self.contains(point) else "OUT"

    def set_destination_corners(self, dst_corners: Sequence[Point]) -> None:
        """Build homography mapping video court corners → destination (e.g. mini court)."""
        src = self.polygon.astype(np.float32)
        dst = np.asarray(dst_corners, dtype=np.float32).reshape(-1, 2)
        if len(src) != 4 or len(dst) != 4:
            raise ValueError("Homography requires exactly 4 source and 4 destination points")
        self._homography_to_dst = cv2.getPerspectiveTransform(src, dst)
        self._homography_to_src = cv2.getPerspectiveTransform(dst, src)

    def project_to_destination(self, point: Point) -> Optional[Point]:
        if self._homography_to_dst is None:
            return None
        pts = np.array([[[point[0], point[1]]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pts, self._homography_to_dst)[0][0]
        return float(mapped[0]), float(mapped[1])

    def project_points(self, points: Iterable[Optional[Point]]) -> list[Optional[Point]]:
        return [self.project_to_destination(p) if p is not None else None for p in points]

    def draw(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2,
        label_corners: bool = True,
    ) -> np.ndarray:
        polygon = self.as_opencv_polygon()
        cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=thickness)
        if label_corners:
            for i, (x, y) in enumerate(self.polygon.astype(int)):
                cv2.circle(frame, (x, y), 5, color, -1)
                cv2.putText(
                    frame,
                    str(i),
                    (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        return frame
