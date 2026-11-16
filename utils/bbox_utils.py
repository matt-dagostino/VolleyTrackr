from __future__ import annotations

from typing import Sequence, Tuple, Union

Number = Union[int, float]
BBox = Sequence[Number]
Point = Tuple[float, float]


def get_centre_of_bbox(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def get_bottom_centre_of_bbox(bbox: BBox) -> Point:
    """Useful for feet / landing estimates."""
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


def measure_distance(p1: Sequence[Number], p2: Sequence[Number]) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def clamp_bbox(bbox: BBox, width: int, height: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return (
        max(0.0, min(float(x1), width - 1)),
        max(0.0, min(float(y1), height - 1)),
        max(0.0, min(float(x2), width - 1)),
        max(0.0, min(float(y2), height - 1)),
    )
