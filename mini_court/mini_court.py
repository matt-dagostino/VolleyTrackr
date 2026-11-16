from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

import constants
from analysis.court_geometry import CourtGeometry
from utils import convert_meters_to_pixel_distance, get_centre_of_bbox

Point = Tuple[float, float]


class MiniCourt:
    """Top-right schematic court with optional live ball/player projection."""

    def __init__(self, frame: np.ndarray):
        self.drawing_rectangle_width = 140
        self.drawing_rectangle_height = 260
        self.buffer = 40
        self.padding_court = 18

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_keypoints()
        self.set_court_lines()
        self._court_geometry: Optional[CourtGeometry] = None

    def convert_meters_to_pixels(self, meters: float) -> float:
        return convert_meters_to_pixel_distance(
            meters, constants.LINE_WIDTH, self.court_drawing_width
        )

    def set_court_drawing_keypoints(self) -> None:
        # Four outer corners + attack-line endpoints on each half
        drawing_key_points = [0] * 16

        # 0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right
        drawing_key_points[0], drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        drawing_key_points[2], drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = int(
            self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT * 2)
        )
        drawing_key_points[6] = int(self.court_end_x)
        drawing_key_points[7] = drawing_key_points[5]

        # Attack lines (3m from net on each side)
        net_y = (drawing_key_points[1] + drawing_key_points[5]) / 2.0
        attack_offset = self.convert_meters_to_pixels(constants.ATTACK_LINE_DISTANCE)
        drawing_key_points[8] = int(self.court_start_x)
        drawing_key_points[9] = int(net_y - attack_offset)
        drawing_key_points[10] = int(self.court_end_x)
        drawing_key_points[11] = int(net_y - attack_offset)
        drawing_key_points[12] = int(self.court_start_x)
        drawing_key_points[13] = int(net_y + attack_offset)
        drawing_key_points[14] = int(self.court_end_x)
        drawing_key_points[15] = int(net_y + attack_offset)

        self.drawing_key_points = drawing_key_points
        self.mini_court_corners = [
            (drawing_key_points[0], drawing_key_points[1]),
            (drawing_key_points[2], drawing_key_points[3]),
            (drawing_key_points[6], drawing_key_points[7]),
            (drawing_key_points[4], drawing_key_points[5]),
        ]

    def set_court_lines(self) -> None:
        self.lines = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (4, 5),  # near attack line
            (6, 7),  # far attack line
        ]

    def set_mini_court_position(self) -> None:
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def set_canvas_background_box_position(self, frame: np.ndarray) -> None:
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def bind_video_court(self, court_geometry: CourtGeometry) -> None:
        """Wire homography from video court corners onto the mini-court rectangle."""
        self._court_geometry = court_geometry
        # Match click order: assume user clicked TL, TR, BR, BL (or close enough).
        # Use polygon order as provided.
        src_count = len(court_geometry.polygon)
        if src_count != 4:
            return
        court_geometry.set_destination_corners(self.mini_court_corners)

    def draw_court(self, frame: np.ndarray) -> np.ndarray:
        for i in range(0, 8, 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i + 1])
            cv2.circle(frame, (x, y), 3, (40, 40, 40), -1)

        for line in self.lines:
            start_point = (
                int(self.drawing_key_points[line[0] * 2]),
                int(self.drawing_key_points[line[0] * 2 + 1]),
            )
            end_point = (
                int(self.drawing_key_points[line[1] * 2]),
                int(self.drawing_key_points[line[1] * 2 + 1]),
            )
            cv2.line(frame, start_point, end_point, (30, 30, 30), 2)

        net_start = (
            self.drawing_key_points[0],
            int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2),
        )
        net_end = (
            self.drawing_key_points[2],
            int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2),
        )
        cv2.line(frame, net_start, net_end, (0, 90, 255), 2)
        return frame

    def draw_background_rectangle(self, frame: np.ndarray) -> np.ndarray:
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(
            shapes,
            (self.start_x, self.start_y),
            (self.end_x, self.end_y),
            (255, 255, 255),
            cv2.FILLED,
        )
        out = frame.copy()
        alpha = 0.55
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        return out

    def draw_mini_court(
        self,
        frames: Sequence[np.ndarray],
        ball_detections: Optional[Sequence[dict]] = None,
        player_detections: Optional[Sequence[dict]] = None,
    ) -> List[np.ndarray]:
        output_frames = []
        for idx, frame in enumerate(frames):
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)

            if player_detections is not None and idx < len(player_detections):
                self._draw_players(frame, player_detections[idx])
            if ball_detections is not None and idx < len(ball_detections):
                self._draw_ball(frame, ball_detections[idx])

            output_frames.append(frame)
        return output_frames

    def _project_point(self, point: Point) -> Optional[Point]:
        if self._court_geometry is None:
            return None
        return self._court_geometry.project_to_destination(point)

    def _draw_ball(self, frame: np.ndarray, ball_dict: dict) -> None:
        bbox = ball_dict.get(1)
        if not bbox or len(bbox) < 4:
            return
        center = get_centre_of_bbox(bbox)
        projected = self._project_point(center)
        if projected is None:
            return
        cv2.circle(frame, (int(projected[0]), int(projected[1])), 5, (0, 0, 255), -1)

    def _draw_players(self, frame: np.ndarray, player_dict: dict) -> None:
        for bbox in player_dict.values():
            center = get_centre_of_bbox(bbox)
            projected = self._project_point(center)
            if projected is None:
                continue
            cv2.circle(frame, (int(projected[0]), int(projected[1])), 4, (0, 180, 0), -1)

    def get_start_point_of_mini_court(self):
        return (self.court_start_x, self.court_start_y)

    def get_width_of_mini_court(self):
        return self.court_drawing_width

    def get_court_drawing_keypoints(self):
        return self.drawing_key_points
