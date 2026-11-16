"""HUD overlays for in/out status, events, and rally stats."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .court_geometry import CourtGeometry
from .events import EventDetector, EventType, RallyEvent
from .trajectory import BallTrajectoryAnalyzer

Point = Tuple[float, float]

_EVENT_COLORS = {
    EventType.BOUNCE_IN: (0, 220, 0),
    EventType.BOUNCE_OUT: (0, 0, 255),
    EventType.RALLY_START: (255, 200, 0),
    EventType.HIGH_BALL: (255, 100, 0),
    EventType.OUT_OF_VIEW: (160, 160, 160),
}


def _draw_panel(frame: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.45) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_analysis_overlays(
    frames: Sequence[np.ndarray],
    trajectory: BallTrajectoryAnalyzer,
    court: CourtGeometry,
    event_detector: EventDetector,
    draw_trajectory: bool = True,
) -> List[np.ndarray]:
    output: List[np.ndarray] = []
    summary = event_detector.summary()

    for i, frame in enumerate(frames):
        canvas = frame
        if draw_trajectory:
            canvas = trajectory.draw_on_frame(canvas, i)

        ball_pos = trajectory.get_center(i)
        in_out = court.in_out_label(ball_pos)
        color = (0, 220, 0) if in_out == "IN" else (0, 0, 255) if in_out == "OUT" else (180, 180, 180)

        _draw_panel(canvas, 8, 8, 280, 118, alpha=0.5)
        cv2.putText(
            canvas,
            f"Frame {i}",
            (18, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Ball: {in_out}",
            (18, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            color,
            2,
            cv2.LINE_AA,
        )

        bounce_in = summary.get(EventType.BOUNCE_IN.value, 0)
        bounce_out = summary.get(EventType.BOUNCE_OUT.value, 0)
        cv2.putText(
            canvas,
            f"IN {bounce_in}   OUT {bounce_out}",
            (18, 104),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )

        frame_events = event_detector.events_at(i)
        _draw_event_banners(canvas, frame_events)

        # Persist a short-lived callout for recent bounce judgments
        recent = event_detector.latest_before(i)
        if recent and i - recent.frame_idx <= 18 and recent.event_type in (
            EventType.BOUNCE_IN,
            EventType.BOUNCE_OUT,
        ):
            _draw_callout(canvas, recent)

        output.append(canvas)

    return output


def _draw_event_banners(frame: np.ndarray, events: Sequence[RallyEvent]) -> None:
    y = 150
    for event in events:
        color = _EVENT_COLORS.get(event.event_type, (255, 255, 255))
        text = event.event_type.value.replace("_", " ")
        cv2.putText(
            frame,
            text,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 28


def _draw_callout(frame: np.ndarray, event: RallyEvent) -> None:
    if event.position is None:
        return
    color = _EVENT_COLORS.get(event.event_type, (255, 255, 255))
    x, y = int(event.position[0]), int(event.position[1])
    label = "IN" if event.event_type == EventType.BOUNCE_IN else "OUT"
    cv2.circle(frame, (x, y), 16, color, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        label,
        (x + 20, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )
