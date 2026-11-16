"""Heuristic rally event detection from ball trajectory + court geometry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from .court_geometry import CourtGeometry
from .trajectory import BallTrajectoryAnalyzer

Point = Tuple[float, float]


class EventType(str, Enum):
    BOUNCE_IN = "BOUNCE_IN"
    BOUNCE_OUT = "BOUNCE_OUT"
    RALLY_START = "RALLY_START"
    HIGH_BALL = "SPIKE_CANDIDATE"
    OUT_OF_VIEW = "OUT_OF_VIEW"


@dataclass
class RallyEvent:
    frame_idx: int
    event_type: EventType
    position: Optional[Point] = None
    detail: str = ""


class EventDetector:
    """
    Lightweight event tagging:
    - Bounce IN / OUT from vertical velocity reversals + court polygon
    - Spike candidates from large downward acceleration
    - Rally start when ball reappears after a gap
    """

    def __init__(
        self,
        trajectory: BallTrajectoryAnalyzer,
        court: CourtGeometry,
        bounce_velocity_threshold: float = 2.5,
        spike_drop_threshold: float = 12.0,
        gap_frames: int = 8,
    ):
        self.trajectory = trajectory
        self.court = court
        self.bounce_velocity_threshold = bounce_velocity_threshold
        self.spike_drop_threshold = spike_drop_threshold
        self.gap_frames = gap_frames
        self.events: List[RallyEvent] = []

    def detect(self) -> List[RallyEvent]:
        self.events = []
        self._detect_visibility_and_rally_starts()
        self._detect_bounces()
        self._detect_spike_candidates()
        self.events.sort(key=lambda e: e.frame_idx)
        return self.events

    def events_at(self, frame_idx: int) -> List[RallyEvent]:
        return [e for e in self.events if e.frame_idx == frame_idx]

    def latest_before(self, frame_idx: int) -> Optional[RallyEvent]:
        prior = [e for e in self.events if e.frame_idx <= frame_idx]
        return prior[-1] if prior else None

    def _detect_visibility_and_rally_starts(self) -> None:
        missing_streak = 0
        for i, center in enumerate(self.trajectory.centers):
            if center is None:
                missing_streak += 1
                if missing_streak == self.gap_frames:
                    self.events.append(
                        RallyEvent(i, EventType.OUT_OF_VIEW, None, "Ball lost")
                    )
                continue

            if missing_streak >= self.gap_frames:
                self.events.append(
                    RallyEvent(i, EventType.RALLY_START, center, "Ball reacquired")
                )
            missing_streak = 0

    def _detect_bounces(self) -> None:
        for frame_idx in self.trajectory.find_bounce_frames(self.bounce_velocity_threshold):
            pos = self.trajectory.get_center(frame_idx)
            if pos is None:
                continue
            label = self.court.in_out_label(pos)
            event_type = EventType.BOUNCE_IN if label == "IN" else EventType.BOUNCE_OUT
            self.events.append(
                RallyEvent(frame_idx, event_type, pos, f"Bounce judged {label}")
            )

    def _detect_spike_candidates(self) -> None:
        velocities = self.trajectory.velocities
        for i in range(2, len(velocities)):
            prev_v = velocities[i - 1]
            curr_v = velocities[i]
            if prev_v is None or curr_v is None:
                continue
            # Large increase in downward speed
            drop = curr_v[1] - prev_v[1]
            if drop >= self.spike_drop_threshold and curr_v[1] > self.spike_drop_threshold:
                pos = self.trajectory.get_center(i)
                self.events.append(
                    RallyEvent(i, EventType.HIGH_BALL, pos, f"Downward accel {drop:.1f}px/f")
                )

    def summary(self) -> dict:
        counts = {event.value: 0 for event in EventType}
        for event in self.events:
            counts[event.event_type.value] += 1
        return counts
