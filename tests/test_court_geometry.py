import numpy as np
import pytest

from analysis.court_geometry import CourtGeometry
from analysis.trajectory import BallTrajectoryAnalyzer
from analysis.events import EventDetector, EventType


def _square_court():
    # TL, TR, BR, BL
    return np.array([[10, 10], [110, 10], [110, 210], [10, 210]], dtype=np.float32)


def test_contains_inside_and_outside():
    court = CourtGeometry(_square_court())
    assert court.contains((60, 100)) is True
    assert court.contains((5, 5)) is False
    assert court.in_out_label((60, 100)) == "IN"
    assert court.in_out_label((200, 200)) == "OUT"
    assert court.in_out_label(None) == "UNKNOWN"


def test_homography_round_trip_corners():
    court = CourtGeometry(_square_court())
    dst = [(0, 0), (100, 0), (100, 200), (0, 200)]
    court.set_destination_corners(dst)

    mapped = court.project_to_destination((10, 10))
    assert mapped is not None
    assert mapped[0] == pytest.approx(0, abs=1e-3)
    assert mapped[1] == pytest.approx(0, abs=1e-3)

    center = court.project_to_destination((60, 110))
    assert center is not None
    assert center[0] == pytest.approx(50, abs=1e-2)
    assert center[1] == pytest.approx(100, abs=1e-2)


def test_json_roundtrip(tmp_path):
    court = CourtGeometry(_square_court())
    path = tmp_path / "court.json"
    court.to_json(path)
    loaded = CourtGeometry.from_json(path)
    np.testing.assert_allclose(loaded.polygon, court.polygon)


def test_bounce_and_event_detection():
    # Synthetic ball: falls then rises (bounce) inside court
    detections = []
    y = 20
    vy = 8
    for i in range(20):
        if i == 10:
            vy = -6  # bounce
        y += vy
        detections.append({1: [50, y - 5, 60, y + 5]})

    trajectory = BallTrajectoryAnalyzer(detections, history=10)
    bounces = trajectory.find_bounce_frames(velocity_threshold=2.0)
    assert 10 in bounces or 11 in bounces

    court = CourtGeometry(_square_court())
    detector = EventDetector(trajectory, court, bounce_velocity_threshold=2.0)
    events = detector.detect()
    bounce_events = [e for e in events if e.event_type in (EventType.BOUNCE_IN, EventType.BOUNCE_OUT)]
    assert bounce_events
    assert bounce_events[0].event_type == EventType.BOUNCE_IN


def test_ball_interpolation_empty_safe():
    from trackers.ball_tracker import BallTracker

    result = BallTracker.interpolate_ball_positions([{}, {}, {}])
    assert len(result) == 3
    assert result[0] == {1: []}
