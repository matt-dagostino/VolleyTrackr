"""VolleyTrackr — volleyball ball/player tracking and in/out analysis pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis import (
    BallTrajectoryAnalyzer,
    CourtGeometry,
    EventDetector,
    draw_analysis_overlays,
)
from constants import (
    BOUNCE_VELOCITY_THRESHOLD,
    DEFAULT_BALL_CONFIDENCE,
    DEFAULT_PLAYER_CONFIDENCE,
    TRAJECTORY_HISTORY,
)
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
from trackers import BallTracker, PlayerTracker
from utils import read_video, save_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track volleyball play, judge IN/OUT, and render analysis overlays."
    )
    parser.add_argument(
        "--input",
        default="input_videos/blockbusters.mkv",
        help="Path to input video",
    )
    parser.add_argument(
        "--output",
        default="output_videos/output_video.avi",
        help="Path for annotated output video",
    )
    parser.add_argument(
        "--ball-model",
        default="models/best_100.pt",
        help="Path to trained ball YOLO weights",
    )
    parser.add_argument(
        "--player-model",
        default="yolov8x",
        help="Ultralytics player model name or path",
    )
    parser.add_argument(
        "--court-keypoints",
        default="input_videos/court_keypoints.json",
        help="JSON file for saved court corners (created on first interactive run)",
    )
    parser.add_argument(
        "--read-stubs",
        action="store_true",
        help="Load detections from tracker_stubs instead of running YOLO",
    )
    parser.add_argument(
        "--write-stubs",
        action="store_true",
        help="Cache detections to tracker_stubs after inference",
    )
    parser.add_argument(
        "--filter-players",
        action="store_true",
        help="Keep only the players nearest the court polygon",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on frames processed (useful for quick tests)",
    )
    parser.add_argument(
        "--ball-conf",
        type=float,
        default=DEFAULT_BALL_CONFIDENCE,
        help="Ball detection confidence threshold",
    )
    parser.add_argument(
        "--player-conf",
        type=float,
        default=DEFAULT_PLAYER_CONFIDENCE,
        help="Player detection confidence threshold",
    )
    parser.add_argument(
        "--events-json",
        default="output_videos/events.json",
        help="Where to write detected rally events",
    )
    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Disable ball trajectory ribbon overlay",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_frames, fps = read_video(args.input, max_frames=args.max_frames)
    print(f"Loaded {len(video_frames)} frames from {args.input} @ {fps:.2f} FPS")

    player_stub = "tracker_stubs/player_detections.pkl"
    ball_stub = "tracker_stubs/ball_detections.pkl"
    stub_path_players = player_stub if (args.read_stubs or args.write_stubs) else None
    stub_path_ball = ball_stub if (args.read_stubs or args.write_stubs) else None

    load_models = not args.read_stubs
    player_tracker = PlayerTracker(
        model_path=args.player_model,
        conf=args.player_conf,
        load_model=load_models,
    )
    ball_tracker = BallTracker(
        model_path=args.ball_model,
        conf=args.ball_conf,
        load_model=load_models,
    )

    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub=args.read_stubs,
        stub_path=stub_path_players,
    )
    ball_detections = ball_tracker.detect_frames(
        video_frames,
        read_from_stub=args.read_stubs,
        stub_path=stub_path_ball,
    )
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Court calibration (interactive once, reusable via JSON)
    court_line_detector = CourtLineDetector()
    court_keypoints = court_line_detector.predict(
        video_frames[0],
        keypoints_path=args.court_keypoints,
        save_keypoints_path=args.court_keypoints,
    )
    court = CourtGeometry(court_keypoints)

    if args.filter_players:
        player_detections = player_tracker.choose_and_filter_players(
            court_keypoints, player_detections
        )

    trajectory = BallTrajectoryAnalyzer(ball_detections, history=TRAJECTORY_HISTORY)
    event_detector = EventDetector(
        trajectory,
        court,
        bounce_velocity_threshold=BOUNCE_VELOCITY_THRESHOLD,
    )
    events = event_detector.detect()
    print(f"Detected {len(events)} rally events → {event_detector.summary()}")

    mini_court = MiniCourt(video_frames[0])
    mini_court.bind_video_court(court)

    output_frames = [frame.copy() for frame in video_frames]
    output_frames = player_tracker.draw_bounding_boxes(output_frames, player_detections)
    output_frames = ball_tracker.draw_bounding_boxes(output_frames, ball_detections)
    output_frames = court_line_detector.draw_keypoints_on_video(output_frames, court_keypoints)
    output_frames = mini_court.draw_mini_court(
        output_frames,
        ball_detections=ball_detections,
        player_detections=player_detections,
    )
    output_frames = draw_analysis_overlays(
        output_frames,
        trajectory,
        court,
        event_detector,
        draw_trajectory=not args.no_trajectory,
    )

    save_video(output_frames, args.output, fps=fps)

    events_path = Path(args.events_json)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "frame": event.frame_idx,
            "type": event.event_type.value,
            "position": event.position,
            "detail": event.detail,
        }
        for event in events
    ]
    events_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Events written to {events_path}")


if __name__ == "__main__":
    main()
