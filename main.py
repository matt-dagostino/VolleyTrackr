from utils import (read_video,
                    save_video)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt

import cv2
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def main():
    # Read Video
    input_video_path = 'input_videos/blockbusters.mkv'
    video_frames = read_video(input_video_path)

    # Detect Players/Ball
    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path='models/best_100.pt')

    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=False,
                                                     stub_path="tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=False,
                                                     stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    
    # Detect Court Lines (MANUAL)
    court_line_detector = CourtLineDetector()
    cv2.imwrite('input_videos/input_picture.PNG', video_frames[0])
    court_keypoints = court_line_detector.predict('input_videos/input_picture.PNG')

    # Filter players (12 players)
    #player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # Initialize Minicourt
    mini_court = MiniCourt(video_frames[0])

    # Draw Bounding Boxes
    output_video_frames = player_tracker.draw_bounding_boxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bounding_boxes(output_video_frames, ball_detections)

    # Draw Court Markings
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Draw Mini Court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)

    # Draw Frame Number
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == "__main__":
    main()