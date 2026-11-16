from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import pickle
import numpy as np

from constants import DEFAULT_PLAYER_CONFIDENCE, MAX_PLAYERS_ON_COURT
from utils import measure_distance, get_centre_of_bbox


class PlayerTracker:
    def __init__(
        self,
        model_path: str = "yolov8x",
        conf: float = DEFAULT_PLAYER_CONFIDENCE,
        max_players: int = MAX_PLAYERS_ON_COURT,
        load_model: bool = True,
    ):
        self.model_path = model_path
        self.conf = conf
        self.max_players = max_players
        self.model = None
        if load_model:
            self._ensure_model()

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        from ultralytics import YOLO  # lazy import — keeps unit tests light

        self.model = YOLO(self.model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections: Sequence[dict]) -> List[dict]:
        if not player_detections:
            return []
        chosen_players = set(self.choose_player(court_keypoints, player_detections[0]))
        filtered = []
        for player_dict in player_detections:
            filtered.append(
                {
                    track_id: bbox
                    for track_id, bbox in player_dict.items()
                    if track_id in chosen_players
                }
            )
        return filtered

    def choose_player(self, court_keypoints, player_dict: dict) -> List[int]:
        if not player_dict:
            return []

        keypoints = np.asarray(court_keypoints, dtype=np.float32).reshape(-1, 2)
        distances = []
        for track_id, bbox in player_dict.items():
            player_centre = get_centre_of_bbox(bbox)
            min_distance = min(
                measure_distance(player_centre, (float(x), float(y)))
                for x, y in keypoints
            )
            distances.append((track_id, min_distance))

        distances.sort(key=lambda item: item[1])
        limit = min(self.max_players, len(distances))
        return [distances[i][0] for i in range(limit)]

    def detect_frames(
        self,
        frames: Sequence,
        read_from_stub: bool = False,
        stub_path: Optional[str] = None,
    ) -> List[dict]:
        if read_from_stub:
            if not stub_path:
                raise ValueError("stub_path is required when read_from_stub=True")
            with open(stub_path, "rb") as handle:
                return pickle.load(handle)

        player_detections = [self.detect_frame(frame) for frame in frames]

        if stub_path is not None:
            Path(stub_path).parent.mkdir(parents=True, exist_ok=True)
            with open(stub_path, "wb") as handle:
                pickle.dump(player_detections, handle)

        return player_detections

    def detect_frame(self, frame) -> Dict[int, List[float]]:
        self._ensure_model()
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            classes=[0],  # COCO person
            verbose=False,
        )[0]
        id_name_dict = results.names
        player_dict: Dict[int, List[float]] = {}

        for box in results.boxes:
            if box.id is None:
                continue
            track_id = int(box.id.tolist()[0])
            cls_id = int(box.cls.tolist()[0])
            cls_name = id_name_dict.get(cls_id, "")
            if cls_name != "person":
                continue
            player_dict[track_id] = box.xyxy.tolist()[0]

        return player_dict

    def draw_bounding_boxes(self, video_frames: Sequence, player_detections: Sequence[dict]):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(
                    frame,
                    f"P{track_id}",
                    (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            output_video_frames.append(frame)
        return output_video_frames
