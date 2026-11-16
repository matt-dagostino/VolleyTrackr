from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import pickle
import pandas as pd

from constants import DEFAULT_BALL_CONFIDENCE


class BallTracker:
    def __init__(
        self,
        model_path: str,
        conf: float = DEFAULT_BALL_CONFIDENCE,
        load_model: bool = True,
    ):
        self.model_path = model_path
        self.conf = conf
        self.model = None
        if load_model:
            self._ensure_model()

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        if self.model_path.endswith(".pt") and not Path(self.model_path).exists():
            raise FileNotFoundError(f"Ball model not found: {self.model_path}")
        from ultralytics import YOLO  # lazy import — keeps unit tests light

        self.model = YOLO(self.model_path)

    @staticmethod
    def interpolate_ball_positions(ball_positions: Sequence[dict]) -> List[dict]:
        rows = []
        for detection in ball_positions:
            bbox = detection.get(1, []) if isinstance(detection, dict) else []
            if bbox is None or len(bbox) != 4:
                rows.append([None, None, None, None])
            else:
                rows.append(list(bbox))

        if not rows:
            return []

        df = pd.DataFrame(rows, columns=["x1", "y1", "x2", "y2"])
        if df.dropna(how="all").empty:
            return [{1: []} for _ in rows]

        df = df.interpolate(limit_direction="both").bfill().ffill()
        return [{1: row} for row in df.to_numpy().tolist()]

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

        ball_detections = [self.detect_frame(frame) for frame in frames]

        if stub_path is not None:
            Path(stub_path).parent.mkdir(parents=True, exist_ok=True)
            with open(stub_path, "wb") as handle:
                pickle.dump(ball_detections, handle)

        return ball_detections

    def detect_frame(self, frame) -> Dict[int, List[float]]:
        self._ensure_model()
        results = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        best_box = None
        best_conf = -1.0

        for box in results.boxes:
            conf = float(box.conf.tolist()[0]) if box.conf is not None else 0.0
            if conf > best_conf:
                best_conf = conf
                best_box = box.xyxy.tolist()[0]

        if best_box is None:
            return {}
        return {1: best_box}

    def draw_bounding_boxes(self, video_frames: Sequence, ball_detections: Sequence[dict]):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            for track_id, bbox in ball_dict.items():
                if not bbox or len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = bbox
                cv2.putText(
                    frame,
                    f"Ball",
                    (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        return output_video_frames
