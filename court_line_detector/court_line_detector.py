from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np


class CourtLineDetector:
    """
    Manual 4-corner court selection.

    Supports interactive clicking or loading/saving keypoints JSON so batch
    runs can skip the GUI after the first calibration.
    """

    def predict(
        self,
        image: Union[str, np.ndarray],
        keypoints_path: Optional[str] = None,
        save_keypoints_path: Optional[str] = None,
    ) -> np.ndarray:
        if keypoints_path and Path(keypoints_path).exists():
            keypoints = self.load_keypoints(keypoints_path)
            if save_keypoints_path:
                self.save_keypoints(keypoints, save_keypoints_path)
            return keypoints

        image_bgr = self._load_image(image)
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 7))
        plt.imshow(img_rgb)
        plt.title("Click 4 corners in order: TL → TR → BR → BL")
        plt.axis("on")
        points = plt.ginput(4, timeout=0)
        plt.close()

        if len(points) != 4:
            raise RuntimeError(f"Expected 4 court corners, received {len(points)}")

        keypoints = np.array(points, dtype=np.float32).reshape((-1, 1, 2))

        target = save_keypoints_path or keypoints_path
        if target:
            self.save_keypoints(keypoints, target)

        return keypoints

    @staticmethod
    def _load_image(image: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Court reference image not found: {image}")
        loaded = cv2.imread(str(path))
        if loaded is None:
            raise RuntimeError(f"Failed to read image: {image}")
        return loaded

    @staticmethod
    def load_keypoints(path: Union[str, Path]) -> np.ndarray:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        points = data["keypoints"] if isinstance(data, dict) else data
        return np.asarray(points, dtype=np.float32).reshape((-1, 1, 2))

    @staticmethod
    def save_keypoints(keypoints: np.ndarray, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        flat = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2).tolist()
        path.write_text(json.dumps({"keypoints": flat}, indent=2), encoding="utf-8")

    def draw_keypoints(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        polygon = np.asarray(keypoints, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [polygon], isClosed=True, color=(0, 255, 255), thickness=2)

        for i, point in enumerate(polygon):
            x, y = int(point[0][0]), int(point[0][1])
            cv2.putText(
                image,
                str(i),
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )
            cv2.circle(image, (x, y), 5, (0, 255, 255), -1)
        return image

    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            output_video_frames.append(self.draw_keypoints(frame, keypoints))
        return output_video_frames
