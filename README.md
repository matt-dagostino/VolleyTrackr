# VolleyTrackr

Computer-vision pipeline that tracks the volleyball and players, calibrates the court, judges **IN / OUT**, projects positions onto a mini-court, and tags rally events.

## Features

- **Ball detection** with a custom YOLO model (`models/best_100.pt`)
- **Player tracking** via Ultralytics YOLO (`yolov8x` by default)
- **Court calibration** — click 4 corners once; keypoints are saved to JSON for reuse
- **IN / OUT judgment** from the court polygon
- **Trajectory ribbon** with bounce heuristics
- **Event tagging** — bounce IN/OUT, spike candidates, rally reacquire
- **Mini-court overlay** with live ball/player projection (homography)
- **CLI** for models, stubs, frame limits, and output paths

## Quick start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Place a clip under `input_videos/` and ball weights under `models/best_100.pt`, then run:

```bash
python main.py --input input_videos/blockbusters.mkv --output output_videos/output_video.avi
```

On the first run a window asks you to click **4 court corners in order: TL → TR → BR → BL**. They are stored at `input_videos/court_keypoints.json` so later runs can skip the GUI.

### Useful flags

| Flag | Purpose |
|------|---------|
| `--read-stubs` | Reuse cached detections in `tracker_stubs/` |
| `--write-stubs` | Cache YOLO outputs for faster iteration |
| `--filter-players` | Keep players nearest the court |
| `--max-frames 120` | Smoke-test on a short slice |
| `--court-keypoints path.json` | Custom calibration file |
| `--events-json path.json` | Where rally events are written |
| `--no-trajectory` | Hide the ball path ribbon |

Example fast iteration loop:

```bash
python main.py --write-stubs --max-frames 90
python main.py --read-stubs --max-frames 90
```

## Pipeline

1. Decode video (+ FPS)
2. Detect / track players and ball
3. Interpolate missing ball boxes
4. Load or capture court corners
5. Build trajectory + detect bounce / spike / rally events
6. Draw boxes, court, mini-court projections, HUD
7. Write annotated video + `events.json`

## Project layout

```
analysis/               # IN/OUT, trajectory, events, HUD overlays
trackers/               # Ball + player YOLO trackers
court_line_detector/    # Manual court calibration (+ JSON I/O)
mini_court/             # Schematic court + homography projection
utils/                  # Video I/O, bbox helpers, unit conversions
constants/              # Official court dimensions
tests/                  # Geometry / event unit tests
training/               # Dataset + training notebook
```

## Tests

```bash
pytest -q
```

Geometry and event heuristics are covered without requiring GPU weights.

## Notes

- Full-length clips are loaded into memory; use `--max-frames` while developing.
- Player model is COCO-pretrained; ball model is the project-specific checkpoint.
- Training artifacts live under `training/` and are not required for inference.
