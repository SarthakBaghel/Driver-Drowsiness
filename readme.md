# Drowsiness Detection (Webcam)

Real-time driver drowsiness detection using facial landmarks and Eye Aspect Ratio (EAR).

When you run the app, it opens your webcam automatically, processes frames in real time, and shows an alert when eyes stay closed for a configured number of consecutive frames.

## Project Architecture

```text
Drowsiness_Detection/
├── Drowsiness_Detection.py         # Backward-compatible launcher
├── requirements.txt
├── models/
│   └── shape_predictor_68_face_landmarks.dat
└── drowsiness_detection/
    ├── __init__.py
    ├── __main__.py                 # Enables: python -m drowsiness_detection
    ├── app.py                      # Webcam capture + application loop
    ├── cli.py                      # CLI argument parsing
    ├── config.py                   # Central config dataclass
    ├── detector.py                 # Frame processing + drowsiness state
    ├── eye.py                      # EAR computation
    └── landmarks.py                # Landmark conversion utilities
```

This structure is ready for extension (audio alarms, logging, recording, model swaps, APIs, tests).

## Setup

1. Create and activate virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Use either command:

```bash
python3 Drowsiness_Detection.py
```

or

```bash
python3 -m drowsiness_detection
```

Press `q` to quit.

## CLI Options

```bash
python3 -m drowsiness_detection \
  --camera-index -1 \
  --camera-scan-limit 6 \
  --predictor-path models/shape_predictor_68_face_landmarks.dat \
  --ear-threshold 0.25 \
  --frame-threshold 20 \
  --frame-width 450
```

Additional option:
- `--hide-eye-contours` to disable eye contour drawing.

## Notes

- Ensure the predictor file exists at `models/shape_predictor_68_face_landmarks.dat`.
- By default, `--camera-index -1` auto-selects camera and prefers non-zero indexes when multiple cameras exist (helps avoid OBS virtual camera on many setups).
- If needed, force normal webcam manually, usually:

```bash
python3 Drowsiness_Detection.py --camera-index 1
```

- If webcam does not open, check camera permissions and try another `--camera-index` (for example `2`).

## Development

Run unit tests:

```bash
python3 -m unittest discover -s tests
```
