# Drowsiness Detection (Webcam)

Real-time driver drowsiness detection using facial landmarks, Eye Aspect Ratio (EAR), and Mouth Aspect Ratio (MAR).

When you run the app, it opens your webcam automatically, processes frames in real time, and shows:
- a drowsiness alert when eyes stay closed for a configured number of consecutive frames.
- a yawning alert when mouth opening (MAR) stays above threshold for a configured number of consecutive frames.

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
    ├── mouth.py                    # MAR computation
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
  --camera-scan-limit 3 \
  --exclude-camera-index 1 \
  --predictor-path models/shape_predictor_68_face_landmarks.dat \
  --ear-threshold 0.25 \
  --frame-threshold 20 \
  --mar-threshold 0.6 \
  --yawn-frame-threshold 30 \
  --frame-width 450
```

Additional option:
- `--hide-eye-contours` to disable eye contour drawing.
- `--hide-mouth-contours` to disable inner-mouth contour drawing.
- `--exclude-camera-index N` (repeatable) to skip virtual/problematic cameras in auto mode.

## Notes

- Ensure the predictor file exists at `models/shape_predictor_68_face_landmarks.dat`.
- By default, `--camera-index -1` auto-selects cameras heuristically and prefers higher indexes when multiple cameras exist (common fix for iPhone/OBS taking lower indexes).
- Use `--exclude-camera-index` to skip OBS or other virtual cameras (for example `--exclude-camera-index 1`).
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
