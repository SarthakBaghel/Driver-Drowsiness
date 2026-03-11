from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DetectorConfig:
    predictor_path: Path = Path("models/shape_predictor_68_face_landmarks.dat")
    eye_aspect_ratio_threshold: float = 0.25
    consecutive_frame_threshold: int = 20
    camera_index: int = -1
    camera_scan_limit: int = 6
    frame_width: int = 450
    window_title: str = "Drowsiness Detection"
    show_eye_contours: bool = True
