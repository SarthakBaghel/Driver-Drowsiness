from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class DetectorConfig:
    predictor_path: Path = Path("models/shape_predictor_68_face_landmarks.dat")
    eye_aspect_ratio_threshold: float = 0.25
    consecutive_frame_threshold: int = 20
    enable_ear_calibration: bool = True
    ear_calibration_frames: int = 300
    mouth_aspect_ratio_threshold: float = 0.6
    yawn_frame_threshold: int = 30
    mar_smoothing_window: int = 5
    face_detector_upsample: int = 1
    enable_clahe_preprocess: bool = True
    camera_index: int = -1
    camera_scan_limit: int = 3
    excluded_camera_indices: Tuple[int, ...] = ()
    frame_width: int = 450
    window_title: str = "Drowsiness Detection"
    show_eye_contours: bool = True
    show_mouth_contours: bool = True
