from typing import Tuple

import cv2
import dlib
import numpy as np

from .config import DetectorConfig
from .eye import eye_aspect_ratio
from .landmarks import LEFT_EYE_SLICE, RIGHT_EYE_SLICE, shape_to_np


class DrowsinessDetector:
    """Stateful detector that tracks drowsiness across frames."""

    def __init__(self, config: DetectorConfig):
        self.config = config
        if not config.predictor_path.exists():
            raise FileNotFoundError(
                f"Predictor file not found at '{config.predictor_path}'. "
                "Download/copy shape_predictor_68_face_landmarks.dat into models/."
            )
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(str(config.predictor_path))
        self.closed_eyes_frame_count = 0

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        resized = self._resize_frame(frame)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        subjects = self.face_detector(gray, 0)

        drowsy_in_frame = False
        displayed_ear = 0.0

        for subject in subjects:
            shape = self.shape_predictor(gray, subject)
            coordinates = shape_to_np(shape)

            left_start, left_end = LEFT_EYE_SLICE
            right_start, right_end = RIGHT_EYE_SLICE
            left_eye = coordinates[left_start:left_end]
            right_eye = coordinates[right_start:right_end]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            displayed_ear = ear

            if self.config.show_eye_contours:
                cv2.drawContours(resized, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
                cv2.drawContours(resized, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)

            if ear < self.config.eye_aspect_ratio_threshold:
                drowsy_in_frame = True

        if drowsy_in_frame:
            self.closed_eyes_frame_count += 1
        else:
            self.closed_eyes_frame_count = 0

        self._draw_overlay(resized, displayed_ear)
        alert_triggered = self.closed_eyes_frame_count >= self.config.consecutive_frame_threshold

        if alert_triggered:
            cv2.putText(
                resized,
                "ALERT! Drowsiness Detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        return resized, alert_triggered

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.config.frame_width <= 0:
            return frame
        height, width = frame.shape[:2]
        ratio = self.config.frame_width / float(width)
        resized_height = int(height * ratio)
        return cv2.resize(frame, (self.config.frame_width, resized_height), interpolation=cv2.INTER_AREA)

    def _draw_overlay(self, frame: np.ndarray, ear: float) -> None:
        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Closed Frame Count: {self.closed_eyes_frame_count}",
            (10, frame.shape[0] - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
