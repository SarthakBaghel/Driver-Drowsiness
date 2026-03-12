from collections import deque
from typing import Tuple

import cv2
import dlib
import numpy as np

from .config import DetectorConfig
from .eye import eye_aspect_ratio
from .landmarks import (
    INNER_MOUTH_SLICE,
    LEFT_EYE_SLICE,
    OUTER_MOUTH_SLICE,
    RIGHT_EYE_SLICE,
    shape_to_np,
)
from .mouth import mouth_aspect_ratio


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
        self.yawn_frame_count = 0
        self.mar_history = deque(maxlen=max(1, config.mar_smoothing_window))
        self._clahe = (
            cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            if config.enable_clahe_preprocess
            else None
        )

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        resized = self._resize_frame(frame)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        if self._clahe is not None:
            gray = self._clahe.apply(gray)
        subjects = self.face_detector(gray, self.config.face_detector_upsample)

        drowsy_in_frame = False
        yawn_in_frame = False
        displayed_ear = 0.0
        displayed_mar = 0.0
        max_mar_in_frame = None

        for subject in subjects:
            shape = self.shape_predictor(gray, subject)
            coordinates = shape_to_np(shape)

            left_start, left_end = LEFT_EYE_SLICE
            right_start, right_end = RIGHT_EYE_SLICE
            outer_mouth_start, outer_mouth_end = OUTER_MOUTH_SLICE
            inner_mouth_start, inner_mouth_end = INNER_MOUTH_SLICE
            left_eye = coordinates[left_start:left_end]
            right_eye = coordinates[right_start:right_end]
            outer_mouth = coordinates[outer_mouth_start:outer_mouth_end]
            inner_mouth = coordinates[inner_mouth_start:inner_mouth_end]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            displayed_ear = ear
            mar = mouth_aspect_ratio(inner_mouth)
            if max_mar_in_frame is None or mar > max_mar_in_frame:
                max_mar_in_frame = mar

            if self.config.show_eye_contours:
                cv2.drawContours(resized, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
                cv2.drawContours(resized, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)
            if self.config.show_mouth_contours:
                # Draw actual lip lines (outer + inner), not convex hull, for better visual fidelity.
                cv2.drawContours(resized, [outer_mouth], -1, (0, 255, 255), 1)
                cv2.drawContours(resized, [inner_mouth], -1, (0, 200, 255), 1)
                for point in np.vstack((outer_mouth, inner_mouth)):
                    cv2.circle(resized, tuple(point.astype(int)), 1, (0, 255, 255), -1)

            if ear < self.config.eye_aspect_ratio_threshold:
                drowsy_in_frame = True

        if max_mar_in_frame is not None:
            displayed_mar = self._smooth_mar(max_mar_in_frame)
            if displayed_mar > self.config.mouth_aspect_ratio_threshold:
                yawn_in_frame = True
        else:
            self.mar_history.clear()

        if drowsy_in_frame:
            self.closed_eyes_frame_count += 1
        else:
            self.closed_eyes_frame_count = 0

        if yawn_in_frame:
            self.yawn_frame_count += 1
        else:
            self.yawn_frame_count = 0

        self._draw_overlay(resized, displayed_ear, displayed_mar)
        drowsiness_alert = self.closed_eyes_frame_count >= self.config.consecutive_frame_threshold
        yawning_alert = self.yawn_frame_count >= self.config.yawn_frame_threshold

        if drowsiness_alert:
            cv2.putText(
                resized,
                "ALERT! Drowsiness Detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        if yawning_alert:
            cv2.putText(
                resized,
                "ALERT! Yawning Detected",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
            )

        return resized, drowsiness_alert or yawning_alert

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.config.frame_width <= 0:
            return frame
        height, width = frame.shape[:2]
        ratio = self.config.frame_width / float(width)
        resized_height = int(height * ratio)
        return cv2.resize(frame, (self.config.frame_width, resized_height), interpolation=cv2.INTER_AREA)

    def _draw_overlay(self, frame: np.ndarray, ear: float, mar: float) -> None:
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
            f"MAR (smooth): {mar:.2f}",
            (170, frame.shape[0] - 20),
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
        cv2.putText(
            frame,
            f"Yawn Frame Count: {self.yawn_frame_count}",
            (10, frame.shape[0] - 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    def _smooth_mar(self, mar: float) -> float:
        self.mar_history.append(mar)
        return float(np.mean(self.mar_history))
