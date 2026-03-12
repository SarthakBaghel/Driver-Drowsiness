import importlib
import sys
import types
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import numpy as np

from drowsiness_detection.config import DetectorConfig


def _build_face_coordinates(mar_points: np.ndarray) -> np.ndarray:
    coordinates = np.zeros((68, 2), dtype=float)

    eye_points = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [4.0, 0.0],
            [2.0, -1.0],
            [1.0, -1.0],
        ]
    )
    coordinates[36:42] = eye_points
    coordinates[42:48] = eye_points + np.array([8.0, 0.0])
    coordinates[60:68] = mar_points
    return coordinates


class DetectorYawnTests(unittest.TestCase):
    def setUp(self) -> None:
        fake_dlib = types.ModuleType("dlib")
        fake_dlib.get_frontal_face_detector = lambda: (lambda _gray, _upsample: [object()])
        fake_dlib.shape_predictor = lambda _path: (lambda _gray, _subject: object())
        self.dlib_patch = patch.dict(sys.modules, {"dlib": fake_dlib})
        self.dlib_patch.start()

        import drowsiness_detection.detector as detector_module

        self.detector_module = importlib.reload(detector_module)

        temp_predictor = NamedTemporaryFile(suffix=".dat", delete=False)
        temp_predictor.close()
        self.predictor_path = Path(temp_predictor.name)

    def tearDown(self) -> None:
        self.dlib_patch.stop()
        if self.predictor_path.exists():
            self.predictor_path.unlink()

    def test_yawn_alert_requires_consecutive_high_mar_frames(self) -> None:
        high_mar_mouth = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [3.0, 0.5],
                [4.0, 0.0],
                [3.0, -0.5],
                [2.0, -1.0],
                [1.0, -1.0],
            ]
        )
        low_mar_mouth = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.3],
                [2.0, 0.2],
                [3.0, 0.1],
                [4.0, 0.0],
                [3.0, -0.1],
                [2.0, -0.2],
                [1.0, -0.3],
            ]
        )
        high_mar_coordinates = _build_face_coordinates(high_mar_mouth)
        low_mar_coordinates = _build_face_coordinates(low_mar_mouth)

        config = DetectorConfig(
            predictor_path=self.predictor_path,
            eye_aspect_ratio_threshold=0.25,
            consecutive_frame_threshold=20,
            mouth_aspect_ratio_threshold=0.6,
            yawn_frame_threshold=3,
            frame_width=0,
            show_eye_contours=False,
            show_mouth_contours=False,
        )
        detector = self.detector_module.DrowsinessDetector(config)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        with patch.object(self.detector_module, "shape_to_np", return_value=high_mar_coordinates):
            _, alert_1 = detector.process_frame(frame)
            _, alert_2 = detector.process_frame(frame)
            _, alert_3 = detector.process_frame(frame)

        self.assertFalse(alert_1)
        self.assertFalse(alert_2)
        self.assertTrue(alert_3)
        self.assertEqual(detector.yawn_frame_count, 3)

        with patch.object(self.detector_module, "shape_to_np", return_value=low_mar_coordinates):
            _, alert_reset = detector.process_frame(frame)

        self.assertFalse(alert_reset)
        self.assertEqual(detector.yawn_frame_count, 0)


if __name__ == "__main__":
    unittest.main()
