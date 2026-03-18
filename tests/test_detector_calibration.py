import importlib
import sys
import types
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import numpy as np

from drowsiness_detection.config import DetectorConfig


class _FakeRect:
    def __init__(self, width: int = 100, height: int = 100) -> None:
        self._width = width
        self._height = height

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height


def _build_eye_points(ear: float, x_offset: float = 0.0) -> np.ndarray:
    vertical_offset = ear * 2.0
    return np.array(
        [
            [0.0 + x_offset, 0.0],
            [1.0 + x_offset, vertical_offset],
            [2.0 + x_offset, vertical_offset],
            [4.0 + x_offset, 0.0],
            [2.0 + x_offset, -vertical_offset],
            [1.0 + x_offset, -vertical_offset],
        ]
    )


def _build_face_coordinates(ear: float) -> np.ndarray:
    coordinates = np.zeros((68, 2), dtype=float)
    coordinates[36:42] = _build_eye_points(ear)
    coordinates[42:48] = _build_eye_points(ear, x_offset=8.0)
    coordinates[48:60] = np.array([[20.0, 20.0]] * 12)
    coordinates[60:68] = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.2],
            [2.0, 0.2],
            [3.0, 0.1],
            [4.0, 0.0],
            [3.0, -0.1],
            [2.0, -0.2],
            [1.0, -0.2],
        ]
    )
    return coordinates


class DetectorCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        fake_dlib = types.ModuleType("dlib")
        fake_dlib.get_frontal_face_detector = lambda: (lambda _gray, _upsample: [_FakeRect()])
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

    def test_calibration_learns_personalized_threshold_from_observed_ears(self) -> None:
        calibration_ears = [0.40, 0.38, 0.20, 0.36]
        expected_threshold = (max(calibration_ears) + min(calibration_ears)) / 2.0

        config = DetectorConfig(
            predictor_path=self.predictor_path,
            eye_aspect_ratio_threshold=0.25,
            consecutive_frame_threshold=2,
            enable_ear_calibration=True,
            ear_calibration_frames=len(calibration_ears),
            mouth_aspect_ratio_threshold=0.6,
            yawn_frame_threshold=30,
            mar_smoothing_window=1,
            face_detector_upsample=0,
            enable_clahe_preprocess=False,
            frame_width=0,
            show_eye_contours=False,
            show_mouth_contours=False,
        )
        detector = self.detector_module.DrowsinessDetector(config)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        for ear in calibration_ears:
            with patch.object(
                self.detector_module,
                "shape_to_np",
                return_value=_build_face_coordinates(ear),
            ):
                _, alert = detector.process_frame(frame)
            self.assertFalse(alert)

        self.assertTrue(detector.is_ear_calibrated)
        self.assertAlmostEqual(detector.active_ear_threshold, expected_threshold, places=5)
        self.assertAlmostEqual(detector.ear_open_value, 0.40, places=5)
        self.assertAlmostEqual(detector.ear_closed_value, 0.20, places=5)

    def test_drowsiness_alert_starts_only_after_calibration_completes(self) -> None:
        config = DetectorConfig(
            predictor_path=self.predictor_path,
            eye_aspect_ratio_threshold=0.25,
            consecutive_frame_threshold=2,
            enable_ear_calibration=True,
            ear_calibration_frames=2,
            mouth_aspect_ratio_threshold=0.6,
            yawn_frame_threshold=30,
            mar_smoothing_window=1,
            face_detector_upsample=0,
            enable_clahe_preprocess=False,
            frame_width=0,
            show_eye_contours=False,
            show_mouth_contours=False,
        )
        detector = self.detector_module.DrowsinessDetector(config)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        with patch.object(self.detector_module, "shape_to_np", return_value=_build_face_coordinates(0.40)):
            _, first_alert = detector.process_frame(frame)
        with patch.object(self.detector_module, "shape_to_np", return_value=_build_face_coordinates(0.20)):
            _, second_alert = detector.process_frame(frame)
        with patch.object(self.detector_module, "shape_to_np", return_value=_build_face_coordinates(0.25)):
            _, third_alert = detector.process_frame(frame)
        with patch.object(self.detector_module, "shape_to_np", return_value=_build_face_coordinates(0.25)):
            _, fourth_alert = detector.process_frame(frame)

        self.assertFalse(first_alert)
        self.assertFalse(second_alert)
        self.assertFalse(third_alert)
        self.assertTrue(fourth_alert)
        self.assertAlmostEqual(detector.active_ear_threshold, 0.30, places=5)

    def test_static_threshold_is_used_when_calibration_is_disabled(self) -> None:
        config = DetectorConfig(
            predictor_path=self.predictor_path,
            eye_aspect_ratio_threshold=0.25,
            consecutive_frame_threshold=2,
            enable_ear_calibration=False,
            ear_calibration_frames=2,
            mouth_aspect_ratio_threshold=0.6,
            yawn_frame_threshold=30,
            mar_smoothing_window=1,
            face_detector_upsample=0,
            enable_clahe_preprocess=False,
            frame_width=0,
            show_eye_contours=False,
            show_mouth_contours=False,
        )
        detector = self.detector_module.DrowsinessDetector(config)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        with patch.object(self.detector_module, "shape_to_np", return_value=_build_face_coordinates(0.20)):
            _, first_alert = detector.process_frame(frame)
            _, second_alert = detector.process_frame(frame)

        self.assertTrue(detector.is_ear_calibrated)
        self.assertEqual(detector.ear_calibration_frame_count, 0)
        self.assertAlmostEqual(detector.active_ear_threshold, 0.25, places=5)
        self.assertFalse(first_alert)
        self.assertTrue(second_alert)


if __name__ == "__main__":
    unittest.main()
