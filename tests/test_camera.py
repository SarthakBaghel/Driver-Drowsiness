import unittest

from drowsiness_detection.camera import select_camera_index


class CameraSelectionTests(unittest.TestCase):
    def test_auto_mode_prefers_non_zero_when_multiple_cameras_exist(self) -> None:
        selected = select_camera_index(requested_index=-1, available_indices=[0, 1, 2])
        self.assertEqual(selected, 1)

    def test_auto_mode_uses_zero_when_it_is_only_camera(self) -> None:
        selected = select_camera_index(requested_index=-1, available_indices=[0])
        self.assertEqual(selected, 0)

    def test_manual_mode_uses_requested_index(self) -> None:
        selected = select_camera_index(requested_index=3, available_indices=[0, 1])
        self.assertEqual(selected, 3)

    def test_auto_mode_raises_when_no_camera_available(self) -> None:
        with self.assertRaises(RuntimeError):
            select_camera_index(requested_index=-1, available_indices=[])


if __name__ == "__main__":
    unittest.main()
