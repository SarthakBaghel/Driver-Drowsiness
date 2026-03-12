import unittest

from drowsiness_detection.cli import build_parser


class CliArgumentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = build_parser()

    def test_mar_and_yawn_threshold_arguments_are_parsed(self) -> None:
        args = self.parser.parse_args(["--mar-threshold", "0.72", "--yawn-frame-threshold", "40"])
        self.assertAlmostEqual(args.mar_threshold, 0.72, places=5)
        self.assertEqual(args.yawn_frame_threshold, 40)

    def test_mar_smoothing_and_face_upsample_arguments_are_parsed(self) -> None:
        args = self.parser.parse_args(["--mar-smoothing-window", "7", "--face-upsample", "2"])
        self.assertEqual(args.mar_smoothing_window, 7)
        self.assertEqual(args.face_upsample, 2)

    def test_hide_mouth_contours_flag_defaults_to_false(self) -> None:
        args = self.parser.parse_args([])
        self.assertFalse(args.hide_mouth_contours)

    def test_hide_mouth_contours_flag_can_be_enabled(self) -> None:
        args = self.parser.parse_args(["--hide-mouth-contours"])
        self.assertTrue(args.hide_mouth_contours)

    def test_disable_clahe_flag_can_be_enabled(self) -> None:
        args = self.parser.parse_args(["--disable-clahe"])
        self.assertTrue(args.disable_clahe)

    def test_exclude_camera_index_can_be_repeated(self) -> None:
        args = self.parser.parse_args(["--exclude-camera-index", "0", "--exclude-camera-index", "2"])
        self.assertEqual(args.exclude_camera_index, [0, 2])


if __name__ == "__main__":
    unittest.main()
