import unittest

import numpy as np

from drowsiness_detection.eye import eye_aspect_ratio


class EyeAspectRatioTests(unittest.TestCase):
    def test_eye_aspect_ratio_matches_expected_geometry(self) -> None:
        eye = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [4.0, 0.0],
                [2.0, -1.0],
                [1.0, -1.0],
            ]
        )
        self.assertAlmostEqual(eye_aspect_ratio(eye), 0.5, places=5)

    def test_eye_aspect_ratio_handles_zero_horizontal_distance(self) -> None:
        eye = np.array(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [1.0, 0.0],
                [2.0, -1.0],
                [1.0, -1.0],
            ]
        )
        self.assertEqual(eye_aspect_ratio(eye), 0.0)


if __name__ == "__main__":
    unittest.main()
