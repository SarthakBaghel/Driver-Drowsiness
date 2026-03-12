import unittest

import numpy as np

from drowsiness_detection.mouth import mouth_aspect_ratio


class MouthAspectRatioTests(unittest.TestCase):
    def test_mouth_aspect_ratio_matches_expected_geometry(self) -> None:
        inner_mouth = np.array(
            [
                [0.0, 0.0],   # p1
                [1.0, 1.0],   # p2
                [2.0, 1.0],   # p3
                [3.0, 0.5],   # p4
                [4.0, 0.0],   # p5
                [3.0, -0.5],  # p6
                [2.0, -1.0],  # p7
                [1.0, -1.0],  # p8
            ]
        )
        self.assertAlmostEqual(mouth_aspect_ratio(inner_mouth), 0.625, places=5)

    def test_mouth_aspect_ratio_handles_zero_horizontal_distance(self) -> None:
        inner_mouth = np.array(
            [
                [1.0, 0.0],   # p1
                [1.0, 1.0],   # p2
                [2.0, 1.0],   # p3
                [3.0, 0.5],   # p4
                [1.0, 0.0],   # p5
                [3.0, -0.5],  # p6
                [2.0, -1.0],  # p7
                [1.0, -1.0],  # p8
            ]
        )
        self.assertEqual(mouth_aspect_ratio(inner_mouth), 0.0)


if __name__ == "__main__":
    unittest.main()
