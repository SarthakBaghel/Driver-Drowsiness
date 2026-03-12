from typing import Tuple

import numpy as np

LEFT_EYE_SLICE: Tuple[int, int] = (42, 48)
RIGHT_EYE_SLICE: Tuple[int, int] = (36, 42)
OUTER_MOUTH_SLICE: Tuple[int, int] = (48, 60)
INNER_MOUTH_SLICE: Tuple[int, int] = (60, 68)


def shape_to_np(shape: object, dtype: str = "int") -> np.ndarray:
    """Convert dlib shape object to a NumPy array of (x, y) coordinates."""
    coordinates = np.zeros((68, 2), dtype=dtype)
    for index in range(68):
        coordinates[index] = (shape.part(index).x, shape.part(index).y)
    return coordinates
