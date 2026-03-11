import numpy as np
from scipy.spatial import distance



def eye_aspect_ratio(eye: np.ndarray) -> float:
    """Compute eye aspect ratio (EAR) for one eye with 6 landmark points."""
    vertical_1 = distance.euclidean(eye[1], eye[5])
    vertical_2 = distance.euclidean(eye[2], eye[4])
    horizontal = distance.euclidean(eye[0], eye[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)
