import numpy as np
from scipy.spatial import distance


def mouth_aspect_ratio(mouth: np.ndarray) -> float:
    """Compute Mouth Aspect Ratio (MAR) for inner-lip landmarks (8 points)."""
    vertical_1 = distance.euclidean(mouth[1], mouth[7])
    vertical_2 = distance.euclidean(mouth[2], mouth[6])
    vertical_3 = distance.euclidean(mouth[3], mouth[5])
    horizontal = distance.euclidean(mouth[0], mouth[4])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2 + vertical_3) / (2.0 * horizontal)
