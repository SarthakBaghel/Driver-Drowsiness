from typing import List

import cv2


def discover_available_camera_indices(max_index: int = 6) -> List[int]:
    available_indices: List[int] = []
    if max_index < 1:
        return available_indices

    for index in range(max_index):
        capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            capture.release()
            continue

        ok, _ = capture.read()
        capture.release()
        if ok:
            available_indices.append(index)

    return available_indices


def select_camera_index(requested_index: int, available_indices: List[int]) -> int:
    if requested_index >= 0:
        return requested_index

    if not available_indices:
        raise RuntimeError(
            "No camera device detected. Connect a webcam and check camera permissions."
        )

    # OBS Virtual Camera is commonly index 0. If more devices exist, prefer a non-zero index.
    if len(available_indices) > 1 and 0 in available_indices:
        for index in available_indices:
            if index != 0:
                return index

    return available_indices[0]


def resolve_camera_index(requested_index: int, scan_limit: int = 6) -> int:
    if requested_index >= 0:
        return requested_index

    available_indices = discover_available_camera_indices(max_index=scan_limit)
    return select_camera_index(requested_index=requested_index, available_indices=available_indices)
