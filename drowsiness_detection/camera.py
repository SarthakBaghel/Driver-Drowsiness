import platform
import time
from typing import Iterable, List

import cv2


def discover_available_camera_indices(max_index: int = 6) -> List[int]:
    available_indices: List[int] = []
    if max_index < 1:
        return available_indices

    for index in range(max_index):
        if platform.system() == "Darwin":
            capture = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        else:
            capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            capture.release()
            continue

        ok = False
        # Some physical webcams need a brief warm-up and may fail the first read.
        for _ in range(10):
            ok, frame = capture.read()
            if ok and frame is not None and frame.size > 0:
                break
            time.sleep(0.03)
        capture.release()
        if ok:
            available_indices.append(index)

    return available_indices


def _ordered_auto_candidates(available_indices: List[int]) -> List[int]:
    ordered = sorted(set(available_indices))
    if len(ordered) >= 2:
        # On many macOS setups with Continuity/virtual cameras, built-in webcams end up
        # at higher indexes than iPhone/OBS devices.
        return list(reversed(ordered))
    return ordered


def select_camera_index(
    requested_index: int,
    available_indices: List[int],
    excluded_indices: Iterable[int] = (),
) -> int:
    if requested_index >= 0:
        return requested_index

    if not available_indices:
        raise RuntimeError(
            "No camera device detected. Connect a webcam and check camera permissions."
        )

    excluded = set(excluded_indices)
    ordered_candidates = _ordered_auto_candidates(available_indices)
    filtered_indices = [index for index in ordered_candidates if index not in excluded]
    if filtered_indices:
        return filtered_indices[0]

    return ordered_candidates[0]


def resolve_camera_index(
    requested_index: int,
    scan_limit: int = 6,
    excluded_indices: Iterable[int] = (),
) -> int:
    if requested_index >= 0:
        return requested_index

    available_indices = discover_available_camera_indices(max_index=scan_limit)
    return select_camera_index(
        requested_index=requested_index,
        available_indices=available_indices,
        excluded_indices=excluded_indices,
    )
