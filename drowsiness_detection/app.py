import cv2

from .camera import resolve_camera_index
from .config import DetectorConfig
from .detector import DrowsinessDetector



def run(config: DetectorConfig) -> None:
    resolved_camera_index = resolve_camera_index(
        requested_index=config.camera_index,
        scan_limit=config.camera_scan_limit,
        excluded_indices=config.excluded_camera_indices,
    )
    detector = DrowsinessDetector(config)
    capture = cv2.VideoCapture(resolved_camera_index)

    if not capture.isOpened():
        raise RuntimeError(
            f"Unable to open webcam at index {resolved_camera_index}. "
            "Check camera permissions and index."
        )

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Unable to read frame from webcam.")

            processed_frame, _ = detector.process_frame(frame)
            cv2.imshow(config.window_title, processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
