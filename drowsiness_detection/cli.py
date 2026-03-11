import argparse
from pathlib import Path

from .config import DetectorConfig



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-time drowsiness detection using webcam.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=-1,
        help="Webcam index. Use -1 for auto-select (default: -1)",
    )
    parser.add_argument(
        "--camera-scan-limit",
        type=int,
        default=6,
        help="How many camera indexes to probe when --camera-index=-1",
    )
    parser.add_argument(
        "--predictor-path",
        type=Path,
        default=Path("models/shape_predictor_68_face_landmarks.dat"),
        help="Path to dlib facial landmark predictor .dat file",
    )
    parser.add_argument(
        "--ear-threshold",
        type=float,
        default=0.25,
        help="Eye aspect ratio threshold to treat eyes as closed",
    )
    parser.add_argument(
        "--frame-threshold",
        type=int,
        default=20,
        help="Consecutive closed-eye frames required to trigger alert",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=450,
        help="Resize output frame width (<=0 disables resize)",
    )
    parser.add_argument(
        "--window-title",
        default="Drowsiness Detection",
        help="OpenCV window title",
    )
    parser.add_argument(
        "--hide-eye-contours",
        action="store_true",
        help="Disable eye contour drawing",
    )
    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not 0.0 < args.ear_threshold < 1.0:
        parser.error("--ear-threshold must be between 0 and 1.")
    if args.frame_threshold < 1:
        parser.error("--frame-threshold must be at least 1.")
    if args.camera_index < -1:
        parser.error("--camera-index must be -1 or greater.")
    if args.camera_scan_limit < 1:
        parser.error("--camera-scan-limit must be at least 1.")

    config = DetectorConfig(
        predictor_path=args.predictor_path,
        eye_aspect_ratio_threshold=args.ear_threshold,
        consecutive_frame_threshold=args.frame_threshold,
        camera_index=args.camera_index,
        camera_scan_limit=args.camera_scan_limit,
        frame_width=args.frame_width,
        window_title=args.window_title,
        show_eye_contours=not args.hide_eye_contours,
    )

    from .app import run

    run(config)


if __name__ == "__main__":
    main()
