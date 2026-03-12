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
        default=3,
        help="How many camera indexes to probe when --camera-index=-1",
    )
    parser.add_argument(
        "--exclude-camera-index",
        type=int,
        action="append",
        default=[],
        help="Camera index to skip in auto-select mode (repeatable)",
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
        "--ear-calibration-frames",
        type=int,
        default=300,
        help="Number of initial frames to calibrate personalized EAR threshold",
    )
    parser.add_argument(
        "--mar-threshold",
        type=float,
        default=0.6,
        help="Mouth aspect ratio threshold to treat mouth as yawning",
    )
    parser.add_argument(
        "--yawn-frame-threshold",
        type=int,
        default=30,
        help="Consecutive high-MAR frames required to trigger yawning alert",
    )
    parser.add_argument(
        "--mar-smoothing-window",
        type=int,
        default=5,
        help="Temporal smoothing window for MAR (in frames)",
    )
    parser.add_argument(
        "--face-upsample",
        type=int,
        default=1,
        help="Dlib face detector upsample level for better landmark accuracy",
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
    parser.add_argument(
        "--hide-mouth-contours",
        action="store_true",
        help="Disable mouth contour drawing",
    )
    parser.add_argument(
        "--disable-clahe",
        action="store_true",
        help="Disable CLAHE contrast enhancement before landmark detection",
    )
    parser.add_argument(
        "--disable-ear-calibration",
        action="store_true",
        help="Disable startup EAR calibration and use static --ear-threshold",
    )
    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not 0.0 < args.ear_threshold < 1.0:
        parser.error("--ear-threshold must be between 0 and 1.")
    if not 0.0 < args.mar_threshold < 2.0:
        parser.error("--mar-threshold must be greater than 0 and less than 2.")
    if args.frame_threshold < 1:
        parser.error("--frame-threshold must be at least 1.")
    if args.ear_calibration_frames < 1:
        parser.error("--ear-calibration-frames must be at least 1.")
    if args.yawn_frame_threshold < 1:
        parser.error("--yawn-frame-threshold must be at least 1.")
    if args.mar_smoothing_window < 1:
        parser.error("--mar-smoothing-window must be at least 1.")
    if args.face_upsample < 0:
        parser.error("--face-upsample must be 0 or greater.")
    if args.camera_index < -1:
        parser.error("--camera-index must be -1 or greater.")
    if args.camera_scan_limit < 1:
        parser.error("--camera-scan-limit must be at least 1.")
    for excluded_index in args.exclude_camera_index:
        if excluded_index < 0:
            parser.error("--exclude-camera-index values must be 0 or greater.")

    config = DetectorConfig(
        predictor_path=args.predictor_path,
        eye_aspect_ratio_threshold=args.ear_threshold,
        consecutive_frame_threshold=args.frame_threshold,
        enable_ear_calibration=not args.disable_ear_calibration,
        ear_calibration_frames=args.ear_calibration_frames,
        mouth_aspect_ratio_threshold=args.mar_threshold,
        yawn_frame_threshold=args.yawn_frame_threshold,
        mar_smoothing_window=args.mar_smoothing_window,
        face_detector_upsample=args.face_upsample,
        enable_clahe_preprocess=not args.disable_clahe,
        camera_index=args.camera_index,
        camera_scan_limit=args.camera_scan_limit,
        excluded_camera_indices=tuple(args.exclude_camera_index),
        frame_width=args.frame_width,
        window_title=args.window_title,
        show_eye_contours=not args.hide_eye_contours,
        show_mouth_contours=not args.hide_mouth_contours,
    )

    from .app import run

    run(config)


if __name__ == "__main__":
    main()
