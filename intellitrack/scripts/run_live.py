"""Live tracking demo script.

Runs the full pipeline (camera → detection → tracking → target selection →
prediction → PID/servo) and displays an annotated video window with:
  - All tracked objects shown in green
  - The selected target shown in red with its track_id and class label
  - A rolling FPS counter in the top-left corner

Usage::

    python scripts/run_live.py --config configs/default.yaml \\
                               --mode reactive_pid

Exit with the ``q`` key.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np

# Ensure the src/ package is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intellitrack.capture.camera import Camera, CameraUnavailableError
from intellitrack.control.pid_controller import PIDController
from intellitrack.control.servo_mapper import ServoMapper
from intellitrack.detection.yolo_detector import Detection, YoloDetector
from intellitrack.pipeline.modes import TrackingMode
from intellitrack.prediction.kalman import ConstantVelocityKalman2D
from intellitrack.prediction.no_prediction import NoPrediction
from intellitrack.tracking.byte_tracker_wrapper import ByteTrackerWrapper, TrackedObject
from intellitrack.tracking.target_selector import TargetSelector
from intellitrack.utils.config import get, load_config

logger = logging.getLogger(__name__)

# Drawing colours (BGR)
_COLOR_TRACKED = (0, 200, 0)    # Green — all other tracks
_COLOR_TARGET = (0, 0, 255)     # Red — selected target
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_tracks(
    frame: np.ndarray,
    tracks: List[TrackedObject],
    target: Optional[TrackedObject],
) -> None:
    """Draw bounding boxes and labels for all tracked objects.

    Args:
        frame: BGR frame to annotate in-place.
        tracks: All active tracked objects.
        target: The currently selected target (drawn in red).
    """
    target_id = target.track_id if target is not None else -1

    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t.bbox_xyxy]
        color = _COLOR_TARGET if t.track_id == target_id else _COLOR_TRACKED
        thickness = 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = f"ID:{t.track_id} {t.class_name} {t.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(frame, label, (x1 + 1, y1 - 2), _FONT, 0.5, (255, 255, 255), 1)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Overlay a rolling FPS counter in the top-left corner.

    Args:
        frame: BGR frame to annotate in-place.
        fps: Current frames-per-second value.
    """
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (10, 25), _FONT, 0.8, (0, 255, 255), 2)


def draw_servo_angles(frame: np.ndarray, pan: float, tilt: float, mode: str) -> None:
    """Overlay pan/tilt angles and mode info at the bottom of the frame.

    Args:
        frame: BGR frame to annotate in-place.
        pan: Current pan angle in degrees.
        tilt: Current tilt angle in degrees.
        mode: Active tracking mode string.
    """
    h = frame.shape[0]
    text = f"Mode: {mode}  Pan: {pan:.1f}°  Tilt: {tilt:.1f}°"
    cv2.putText(frame, text, (10, h - 10), _FONT, 0.6, (255, 200, 0), 1)


def main() -> None:
    # Resolve project root so relative defaults work from any CWD
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="IntelliTrack live demo")
    parser.add_argument(
        "--config",
        type=str,
        default=str(_PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=[m.value for m in TrackingMode],
        help="Override prediction.mode from config.",
    )
    args = parser.parse_args()

    # ---- Setup logging ----
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ---- Load config ----
    config = load_config(args.config)
    mode_str = args.mode or get(config, "prediction.mode", "reactive_pid")
    mode = TrackingMode(mode_str)
    logger.info("Starting live demo in mode '%s'.", mode.value)

    # ---- Initialise subsystems ----
    frame_w = get(config, "camera.width", 640)
    frame_h = get(config, "camera.height", 480)
    fps_target = get(config, "camera.fps_target", 30)

    camera = Camera(
        index=get(config, "camera.index", 0),
        width=frame_w,
        height=frame_h,
    )

    detector = YoloDetector(
        model_path=get(config, "detection.model_path", "yolo11n.pt"),
        confidence_threshold=get(config, "detection.confidence_threshold", 0.4),
        iou_threshold=get(config, "detection.iou_threshold", 0.45),
        target_classes=get(config, "detection.target_classes", ["person"]),
    )

    tracker = ByteTrackerWrapper(
        model_path=get(config, "detection.model_path", "yolo11n.pt"),
        confidence_threshold=get(config, "detection.confidence_threshold", 0.4),
        iou_threshold=get(config, "detection.iou_threshold", 0.45),
        target_classes=get(config, "detection.target_classes", ["person"]),
        max_age_frames=get(config, "tracking.max_age_frames", 30),
    )

    selector = TargetSelector(
        strategy=get(config, "tracking.selection_strategy", "closest_to_center"),
        frame_width=frame_w,
        frame_height=frame_h,
        max_lost_frames=get(config, "tracking.max_age_frames", 30),
    )

    kalman = ConstantVelocityKalman2D(dt=1.0 / max(fps_target, 1))
    no_pred = NoPrediction()

    kp = get(config, "control.pid.kp", 0.05)
    ki = get(config, "control.pid.ki", 0.0)
    kd = get(config, "control.pid.kd", 0.01)
    out_lim = get(config, "control.pid.output_limit", 15.0)
    pid_pan = PIDController(kp=kp, ki=ki, kd=kd, output_limit=out_lim)
    pid_tilt = PIDController(kp=kp, ki=ki, kd=kd, output_limit=out_lim)

    servo = ServoMapper(
        pan_min_deg=get(config, "servo.pan_min_deg", 0),
        pan_max_deg=get(config, "servo.pan_max_deg", 180),
        tilt_min_deg=get(config, "servo.tilt_min_deg", 0),
        tilt_max_deg=get(config, "servo.tilt_max_deg", 180),
        pan_center_deg=get(config, "servo.pan_center_deg", 90),
        tilt_center_deg=get(config, "servo.tilt_center_deg", 90),
        deadband_px=get(config, "control.deadband_px", 10),
    )
    frame_center = (frame_w / 2.0, frame_h / 2.0)
    centroid_history: list = []
    seq_len = get(config, "prediction.sequence_length", 15)

    # ---- Start camera ----
    try:
        camera.start()
    except CameraUnavailableError as exc:
        logger.error("Cannot start: %s", exc)
        sys.exit(1)

    # ---- Rolling FPS tracking ----
    prev_time = time.perf_counter()

    logger.info("Live window open — press 'q' to quit.")
    try:
        while True:
            t_frame = time.perf_counter()
            ret, frame = camera.read()
            if not ret or frame is None:
                continue

            # Detection
            detections = detector.detect(frame)

            # Tracking
            tracks = tracker.update(frame, detections)

            # Target selection
            target = selector.select(tracks)

            # Prediction
            predicted: Optional[tuple] = None
            if target is not None:
                raw = target.centroid
                centroid_history.append(raw)
                if len(centroid_history) > seq_len:
                    centroid_history.pop(0)

                if mode == TrackingMode.REACTIVE_PID:
                    predicted = no_pred.predict_next(centroid_history)
                elif mode == TrackingMode.KALMAN:
                    kalman.update(raw)
                    predicted = kalman.predict()
                else:
                    predicted = no_pred.predict_next(centroid_history)

                # PID + servo
                dt = t_frame - prev_time if t_frame - prev_time > 0 else 1.0 / fps_target
                dx = predicted[0] - frame_center[0]
                dy = predicted[1] - frame_center[1]
                pan_d = pid_pan.compute(dx, dt)
                tilt_d = pid_tilt.compute(dy, dt)
                cmd = servo.update(dx, dy, pan_d, tilt_d)
                pan_deg, tilt_deg = cmd.pan_deg, cmd.tilt_deg
            else:
                pan_deg, tilt_deg = servo.current_angles

            prev_time = t_frame

            # FPS
            elapsed = time.perf_counter() - t_frame
            fps = 1.0 / max(elapsed, 1e-6)

            # Draw
            annotated = frame.copy()
            draw_tracks(annotated, tracks, target)
            draw_fps(annotated, fps)
            draw_servo_angles(annotated, pan_deg, tilt_deg, mode.value)

            cv2.imshow("IntelliTrack — Live", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        logger.info("Live demo stopped.")


if __name__ == "__main__":
    main()
