"""End-to-end tracking pipeline.

Wires together all IntelliTrack subsystems into a single
:class:`TrackingPipeline` object:

    Camera → YoloDetector → ByteTrackerWrapper → TargetSelector
          → Prediction stage → PIDController × 2 → ServoMapper
          → Hardware output (stubbed as a log line until Phase 4)

The pipeline is designed to degrade gracefully: if any hardware is unavailable
the system continues in simulation/mock mode rather than crashing.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from intellitrack.capture.camera import Camera, CameraUnavailableError
from intellitrack.control.pid_controller import PIDController
from intellitrack.control.servo_mapper import ServoMapper
from intellitrack.detection.yolo_detector import YoloDetector
from intellitrack.pipeline.modes import TrackingMode
from intellitrack.prediction.kalman import ConstantVelocityKalman2D
from intellitrack.prediction.no_prediction import NoPrediction
from intellitrack.tracking.byte_tracker_wrapper import ByteTrackerWrapper
from intellitrack.tracking.target_selector import TargetSelector
from intellitrack.utils.config import get

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FrameMetrics dataclass
# ---------------------------------------------------------------------------


@dataclass
class FrameMetrics:
    """Metrics captured for a single pipeline frame.

    Attributes:
        timestamp: Unix epoch time at frame capture.
        mode: Active :class:`TrackingMode`.
        target_found: Whether a target was detected this frame.
        raw_centroid: Raw (x, y) centroid from the tracker (or ``None``).
        predicted_centroid: Centroid after the prediction stage (or ``None``).
        pan_deg: Current pan angle in degrees.
        tilt_deg: Current tilt angle in degrees.
        latency_ms: End-to-end frame processing time in milliseconds.
        fps_instant: Instantaneous FPS estimated from ``latency_ms``.
    """

    timestamp: float
    mode: TrackingMode
    target_found: bool
    raw_centroid: Optional[Tuple[float, float]]
    predicted_centroid: Optional[Tuple[float, float]]
    pan_deg: float
    tilt_deg: float
    latency_ms: float
    fps_instant: float


# ---------------------------------------------------------------------------
# TrackingPipeline
# ---------------------------------------------------------------------------


class TrackingPipeline:
    """Full IntelliTrack pipeline, configurable via a nested config dict.

    All tunable parameters are read from the ``config`` dict (as loaded by
    :func:`~intellitrack.utils.config.load_config`) — no hardcoded values.

    Args:
        config: Parsed YAML configuration dictionary.
        mode_override: If supplied, overrides ``config["prediction"]["mode"]``.
    """

    def __init__(
        self,
        config: dict,
        mode_override: Optional[str] = None,
    ) -> None:
        self._config = config
        mode_str = mode_override or get(config, "prediction.mode", "reactive_pid")
        self._mode = TrackingMode(mode_str)

        # --- Camera ---
        self._camera = Camera(
            index=get(config, "camera.index", 0),
            width=get(config, "camera.width", 640),
            height=get(config, "camera.height", 480),
        )

        # --- Detector ---
        self._detector = YoloDetector(
            model_path=get(config, "detection.model_path", "yolo11n.pt"),
            confidence_threshold=get(config, "detection.confidence_threshold", 0.4),
            iou_threshold=get(config, "detection.iou_threshold", 0.45),
            target_classes=get(config, "detection.target_classes", ["person"]),
        )

        # --- Tracker ---
        self._tracker = ByteTrackerWrapper(
            model_path=get(config, "detection.model_path", "yolo11n.pt"),
            confidence_threshold=get(config, "detection.confidence_threshold", 0.4),
            iou_threshold=get(config, "detection.iou_threshold", 0.45),
            target_classes=get(config, "detection.target_classes", ["person"]),
            max_age_frames=get(config, "tracking.max_age_frames", 30),
        )

        frame_w = get(config, "camera.width", 640)
        frame_h = get(config, "camera.height", 480)

        # --- Target selector ---
        self._selector = TargetSelector(
            strategy=get(config, "tracking.selection_strategy", "closest_to_center"),
            frame_width=frame_w,
            frame_height=frame_h,
            max_lost_frames=get(config, "tracking.max_age_frames", 30),
        )

        # --- Prediction stage ---
        self._kalman = ConstantVelocityKalman2D(
            dt=1.0 / max(get(config, "camera.fps_target", 30), 1),
        )
        self._no_pred = NoPrediction()
        # LSTM / Transformer predictors are wired in Phase 5/6
        self._centroid_history: list = []
        seq_len = get(config, "prediction.sequence_length", 15)
        self._seq_len = seq_len

        # --- PID controllers (one per axis) ---
        kp = get(config, "control.pid.kp", 0.05)
        ki = get(config, "control.pid.ki", 0.0)
        kd = get(config, "control.pid.kd", 0.01)
        out_lim = get(config, "control.pid.output_limit", 15.0)

        self._pid_pan = PIDController(kp=kp, ki=ki, kd=kd, output_limit=out_lim)
        self._pid_tilt = PIDController(kp=kp, ki=ki, kd=kd, output_limit=out_lim)

        # --- Servo mapper ---
        self._servo = ServoMapper(
            pan_min_deg=get(config, "servo.pan_min_deg", 0),
            pan_max_deg=get(config, "servo.pan_max_deg", 180),
            tilt_min_deg=get(config, "servo.tilt_min_deg", 0),
            tilt_max_deg=get(config, "servo.tilt_max_deg", 180),
            pan_center_deg=get(config, "servo.pan_center_deg", 90),
            tilt_center_deg=get(config, "servo.tilt_center_deg", 90),
            deadband_px=get(config, "control.deadband_px", 10),
        )

        self._frame_center = (frame_w / 2.0, frame_h / 2.0)
        self._started = False
        self._last_frame_time: Optional[float] = None

        logger.info("TrackingPipeline initialised in mode '%s'.", self._mode.value)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the camera capture.  Must be called before :meth:`run_once`."""
        try:
            self._camera.start()
            self._started = True
        except CameraUnavailableError as exc:
            logger.error("Pipeline start failed — camera unavailable: %s", exc)
            raise

    def stop(self) -> None:
        """Release the camera and reset PID state."""
        self._camera.release()
        self._pid_pan.reset()
        self._pid_tilt.reset()
        self._kalman.reset()
        self._centroid_history.clear()
        self._started = False

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def run_once(self) -> FrameMetrics:
        """Process one frame through the full pipeline.

        Returns:
            :class:`FrameMetrics` for this frame.
        """
        t_start = time.perf_counter()
        now = time.time()

        # --- Frame grab ---
        ret, frame = self._camera.read()
        if not ret or frame is None:
            logger.warning("Pipeline: no frame available this cycle.")
            pan, tilt = self._servo.current_angles
            return self._empty_metrics(now, pan, tilt, t_start)

        # --- Detection ---
        detections = self._detector.detect(frame)

        # --- Tracking ---
        tracks = self._tracker.update(frame, detections)

        # --- Target selection ---
        target = self._selector.select(tracks)

        if target is None:
            pan, tilt = self._servo.current_angles
            return self._empty_metrics(now, pan, tilt, t_start)

        raw_centroid = target.centroid

        # --- Prediction stage ---
        predicted_centroid = self._predict(raw_centroid)

        # --- PID + servo mapping ---
        dt = self._frame_dt()
        cx, cy = self._frame_center
        dx = predicted_centroid[0] - cx
        dy = predicted_centroid[1] - cy

        pan_delta = self._pid_pan.compute(dx, dt)
        tilt_delta = self._pid_tilt.compute(dy, dt)
        cmd = self._servo.update(dx, dy, pan_delta, tilt_delta)

        logger.debug(
            "[%s] raw=(%.1f,%.1f) pred=(%.1f,%.1f) pan=%.1f° tilt=%.1f°",
            self._mode.value,
            raw_centroid[0], raw_centroid[1],
            predicted_centroid[0], predicted_centroid[1],
            cmd.pan_deg, cmd.tilt_deg,
        )

        # Hardware output stub (Phase 4 wires the real serial bridge here)
        logger.info(
            "SERVO CMD — pan=%.1f° tilt=%.1f° (mode=%s)",
            cmd.pan_deg, cmd.tilt_deg, self._mode.value,
        )

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

        return FrameMetrics(
            timestamp=now,
            mode=self._mode,
            target_found=True,
            raw_centroid=raw_centroid,
            predicted_centroid=predicted_centroid,
            pan_deg=cmd.pan_deg,
            tilt_deg=cmd.tilt_deg,
            latency_ms=latency_ms,
            fps_instant=fps,
        )

    def run_loop(self, max_frames: Optional[int] = None) -> None:
        """Continuously call :meth:`run_once`, optionally bounded by frame count.

        Args:
            max_frames: If ``None``, run until interrupted.  Otherwise stop
                after this many frames have been processed.
        """
        if not self._started:
            self.start()

        frame_count = 0
        logger.info("Pipeline loop starting (max_frames=%s).", max_frames)
        try:
            while True:
                self.run_once()
                frame_count += 1
                if max_frames is not None and frame_count >= max_frames:
                    break
        except KeyboardInterrupt:
            logger.info("Pipeline loop interrupted by user.")
        finally:
            self.stop()

    # ------------------------------------------------------------------
    # Mode management
    # ------------------------------------------------------------------

    def set_mode(self, mode: TrackingMode) -> None:
        """Hot-swap the prediction mode without restarting the pipeline.

        Resets PID and Kalman state when switching modes to avoid transients.

        Args:
            mode: The new :class:`TrackingMode` to activate.
        """
        logger.info("Switching mode: %s → %s", self._mode.value, mode.value)
        self._mode = mode
        self._pid_pan.reset()
        self._pid_tilt.reset()
        self._kalman.reset()
        self._centroid_history.clear()

    @property
    def mode(self) -> TrackingMode:
        """Currently active tracking mode."""
        return self._mode

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict(self, raw_centroid: Tuple[float, float]) -> Tuple[float, float]:
        """Route the centroid through the active prediction stage."""
        # Maintain rolling history for LSTM/Transformer (Phase 5/6)
        self._centroid_history.append(raw_centroid)
        if len(self._centroid_history) > self._seq_len:
            self._centroid_history.pop(0)

        if self._mode == TrackingMode.REACTIVE_PID:
            return self._no_pred.predict_next(self._centroid_history)

        elif self._mode == TrackingMode.KALMAN:
            self._kalman.update(raw_centroid)
            return self._kalman.predict()

        elif self._mode == TrackingMode.LSTM:
            # Phase 5: LSTM predictor wired here; fall back to no-pred for now
            logger.warning("LSTM mode selected but predictor not yet wired (Phase 5). Falling back to NoPrediction.")
            return self._no_pred.predict_next(self._centroid_history)

        elif self._mode == TrackingMode.TRANSFORMER:
            # Phase 6: Transformer predictor wired here; fall back to no-pred for now
            logger.warning("Transformer mode selected but predictor not yet wired (Phase 6). Falling back to NoPrediction.")
            return self._no_pred.predict_next(self._centroid_history)

        return raw_centroid

    def _frame_dt(self) -> float:
        """Compute time delta since last frame in seconds."""
        now = time.perf_counter()
        if self._last_frame_time is None:
            dt = 1.0 / max(get(self._config, "camera.fps_target", 30), 1)
        else:
            dt = now - self._last_frame_time
        self._last_frame_time = now
        return max(dt, 1e-6)

    def _empty_metrics(
        self, now: float, pan: float, tilt: float, t_start: float
    ) -> FrameMetrics:
        """Return a FrameMetrics indicating no target found."""
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        return FrameMetrics(
            timestamp=now,
            mode=self._mode,
            target_found=False,
            raw_centroid=None,
            predicted_centroid=None,
            pan_deg=pan,
            tilt_deg=tilt,
            latency_ms=latency_ms,
            fps_instant=fps,
        )
