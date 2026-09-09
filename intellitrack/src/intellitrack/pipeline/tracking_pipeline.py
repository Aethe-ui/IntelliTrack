"""End-to-end tracking pipeline.

Wires together all IntelliTrack subsystems into a single
:class:`TrackingPipeline` object:

    Camera → YoloDetector → ByteTrackerWrapper → TargetSelector
          → Prediction stage → PIDController × 2 → ServoMapper
          → SerialBridge (real hardware or mock fallback)

The pipeline is designed to degrade gracefully: if any hardware is unavailable
the system continues in simulation/mock mode rather than crashing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

from intellitrack.capture.camera import Camera, CameraUnavailableError
from intellitrack.control.pid_controller import PIDController
from intellitrack.control.serial_bridge import MockSerialBridge, SerialBridge
from intellitrack.control.servo_mapper import ServoMapper
from intellitrack.detection.yolo_detector import YoloDetector
from intellitrack.pipeline.modes import TrackingMode
from intellitrack.prediction.kalman import ConstantVelocityKalman2D
from intellitrack.prediction.no_prediction import NoPrediction
from intellitrack.tracking.byte_tracker_wrapper import ByteTrackerWrapper, TrackedObject
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
        enable_metrics: If ``True``, append per-frame records via MetricsLogger.
        camera_source: Optional override for ``camera.source`` (``live`` / ``file``).
        video_path: Optional override for ``camera.file_path`` when using file source.
    """

    def __init__(
        self,
        config: dict,
        mode_override: Optional[str] = None,
        enable_metrics: bool = False,
        camera_source: Optional[str] = None,
        video_path: Optional[str] = None,
    ) -> None:
        self._config = config
        mode_str = mode_override or get(config, "prediction.mode", "reactive_pid")
        self._mode = TrackingMode(mode_str)

        frame_w = get(config, "camera.width", 640)
        frame_h = get(config, "camera.height", 480)
        source = camera_source or get(config, "camera.source", "live")
        file_path = video_path or get(config, "camera.file_path", None)

        # --- Camera ---
        if source == "file" and file_path:
            self._camera = Camera.from_file(
                path=file_path,
                width=frame_w,
                height=frame_h,
            )
        else:
            self._camera = Camera(
                index=get(config, "camera.index", 0),
                width=frame_w,
                height=frame_h,
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
        self._lstm_predictor = None
        self._transformer_predictor = None
        self._centroid_history: list = []
        self._seq_len = get(config, "prediction.sequence_length", 15)
        self._horizon = get(config, "prediction.horizon_frames", 5)
        self._init_predictors()

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

        # --- Hardware / serial bridge ---
        self._bridge: Union[SerialBridge, MockSerialBridge, None] = None
        hw_enabled = get(config, "hardware.enabled", False)
        if hw_enabled:
            self._bridge = SerialBridge(
                port=get(config, "hardware.serial_port", "/dev/ttyUSB0"),
                baud_rate=get(config, "hardware.baud_rate", 115200),
                mock_if_unavailable=get(config, "hardware.mock_if_unavailable", True),
            )
        else:
            self._bridge = MockSerialBridge()

        # --- Metrics logging (Phase 8) ---
        self._metrics_logger = None
        self._enable_metrics = enable_metrics
        if enable_metrics:
            from intellitrack.metrics.logger import MetricsLogger

            self._metrics_logger = MetricsLogger(
                log_dir=get(config, "logging.log_dir", "data/logs"),
                mode=self._mode.value,
            )

        self._frame_center = (frame_w / 2.0, frame_h / 2.0)
        self._frame_size = (frame_w, frame_h)
        self._started = False
        self._last_frame_time: Optional[float] = None
        self._last_metrics: Optional[FrameMetrics] = None
        self._last_frame: Optional[np.ndarray] = None
        self._last_tracks: List[TrackedObject] = []
        self._last_target: Optional[TrackedObject] = None
        self._eof = False

        logger.info("TrackingPipeline initialised in mode '%s'.", self._mode.value)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the camera and serial bridge. Must be called before :meth:`run_once`."""
        try:
            self._camera.start()
            self._started = True
        except CameraUnavailableError as exc:
            logger.error("Pipeline start failed — camera unavailable: %s", exc)
            raise

        if self._bridge is not None:
            self._bridge.connect()

    def stop(self) -> None:
        """Release camera/serial resources and reset control state."""
        self._camera.release()
        if self._bridge is not None:
            self._bridge.close()
        self._pid_pan.reset()
        self._pid_tilt.reset()
        self._kalman.reset()
        self._centroid_history.clear()
        if self._metrics_logger is not None:
            self._metrics_logger.close()
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

        ret, frame = self._camera.read()
        if not ret or frame is None:
            if getattr(self._camera, "is_file_source", False) and self._camera.exhausted:
                self._eof = True
            logger.warning("Pipeline: no frame available this cycle.")
            pan, tilt = self._servo.current_angles
            metrics = self._empty_metrics(now, pan, tilt, t_start)
            self._store_and_log(metrics, None, [], None)
            return metrics

        detections = self._detector.detect(frame)
        tracks = self._tracker.update(frame, detections)
        target = self._selector.select(tracks)

        if target is None:
            pan, tilt = self._servo.current_angles
            metrics = self._empty_metrics(now, pan, tilt, t_start)
            self._store_and_log(metrics, frame, tracks, None)
            return metrics

        raw_centroid = target.centroid
        predicted_centroid = self._predict(raw_centroid)

        logger.info(
            "[%s] raw=(%.1f,%.1f) predicted=(%.1f,%.1f)",
            self._mode.value,
            raw_centroid[0],
            raw_centroid[1],
            predicted_centroid[0],
            predicted_centroid[1],
        )

        dt = self._frame_dt()
        cx, cy = self._frame_center
        dx = predicted_centroid[0] - cx
        dy = predicted_centroid[1] - cy

        pan_delta = self._pid_pan.compute(dx, dt)
        tilt_delta = self._pid_tilt.compute(dy, dt)
        cmd = self._servo.update(dx, dy, pan_delta, tilt_delta)

        if self._bridge is not None:
            self._bridge.send_angles(cmd.pan_deg, cmd.tilt_deg)
        else:
            logger.info(
                "SERVO CMD — pan=%.1f° tilt=%.1f° (mode=%s)",
                cmd.pan_deg,
                cmd.tilt_deg,
                self._mode.value,
            )

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

        metrics = FrameMetrics(
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
        self._store_and_log(metrics, frame, tracks, target)
        return metrics

    def run_loop(
        self,
        max_frames: Optional[int] = None,
        duration_seconds: Optional[float] = None,
    ) -> None:
        """Continuously call :meth:`run_once`, optionally bounded by frames/time.

        Args:
            max_frames: Stop after this many frames (``None`` = unbounded).
            duration_seconds: Stop after this many seconds (``None`` = unbounded).
        """
        if not self._started:
            self.start()

        frame_count = 0
        t0 = time.perf_counter()
        logger.info(
            "Pipeline loop starting (max_frames=%s, duration_seconds=%s).",
            max_frames,
            duration_seconds,
        )
        try:
            while True:
                self.run_once()
                frame_count += 1
                if self._eof:
                    logger.info("Pipeline loop: video file exhausted.")
                    break
                if max_frames is not None and frame_count >= max_frames:
                    break
                if duration_seconds is not None and (time.perf_counter() - t0) >= duration_seconds:
                    break
        except KeyboardInterrupt:
            logger.info("Pipeline loop interrupted by user.")
        finally:
            self.stop()

    # ------------------------------------------------------------------
    # Mode / config hot-swap (Phase 7)
    # ------------------------------------------------------------------

    def set_mode(self, mode: TrackingMode) -> None:
        """Hot-swap the prediction mode without restarting the pipeline."""
        logger.info("Switching mode: %s → %s", self._mode.value, mode.value)
        self._mode = mode
        self._pid_pan.reset()
        self._pid_tilt.reset()
        self._kalman.reset()
        self._centroid_history.clear()
        if self._metrics_logger is not None:
            self._metrics_logger.set_mode(mode.value)

    def update_pid(self, kp: Optional[float] = None, ki: Optional[float] = None,
                   kd: Optional[float] = None, output_limit: Optional[float] = None) -> None:
        """Update PID gains on both axes at runtime."""
        for pid in (self._pid_pan, self._pid_tilt):
            if kp is not None:
                pid._kp = kp
            if ki is not None:
                pid._ki = ki
            if kd is not None:
                pid._kd = kd
            if output_limit is not None:
                pid._output_limit = output_limit
            pid.reset()

    def set_selection_strategy(self, strategy: str) -> None:
        """Update the target selection strategy at runtime."""
        self._selector = TargetSelector(
            strategy=strategy,
            frame_width=int(self._frame_size[0]),
            frame_height=int(self._frame_size[1]),
            max_lost_frames=get(self._config, "tracking.max_age_frames", 30),
        )
        self._config.setdefault("tracking", {})["selection_strategy"] = strategy

    @property
    def mode(self) -> TrackingMode:
        """Currently active tracking mode."""
        return self._mode

    @property
    def last_metrics(self) -> Optional[FrameMetrics]:
        """Most recent :class:`FrameMetrics` (for the API metrics endpoint)."""
        return self._last_metrics

    @property
    def last_annotated_frame(self) -> Optional[np.ndarray]:
        """Most recent annotated BGR frame for MJPEG streaming."""
        if self._last_frame is None:
            return None
        from intellitrack.utils.visualize import visualize_frame

        pan, tilt = self._servo.current_angles
        fps = self._last_metrics.fps_instant if self._last_metrics else 0.0
        return visualize_frame(
            self._last_frame,
            self._last_tracks,
            self._last_target,
            fps=fps,
            pan_deg=pan,
            tilt_deg=tilt,
            mode=self._mode.value,
            raw_centroid=self._last_metrics.raw_centroid if self._last_metrics else None,
            predicted_centroid=(
                self._last_metrics.predicted_centroid if self._last_metrics else None
            ),
        )

    @property
    def config(self) -> dict:
        """Current effective configuration dict."""
        return self._config

    @property
    def metrics_log_path(self) -> Optional[str]:
        """Path to the active metrics JSONL file, if logging is enabled."""
        if self._metrics_logger is None:
            return None
        return str(self._metrics_logger.path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_predictors(self) -> None:
        """Lazily construct LSTM / Transformer wrappers from config."""
        try:
            from intellitrack.prediction.lstm_predictor import LSTMPredictor

            self._lstm_predictor = LSTMPredictor(
                sequence_length=self._seq_len,
                horizon_frames=self._horizon,
                hidden_size=get(self._config, "prediction.lstm.hidden_size", 64),
                num_layers=get(self._config, "prediction.lstm.num_layers", 2),
                checkpoint_path=get(
                    self._config,
                    "prediction.lstm.checkpoint_path",
                    "data/models/lstm_predictor.pt",
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not initialise LSTMPredictor: %s", exc)
            self._lstm_predictor = None

        try:
            from intellitrack.prediction.transformer_predictor import TransformerPredictor

            self._transformer_predictor = TransformerPredictor(
                sequence_length=self._seq_len,
                horizon_frames=self._horizon,
                d_model=get(self._config, "prediction.transformer.d_model", 64),
                nhead=get(self._config, "prediction.transformer.nhead", 4),
                num_layers=get(self._config, "prediction.transformer.num_layers", 2),
                checkpoint_path=get(
                    self._config,
                    "prediction.transformer.checkpoint_path",
                    "data/models/transformer_predictor.pt",
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not initialise TransformerPredictor: %s", exc)
            self._transformer_predictor = None

    def _predict(self, raw_centroid: Tuple[float, float]) -> Tuple[float, float]:
        """Route the centroid through the active prediction stage."""
        self._centroid_history.append(raw_centroid)
        if len(self._centroid_history) > self._seq_len:
            self._centroid_history.pop(0)

        if self._mode == TrackingMode.REACTIVE_PID:
            return self._no_pred.predict_next(self._centroid_history)

        if self._mode == TrackingMode.KALMAN:
            self._kalman.update(raw_centroid)
            return self._kalman.predict()

        if self._mode == TrackingMode.LSTM:
            if self._lstm_predictor is not None:
                return self._lstm_predictor.predict_next(self._centroid_history)
            return self._no_pred.predict_next(self._centroid_history)

        if self._mode == TrackingMode.TRANSFORMER:
            if self._transformer_predictor is not None:
                return self._transformer_predictor.predict_next(self._centroid_history)
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

    def _store_and_log(
        self,
        metrics: FrameMetrics,
        frame: Optional[np.ndarray],
        tracks: List[TrackedObject],
        target: Optional[TrackedObject],
    ) -> None:
        """Cache last frame state and optionally append a metrics log record."""
        self._last_metrics = metrics
        if frame is not None:
            self._last_frame = frame
        self._last_tracks = tracks
        self._last_target = target
        if self._metrics_logger is not None:
            self._metrics_logger.log_frame(metrics)
