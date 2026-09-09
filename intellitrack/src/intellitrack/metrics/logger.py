"""Structured per-frame metrics logging."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from intellitrack.pipeline.tracking_pipeline import FrameMetrics

logger = logging.getLogger(__name__)

_FLUSH_EVERY = 30


class MetricsLogger:
    """Append one JSON-lines record per frame to ``data/logs/<mode>_<ts>.jsonl``.

    Writes are buffered and flushed every ``_FLUSH_EVERY`` records or on
    :meth:`close`.

    Args:
        log_dir: Directory for log files.
        mode: Tracking mode string used in the filename and each record.
    """

    def __init__(self, log_dir: str, mode: str) -> None:
        self._mode = mode
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._path = self._dir / f"{mode}_{ts}.jsonl"
        self._file = self._path.open("a", encoding="utf-8")
        self._buffer: list[str] = []
        self._count = 0
        logger.info("MetricsLogger writing to %s", self._path)

    @property
    def path(self) -> Path:
        """Path to the active JSONL log file."""
        return self._path

    def set_mode(self, mode: str) -> None:
        """Update the mode field written into subsequent records."""
        self._mode = mode

    def log_frame(self, metrics: "FrameMetrics") -> None:
        """Append one structured record for a processed frame.

        Args:
            metrics: :class:`~intellitrack.pipeline.tracking_pipeline.FrameMetrics`
                from ``TrackingPipeline.run_once``.
        """
        raw_x = raw_y = pred_x = pred_y = None
        if metrics.raw_centroid is not None:
            raw_x, raw_y = metrics.raw_centroid
        if metrics.predicted_centroid is not None:
            pred_x, pred_y = metrics.predicted_centroid

        record = {
            "timestamp": metrics.timestamp,
            "mode": metrics.mode.value if hasattr(metrics.mode, "value") else str(metrics.mode),
            "target_found": metrics.target_found,
            "raw_x": raw_x,
            "raw_y": raw_y,
            "predicted_x": pred_x,
            "predicted_y": pred_y,
            "pan_deg": metrics.pan_deg,
            "tilt_deg": metrics.tilt_deg,
            "latency_ms": metrics.latency_ms,
            "fps_instant": metrics.fps_instant,
        }
        self._buffer.append(json.dumps(record))
        self._count += 1
        if self._count % _FLUSH_EVERY == 0:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        self._file.write("\n".join(self._buffer) + "\n")
        self._file.flush()
        self._buffer.clear()

    def close(self) -> None:
        """Flush remaining records and close the file handle."""
        self._flush()
        self._file.close()
        logger.info("MetricsLogger closed (%s).", self._path)
