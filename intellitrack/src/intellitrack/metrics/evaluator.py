"""Evaluate a per-frame metrics JSONL log into summary statistics."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def evaluate_run(
    log_path: str,
    horizon_frames: int = 5,
    frame_center: Optional[tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Compute aggregate metrics from a ``.jsonl`` per-frame log.

    Args:
        log_path: Path to a MetricsLogger JSONL file.
        horizon_frames: Lag used when computing prediction RMSE (LSTM/Transformer).
        frame_center: ``(cx, cy)`` used for tracking stability; defaults to
            ``(320, 240)`` if not inferable.

    Returns:
        Dict with keys: ``tracking_accuracy``, ``target_loss_events``,
        ``mean_latency_ms``, ``p95_latency_ms``, ``mean_fps``,
        ``prediction_rmse_px``, ``tracking_stability``, ``mode``, ``n_frames``.
    """
    path = Path(log_path)
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        logger.warning("evaluate_run: empty log %s", log_path)
        return {
            "tracking_accuracy": 0.0,
            "target_loss_events": 0,
            "mean_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "mean_fps": 0.0,
            "prediction_rmse_px": None,
            "tracking_stability": 0.0,
            "mode": None,
            "n_frames": 0,
        }

    mode = records[0].get("mode")
    n = len(records)
    found = [bool(r.get("target_found")) for r in records]
    tracking_accuracy = sum(found) / n

    loss_events = 0
    for i in range(1, n):
        if found[i - 1] and not found[i]:
            loss_events += 1

    latencies = np.array([float(r.get("latency_ms", 0.0)) for r in records], dtype=np.float64)
    fps_vals = np.array([float(r.get("fps_instant", 0.0)) for r in records], dtype=np.float64)

    mean_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    mean_fps = float(np.mean(fps_vals))

    # Prediction RMSE: predicted at t vs raw at t + horizon (lstm/transformer only)
    prediction_rmse: Optional[float] = None
    if mode in ("lstm", "transformer"):
        sq_err: List[float] = []
        for i in range(n - horizon_frames):
            px = records[i].get("predicted_x")
            py = records[i].get("predicted_y")
            future = records[i + horizon_frames]
            rx = future.get("raw_x")
            ry = future.get("raw_y")
            if None in (px, py, rx, ry):
                continue
            sq_err.append((float(px) - float(rx)) ** 2 + (float(py) - float(ry)) ** 2)
        if sq_err:
            prediction_rmse = float(math.sqrt(sum(sq_err) / len(sq_err)))

    # Tracking stability: std of frame-to-frame pixel error from centre
    if frame_center is None:
        frame_center = (320.0, 240.0)
    cx, cy = frame_center
    errors: List[float] = []
    for r in records:
        if not r.get("target_found"):
            continue
        rx, ry = r.get("raw_x"), r.get("raw_y")
        if rx is None or ry is None:
            continue
        errors.append(math.hypot(float(rx) - cx, float(ry) - cy))
    tracking_stability = float(np.std(errors)) if len(errors) > 1 else 0.0

    return {
        "tracking_accuracy": tracking_accuracy,
        "target_loss_events": loss_events,
        "mean_latency_ms": mean_latency,
        "p95_latency_ms": p95_latency,
        "mean_fps": mean_fps,
        "prediction_rmse_px": prediction_rmse,
        "tracking_stability": tracking_stability,
        "mode": mode,
        "n_frames": n,
    }
