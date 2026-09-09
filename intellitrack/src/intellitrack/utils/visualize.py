"""Shared frame annotation utilities for live demo and API MJPEG stream."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from intellitrack.tracking.byte_tracker_wrapper import TrackedObject

# Drawing colours (BGR)
_COLOR_TRACKED = (0, 200, 0)
_COLOR_TARGET = (0, 0, 255)
_COLOR_RAW = (255, 128, 0)
_COLOR_PRED = (255, 0, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def visualize_frame(
    frame: np.ndarray,
    tracks: List[TrackedObject],
    target: Optional[TrackedObject] = None,
    fps: float = 0.0,
    pan_deg: float = 90.0,
    tilt_deg: float = 90.0,
    mode: str = "",
    raw_centroid: Optional[Tuple[float, float]] = None,
    predicted_centroid: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Return an annotated copy of ``frame`` for display / streaming.

    Args:
        frame: Source BGR image.
        tracks: Active tracked objects.
        target: Currently selected target (drawn in red).
        fps: Instantaneous FPS to overlay.
        pan_deg: Current pan command.
        tilt_deg: Current tilt command.
        mode: Active tracking mode string.
        raw_centroid: Optional raw target centroid to mark.
        predicted_centroid: Optional predicted centroid to mark.

    Returns:
        Annotated BGR image (copy of ``frame``).
    """
    out = frame.copy()
    target_id = target.track_id if target is not None else -1

    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t.bbox_xyxy]
        color = _COLOR_TARGET if t.track_id == target_id else _COLOR_TRACKED
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{t.track_id} {t.class_name} {t.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, label, (x1 + 1, y1 - 2), _FONT, 0.5, (255, 255, 255), 1)

    if raw_centroid is not None:
        cv2.circle(out, (int(raw_centroid[0]), int(raw_centroid[1])), 5, _COLOR_RAW, -1)
    if predicted_centroid is not None:
        cv2.circle(
            out,
            (int(predicted_centroid[0]), int(predicted_centroid[1])),
            5,
            _COLOR_PRED,
            -1,
        )

    cv2.putText(out, f"FPS: {fps:.1f}", (10, 25), _FONT, 0.8, (0, 255, 255), 2)
    h = out.shape[0]
    footer = f"Mode: {mode}  Pan: {pan_deg:.1f}°  Tilt: {tilt_deg:.1f}°"
    cv2.putText(out, footer, (10, h - 10), _FONT, 0.6, (255, 200, 0), 1)
    return out
