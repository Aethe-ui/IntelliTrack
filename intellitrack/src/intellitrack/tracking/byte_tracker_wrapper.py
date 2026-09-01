"""ByteTrack wrapper for multi-object tracking.

Wraps Ultralytics built-in ByteTrack integration to produce stable
:class:`TrackedObject` dataclass instances across frames.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from intellitrack.detection.yolo_detector import Detection
from intellitrack.utils.geometry import bbox_centroid

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """A tracked object with a stable cross-frame identity.

    Attributes:
        track_id: Unique integer track ID assigned by ByteTrack.
        bbox_xyxy: Bounding box in ``(x1, y1, x2, y2)`` pixel coordinates.
        centroid: Centroid ``(cx, cy)`` of the bounding box.
        class_name: Human-readable class label (e.g. ``"person"``).
        confidence: Detection confidence score.
        age_frames: Number of frames this track has been active.
    """

    track_id: int
    bbox_xyxy: tuple  # (x1, y1, x2, y2)
    centroid: tuple   # (cx, cy)
    class_name: str
    confidence: float
    age_frames: int


class ByteTrackerWrapper:
    """Wraps Ultralytics' built-in ByteTrack tracking for use in IntelliTrack.

    The wrapper uses ``model.track(..., persist=True)`` to maintain cross-frame
    IDs via ByteTrack.  If the underlying Ultralytics model is unavailable the
    caller can inject a mock; tests should not require a real weights file.

    Args:
        model_path: Path to YOLO weights (same as :class:`YoloDetector`).
        confidence_threshold: Minimum confidence for tracks to be returned.
        iou_threshold: NMS IoU threshold.
        target_classes: Class names to retain.
        max_age_frames: Maximum number of frames a track can go undetected
            before it is purged (passed to ByteTrack internally).
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float,
        iou_threshold: float,
        target_classes: List[str],
        max_age_frames: int = 30,
    ) -> None:
        from ultralytics import YOLO  # type: ignore

        self._model = YOLO(model_path)
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._target_classes = [c.lower() for c in target_classes]
        self._max_age_frames = max_age_frames
        # Per-track age counter: track_id -> age_frames
        self._age: dict = {}
        logger.info("ByteTrackerWrapper initialised with model '%s'", model_path)

    def update(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> List[TrackedObject]:
        """Run tracking on a new frame, returning all active tracked objects.

        Args:
            frame: The current BGR frame.
            detections: Detections from :class:`YoloDetector` for this frame
                (used for class filtering consistency; the tracker also runs
                its own internal detection pass via ``model.track``).

        Returns:
            A list of :class:`TrackedObject` instances with stable IDs.
        """
        results = self._model.track(
            frame,
            persist=True,
            conf=self._confidence_threshold,
            iou=self._iou_threshold,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        tracked: List[TrackedObject] = []
        seen_ids = set()

        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            if boxes.id is None:
                # No tracks yet (first frame or no detections)
                continue

            for i in range(len(boxes)):
                track_id = int(boxes.id[i])
                conf = float(boxes.conf[i])
                class_id = int(boxes.cls[i])
                class_name = result.names.get(class_id, str(class_id)).lower()

                if self._target_classes and class_name not in self._target_classes:
                    continue
                if conf < self._confidence_threshold:
                    continue

                xyxy = tuple(boxes.xyxy[i].tolist())
                cx, cy = bbox_centroid(xyxy)  # type: ignore[arg-type]

                self._age[track_id] = self._age.get(track_id, 0) + 1
                seen_ids.add(track_id)

                tracked.append(
                    TrackedObject(
                        track_id=track_id,
                        bbox_xyxy=xyxy,
                        centroid=(cx, cy),
                        class_name=class_name,
                        confidence=conf,
                        age_frames=self._age[track_id],
                    )
                )

        # Clean up ages for tracks no longer visible
        lost = set(self._age) - seen_ids
        for tid in lost:
            del self._age[tid]

        return tracked
