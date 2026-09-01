"""YOLO-based object detector module.

Wraps Ultralytics YOLO inference and provides a clean :class:`Detection`
dataclass interface for downstream tracking modules.
"""

import logging
from dataclasses import dataclass
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO  # type: ignore
except ImportError:  # pragma: no cover
    YOLO = None  # type: ignore[assignment,misc]


@dataclass
class Detection:
    """A single detected object in a frame.

    Attributes:
        bbox_xyxy: Bounding box in ``(x1, y1, x2, y2)`` pixel coordinates.
        confidence: Detection confidence score in ``[0, 1]``.
        class_id: Integer class identifier from the YOLO model.
        class_name: Human-readable class label (e.g. ``"person"``).
    """

    bbox_xyxy: tuple  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str


class YoloDetector:
    """Run YOLO inference on frames and return filtered :class:`Detection` objects.

    All tunable parameters (confidence threshold, IoU threshold, target class
    names) are injected at construction time from the application config — no
    magic numbers live inside this class.

    Args:
        model_path: Path (or Ultralytics shorthand, e.g. ``"yolo11n.pt"``) to
            the YOLO weights file.
        confidence_threshold: Minimum confidence score to retain a detection.
        iou_threshold: NMS IoU threshold passed to YOLO inference.
        target_classes: List of class name strings to keep (e.g. ``["person"]``).
            If empty, all detected classes are returned.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float,
        iou_threshold: float,
        target_classes: List[str],
    ) -> None:
        if YOLO is None:  # pragma: no cover
            raise ImportError(
                "ultralytics is required for YoloDetector. "
                "Install it with: pip install ultralytics"
            )
        self._model = YOLO(model_path)
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._target_classes: List[str] = [c.lower() for c in target_classes]
        logger.info(
            "YoloDetector loaded '%s' (conf=%.2f, iou=%.2f, classes=%s)",
            model_path,
            confidence_threshold,
            iou_threshold,
            self._target_classes,
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a single BGR frame and return filtered detections.

        Args:
            frame: A BGR image as returned by ``cv2.VideoCapture.read``.

        Returns:
            A list of :class:`Detection` objects that pass all filters.
        """
        results = self._model.predict(
            frame,
            conf=self._confidence_threshold,
            iou=self._iou_threshold,
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                if conf < self._confidence_threshold:
                    continue

                class_id = int(boxes.cls[i])
                class_name = result.names.get(class_id, str(class_id)).lower()

                if self._target_classes and class_name not in self._target_classes:
                    continue

                xyxy = boxes.xyxy[i].tolist()
                detections.append(
                    Detection(
                        bbox_xyxy=tuple(xyxy),
                        confidence=conf,
                        class_id=class_id,
                        class_name=class_name,
                    )
                )

        return detections
