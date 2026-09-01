"""Unit tests for the YOLO detector module.

Tests use unittest.mock to patch the module-level ``YOLO`` symbol in
``intellitrack.detection.yolo_detector`` so that no real weights file or
network access is needed.
"""

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_synthetic_frame(width: int = 320, height: int = 240) -> np.ndarray:
    """Create a simple synthetic BGR frame with a solid white rectangle."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[50:150, 80:200] = 255
    return frame


def make_mock_model_result(detections_data: List[dict], names: dict) -> MagicMock:
    """Build a mock Ultralytics result for ``model.predict()``."""
    result = MagicMock()
    result.names = names

    if not detections_data:
        result.boxes = None
        return result

    n = len(detections_data)
    boxes = MagicMock()
    boxes.__len__ = MagicMock(return_value=n)

    # Simulate tensor-indexing for xyxy, conf, cls
    xyxy_items = [MagicMock() for _ in detections_data]
    for i, (item, d) in enumerate(zip(xyxy_items, detections_data)):
        item.tolist.return_value = list(d["xyxy"])

    boxes.xyxy = xyxy_items
    boxes.conf = [d["conf"] for d in detections_data]
    boxes.cls = [d["cls_id"] for d in detections_data]

    result.boxes = boxes
    return result


def _make_detector(
    mock_model: MagicMock,
    confidence_threshold: float = 0.4,
    iou_threshold: float = 0.45,
    target_classes: list = None,
):
    """Create a YoloDetector with a mocked model (no real weights needed)."""
    from intellitrack.detection.yolo_detector import YoloDetector

    if target_classes is None:
        target_classes = ["person"]

    with patch("intellitrack.detection.yolo_detector.YOLO", return_value=mock_model):
        detector = YoloDetector(
            model_path="fake.pt",
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            target_classes=target_classes,
        )
    return detector


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_detect_returns_list():
    """YoloDetector.detect must always return a list."""
    names = {0: "person"}
    mock_result = make_mock_model_result(
        [{"xyxy": (10, 20, 100, 200), "conf": 0.8, "cls_id": 0}],
        names,
    )
    mock_model = MagicMock()
    mock_model.predict.return_value = [mock_result]
    mock_model.names = names

    detector = _make_detector(mock_model)

    frame = make_synthetic_frame()
    with patch("intellitrack.detection.yolo_detector.YOLO", return_value=mock_model):
        # The detector was already constructed, so just call detect directly
        result = detector.detect(frame)

    assert isinstance(result, list)


def test_confidence_filtering():
    """Detections below confidence_threshold must be dropped."""
    names = {0: "person"}
    detections_data = [
        {"xyxy": (0, 0, 50, 50), "conf": 0.9, "cls_id": 0},  # above threshold
        {"xyxy": (0, 0, 50, 50), "conf": 0.2, "cls_id": 0},  # BELOW threshold
    ]
    mock_result = make_mock_model_result(detections_data, names)
    mock_model = MagicMock()
    mock_model.predict.return_value = [mock_result]
    mock_model.names = names

    detector = _make_detector(mock_model, confidence_threshold=0.4)
    results = detector.detect(make_synthetic_frame())

    # Only the high-confidence detection should survive
    assert len(results) == 1
    assert results[0].confidence == pytest.approx(0.9)


def test_class_filtering():
    """Detections outside target_classes must be dropped."""
    names = {0: "person", 1: "car"}
    detections_data = [
        {"xyxy": (0, 0, 50, 50), "conf": 0.9, "cls_id": 0},    # person — keep
        {"xyxy": (0, 0, 80, 80), "conf": 0.85, "cls_id": 1},   # car — drop
    ]
    mock_result = make_mock_model_result(detections_data, names)
    mock_model = MagicMock()
    mock_model.predict.return_value = [mock_result]
    mock_model.names = names

    detector = _make_detector(mock_model, target_classes=["person"])
    results = detector.detect(make_synthetic_frame())

    assert len(results) == 1
    assert results[0].class_name == "person"


def test_empty_target_classes_returns_all():
    """When target_classes=[], all detected classes should be returned."""
    names = {0: "person", 1: "car"}
    detections_data = [
        {"xyxy": (0, 0, 50, 50), "conf": 0.9, "cls_id": 0},
        {"xyxy": (0, 0, 80, 80), "conf": 0.85, "cls_id": 1},
    ]
    mock_result = make_mock_model_result(detections_data, names)
    mock_model = MagicMock()
    mock_model.predict.return_value = [mock_result]
    mock_model.names = names

    detector = _make_detector(mock_model, target_classes=[])
    results = detector.detect(make_synthetic_frame())

    assert len(results) == 2


def test_detection_dataclass_fields():
    """Returned Detection objects must have the expected field types."""
    from intellitrack.detection.yolo_detector import Detection

    names = {0: "person"}
    detections_data = [
        {"xyxy": (10.0, 20.0, 100.0, 200.0), "conf": 0.75, "cls_id": 0},
    ]
    mock_result = make_mock_model_result(detections_data, names)
    mock_model = MagicMock()
    mock_model.predict.return_value = [mock_result]
    mock_model.names = names

    detector = _make_detector(mock_model)
    results = detector.detect(make_synthetic_frame())

    assert len(results) == 1
    det = results[0]
    assert isinstance(det, Detection)
    assert isinstance(det.bbox_xyxy, tuple)
    assert isinstance(det.confidence, float)
    assert isinstance(det.class_id, int)
    assert isinstance(det.class_name, str)
