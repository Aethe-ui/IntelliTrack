"""Record trajectory datasets for predictor training.

Runs camera + detection + tracking (no control) and writes one CSV row per
frame with centroid data.

Usage::

    python scripts/record_dataset.py --config configs/default.yaml --duration 120
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intellitrack.capture.camera import Camera, CameraUnavailableError
from intellitrack.detection.yolo_detector import YoloDetector
from intellitrack.tracking.byte_tracker_wrapper import ByteTrackerWrapper
from intellitrack.tracking.target_selector import TargetSelector
from intellitrack.utils.config import get, load_config

logger = logging.getLogger(__name__)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Record trajectory CSV dataset")
    parser.add_argument("--config", default=str(root / "configs" / "default.yaml"))
    parser.add_argument("--duration", type=float, default=120.0, help="Seconds to record")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--out-dir",
        default=str(root / "data" / "datasets"),
        help="Directory for output CSV",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    frame_w = get(config, "camera.width", 640)
    frame_h = get(config, "camera.height", 480)

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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"trajectories_{ts}.csv"

    try:
        camera.start()
    except CameraUnavailableError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    fieldnames = [
        "timestamp",
        "track_id",
        "centroid_x",
        "centroid_y",
        "frame_width",
        "frame_height",
    ]
    n = 0
    t0 = time.perf_counter()
    logger.info("Recording to %s for %.1fs …", out_path, args.duration)

    try:
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            while True:
                ret, frame = camera.read()
                if not ret or frame is None:
                    continue
                detections = detector.detect(frame)
                tracks = tracker.update(frame, detections)
                target = selector.select(tracks)
                if target is not None:
                    writer.writerow(
                        {
                            "timestamp": time.time(),
                            "track_id": target.track_id,
                            "centroid_x": target.centroid[0],
                            "centroid_y": target.centroid[1],
                            "frame_width": frame_w,
                            "frame_height": frame_h,
                        }
                    )
                    n += 1
                if args.max_frames is not None and n >= args.max_frames:
                    break
                if (time.perf_counter() - t0) >= args.duration:
                    break
    finally:
        camera.release()

    logger.info("Wrote %d rows to %s", n, out_path)


if __name__ == "__main__":
    main()
