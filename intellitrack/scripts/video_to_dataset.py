"""Convert a recorded video file into a trajectory CSV for predictor training.

Runs YOLO detection → ByteTrack tracking → centroid extraction on every
(or every *N*-th) frame and writes a CSV compatible with
:class:`~intellitrack.prediction.lstm_predictor.TrajectoryDataset` and
``scripts/train_predictor.py``.

Usage::

    # Basic — process the whole video
    python scripts/video_to_dataset.py --video data/recordings/demo.mp4

    # Custom output, skip every other frame, show preview
    python scripts/video_to_dataset.py \
        --video data/recordings/session.mp4 \
        --out data/datasets/session.csv \
        --stride 2 --show

    # Then train on the result
    python scripts/train_predictor.py --model lstm \
        --dataset data/datasets/session.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intellitrack.detection.yolo_detector import YoloDetector
from intellitrack.tracking.byte_tracker_wrapper import ByteTrackerWrapper
from intellitrack.utils.config import get, load_config

logger = logging.getLogger(__name__)


def _resolve_output_path(video_path: Path, out_arg: str | None, out_dir: Path) -> Path:
    """Determine the output CSV path."""
    if out_arg:
        return Path(out_arg)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = video_path.stem
    return out_dir / f"trajectories_{stem}_{ts}.csv"


def process_video(
    video_path: Path,
    output_path: Path,
    detector: YoloDetector,
    tracker: ByteTrackerWrapper,
    frame_w: int,
    frame_h: int,
    stride: int = 1,
    show: bool = False,
    max_frames: int | None = None,
) -> int:
    """Run detection + tracking on *video_path* and write trajectory CSV.

    Args:
        video_path: Path to the input video file.
        output_path: Destination for the trajectory CSV.
        detector: Initialised YOLO detector.
        tracker: Initialised ByteTrack tracker wrapper.
        frame_w: Logical frame width written to the CSV.
        frame_h: Logical frame height written to the CSV.
        stride: Process every *stride*-th frame (1 = every frame).
        show: If ``True``, open an OpenCV preview window.
        max_frames: Optional cap on the number of frames to process.

    Returns:
        Total number of trajectory rows written.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video file: %s", video_path)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    logger.info(
        "Video: %s — %d frames, %.1f FPS, stride=%d",
        video_path.name,
        total_frames,
        fps,
        stride,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "track_id",
        "centroid_x",
        "centroid_y",
        "frame_width",
        "frame_height",
    ]

    rows_written = 0
    frame_idx = 0
    processed = 0
    t0 = time.perf_counter()

    try:
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # Skip frames according to stride
                if (frame_idx - 1) % stride != 0:
                    continue

                # Resize to the expected logical resolution if needed
                h, w = frame.shape[:2]
                if w != frame_w or h != frame_h:
                    frame = cv2.resize(frame, (frame_w, frame_h))

                detections = detector.detect(frame)
                tracks = tracker.update(frame, detections)

                # Synthesise a timestamp from frame position and FPS
                synthetic_ts = frame_idx / fps

                for track in tracks:
                    writer.writerow(
                        {
                            "timestamp": synthetic_ts,
                            "track_id": track.track_id,
                            "centroid_x": track.centroid[0],
                            "centroid_y": track.centroid[1],
                            "frame_width": frame_w,
                            "frame_height": frame_h,
                        }
                    )
                    rows_written += 1

                processed += 1

                if show:
                    # Draw bounding boxes, centroids, and labels
                    vis = frame.copy()
                    for track in tracks:
                        x1, y1, x2, y2 = (int(v) for v in track.bbox_xyxy)
                        cx, cy = int(track.centroid[0]), int(track.centroid[1])
                        # Bounding box
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Centroid dot
                        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
                        # Label with ID and confidence
                        label = f"ID {track.track_id} {track.confidence:.2f}"
                        cv2.putText(
                            vis,
                            label,
                            (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                        )
                    cv2.imshow("video_to_dataset", vis)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        logger.info("Stopped early via preview window (q).")
                        break

                # Progress logging every 500 processed frames
                if processed % 500 == 0:
                    elapsed = time.perf_counter() - t0
                    pct = (frame_idx / total_frames * 100) if total_frames else 0
                    logger.info(
                        "Progress: frame %d/%d (%.0f%%) — %d rows — %.1fs elapsed",
                        frame_idx,
                        total_frames,
                        pct,
                        rows_written,
                        elapsed,
                    )

                if max_frames is not None and processed >= max_frames:
                    break
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t0
    logger.info(
        "Done: processed %d frames in %.1fs — wrote %d rows to %s",
        processed,
        elapsed,
        rows_written,
        output_path,
    )
    return rows_written


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Convert a recorded video into a trajectory CSV for predictor training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The output CSV has the same schema as record_dataset.py and is\n"
            "directly usable with train_predictor.py:\n\n"
            "  python scripts/train_predictor.py --model lstm --dataset <output.csv>"
        ),
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video file (e.g. data/recordings/demo.mp4).",
    )
    parser.add_argument(
        "--config",
        default=str(root / "configs" / "default.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path. Default: data/datasets/trajectories_<stem>_<timestamp>.csv",
    )
    parser.add_argument(
        "--out-dir",
        default=str(root / "data" / "datasets"),
        help="Directory for auto-named output CSV (ignored if --out is set).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every N-th frame (default: 1 = every frame).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after processing this many frames.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show annotated preview window (press q to stop early).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    video_path = Path(args.video)
    if not video_path.is_file():
        logger.error("Video file not found: %s", video_path)
        sys.exit(1)

    config = load_config(args.config)
    frame_w = get(config, "camera.width", 640)
    frame_h = get(config, "camera.height", 480)

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

    output_path = _resolve_output_path(video_path, args.out, Path(args.out_dir))

    rows = process_video(
        video_path=video_path,
        output_path=output_path,
        detector=detector,
        tracker=tracker,
        frame_w=frame_w,
        frame_h=frame_h,
        stride=args.stride,
        show=args.show,
        max_frames=args.max_frames,
    )

    if rows == 0:
        logger.warning(
            "No trajectory rows produced. Check that the video contains "
            "objects matching target_classes=%s in your config.",
            get(config, "detection.target_classes", ["person"]),
        )
    else:
        logger.info(
            "Ready to train:\n  python scripts/train_predictor.py "
            "--model lstm --dataset %s",
            output_path,
        )


if __name__ == "__main__":
    main()
