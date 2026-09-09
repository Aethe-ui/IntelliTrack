"""Run a bounded tracking experiment and write metrics + summary JSON.

By default this script is headless (metrics only). Pass ``--show`` to open a
live annotated preview window. For interactive tracking without an experiment
log, use ``scripts/run_live.py`` instead.

Usage::

    python scripts/run_experiment.py --mode kalman --duration-seconds 60
    python scripts/run_experiment.py --mode kalman --show
    python scripts/run_experiment.py --mode lstm --source recorded --video-path data/recordings/demo.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import psutil

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intellitrack.metrics.evaluator import evaluate_run
from intellitrack.pipeline.modes import TrackingMode
from intellitrack.pipeline.tracking_pipeline import TrackingPipeline
from intellitrack.utils.config import get, load_config

logger = logging.getLogger(__name__)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Run IntelliTrack experiment")
    parser.add_argument("--config", default=str(root / "configs" / "default.yaml"))
    parser.add_argument(
        "--mode",
        choices=[m.value for m in TrackingMode],
        required=True,
    )
    parser.add_argument("--duration-seconds", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--source", choices=["live", "recorded"], default="live")
    parser.add_argument("--video-path", type=str, default=None)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show annotated camera preview (press q to stop early).",
    )
    args = parser.parse_args()

    if args.duration_seconds is None and args.max_frames is None and args.source == "live":
        args.duration_seconds = 60.0

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    log_dir = Path(get(config, "logging.log_dir", "data/logs"))
    if not log_dir.is_absolute():
        log_dir = root / log_dir
    config.setdefault("logging", {})["log_dir"] = str(log_dir)

    camera_source = "file" if args.source == "recorded" else "live"
    video_path = args.video_path or get(config, "camera.file_path", None)
    if camera_source == "file" and not video_path:
        logger.error("--source recorded requires --video-path or camera.file_path in config")
        sys.exit(1)

    if not args.show:
        logger.info(
            "Running headless (metrics only). Use --show for a camera window, "
            "or run: python scripts/run_live.py --mode %s",
            args.mode,
        )

    proc = psutil.Process()
    cpu_samples: list[float] = []

    pipeline = TrackingPipeline(
        config=config,
        mode_override=args.mode,
        enable_metrics=True,
        camera_source=camera_source,
        video_path=video_path,
    )

    pipeline.start()
    t0 = time.perf_counter()
    frames = 0
    try:
        while True:
            pipeline.run_once()
            frames += 1
            cpu_samples.append(proc.cpu_percent(interval=None))

            if args.show:
                frame = pipeline.last_annotated_frame
                if frame is not None:
                    cv2.imshow("IntelliTrack — Experiment", frame)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        logger.info("Stopped early via preview window (q).")
                        break

            if pipeline._eof:
                break
            if args.max_frames is not None and frames >= args.max_frames:
                break
            if args.duration_seconds is not None and (time.perf_counter() - t0) >= args.duration_seconds:
                break
    finally:
        log_path = pipeline.metrics_log_path
        pipeline.stop()
        if args.show:
            cv2.destroyAllWindows()

    if not log_path:
        logger.error("No metrics log produced.")
        sys.exit(1)

    horizon = get(config, "prediction.horizon_frames", 5)
    cx = get(config, "camera.width", 640) / 2.0
    cy = get(config, "camera.height", 480) / 2.0
    summary = evaluate_run(log_path, horizon_frames=horizon, frame_center=(cx, cy))
    summary["cpu_percent_mean"] = float(sum(cpu_samples) / len(cpu_samples)) if cpu_samples else 0.0
    summary["log_path"] = log_path
    summary["duration_seconds"] = time.perf_counter() - t0
    summary["frames_processed"] = frames

    summary_path = Path(log_path).with_name(Path(log_path).stem + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Summary written to %s", summary_path)
    logger.info("Results: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
