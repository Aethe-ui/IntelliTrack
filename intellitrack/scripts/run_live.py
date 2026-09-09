"""Live tracking demo script.

Runs the full :class:`~intellitrack.pipeline.tracking_pipeline.TrackingPipeline`
and displays an annotated video window.

Usage::

    python scripts/run_live.py --config configs/default.yaml \\
                               --mode reactive_pid

Exit with the ``q`` key.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intellitrack.pipeline.modes import TrackingMode
from intellitrack.pipeline.tracking_pipeline import TrackingPipeline
from intellitrack.utils.config import get, load_config

logger = logging.getLogger(__name__)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="IntelliTrack live demo")
    parser.add_argument(
        "--config",
        type=str,
        default=str(root / "configs" / "default.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=[m.value for m in TrackingMode],
        help="Override prediction.mode from config.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    mode_str = args.mode or get(config, "prediction.mode", "reactive_pid")
    logger.info("Starting live demo in mode '%s'.", mode_str)

    pipeline = TrackingPipeline(config, mode_override=mode_str)
    pipeline.start()

    logger.info("Live window open — press 'q' to quit.")
    try:
        while True:
            pipeline.run_once()
            frame = pipeline.last_annotated_frame
            if frame is None:
                time.sleep(0.01)
                continue
            cv2.imshow("IntelliTrack — Live", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        logger.info("Live demo stopped.")


if __name__ == "__main__":
    main()
