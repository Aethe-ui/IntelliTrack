"""Generate a four-way mode comparison report from experiment summaries.

Loads ``*_summary.json`` files from ``data/logs/``, writes
``data/logs/comparison_report.md`` plus three matplotlib plots.

Usage::

    python scripts/generate_report.py --log-dir data/logs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logger = logging.getLogger(__name__)

MODES = ["reactive_pid", "kalman", "lstm", "transformer"]
METRIC_KEYS = [
    "tracking_accuracy",
    "target_loss_events",
    "mean_latency_ms",
    "p95_latency_ms",
    "mean_fps",
    "prediction_rmse_px",
    "tracking_stability",
    "cpu_percent_mean",
]


def _load_summaries(log_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load the newest summary JSON per mode from ``log_dir``."""
    by_mode: Dict[str, Dict[str, Any]] = {}
    for path in sorted(log_dir.glob("*_summary.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        mode = data.get("mode")
        if mode in MODES:
            by_mode[mode] = data
            by_mode[mode]["_path"] = str(path)
    return by_mode


def _load_error_series(log_path: Optional[str], cx: float, cy: float) -> List[float]:
    """Per-frame distance of raw centroid from frame centre."""
    if not log_path:
        return []
    path = Path(log_path)
    if not path.is_file():
        return []
    errors: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("target_found"):
                continue
            rx, ry = r.get("raw_x"), r.get("raw_y")
            if rx is None or ry is None:
                continue
            errors.append(float(np.hypot(float(rx) - cx, float(ry) - cy)))
    return errors


def _fmt(val: Any) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate IntelliTrack comparison report")
    parser.add_argument("--log-dir", default=str(root / "data" / "logs"))
    parser.add_argument("--frame-width", type=float, default=640.0)
    parser.add_argument("--frame-height", type=float, default=480.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    summaries = _load_summaries(log_dir)
    if not summaries:
        logger.error("No *_summary.json files found in %s", log_dir)
        sys.exit(1)

    cx, cy = args.frame_width / 2.0, args.frame_height / 2.0
    plots_dir = log_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot (a): tracking error over time ---
    fig, ax = plt.subplots(figsize=(10, 4))
    for mode in MODES:
        if mode not in summaries:
            continue
        series = _load_error_series(summaries[mode].get("log_path"), cx, cy)
        if series:
            ax.plot(series, label=mode, linewidth=1.2)
    ax.set_xlabel("Frame (target-found only)")
    ax.set_ylabel("Pixel error from centre")
    ax.set_title("Per-frame tracking error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    err_plot = plots_dir / "tracking_error.png"
    fig.tight_layout()
    fig.savefig(err_plot, dpi=120)
    plt.close(fig)

    # --- Plot (b): mean latency bar ---
    modes_present = [m for m in MODES if m in summaries]
    latencies = [float(summaries[m].get("mean_latency_ms", 0.0)) for m in modes_present]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(modes_present, latencies, color="#3d9cf0")
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("Mean latency by mode")
    ax.tick_params(axis="x", rotation=20)
    lat_plot = plots_dir / "mean_latency.png"
    fig.tight_layout()
    fig.savefig(lat_plot, dpi=120)
    plt.close(fig)

    # --- Plot (c): mean FPS bar ---
    fps_vals = [float(summaries[m].get("mean_fps", 0.0)) for m in modes_present]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(modes_present, fps_vals, color="#5ecf8e")
    ax.set_ylabel("Mean FPS")
    ax.set_title("Mean FPS by mode")
    ax.tick_params(axis="x", rotation=20)
    fps_plot = plots_dir / "mean_fps.png"
    fig.tight_layout()
    fig.savefig(fps_plot, dpi=120)
    plt.close(fig)

    # --- Interpretation ---
    best_stability_mode = None
    best_stability = float("inf")
    best_loss_mode = None
    best_loss = float("inf")
    for mode in modes_present:
        stab = summaries[mode].get("tracking_stability")
        if stab is not None and float(stab) < best_stability:
            best_stability = float(stab)
            best_stability_mode = mode
        loss = summaries[mode].get("target_loss_events")
        if loss is not None and float(loss) < best_loss:
            best_loss = float(loss)
            best_loss_mode = mode

    # --- Markdown table ---
    header = "| Metric | " + " | ".join(modes_present) + " |"
    sep = "|---| " + " | ".join(["---"] * len(modes_present)) + " |"
    rows = [header, sep]
    for key in METRIC_KEYS:
        cells = [_fmt(summaries[m].get(key)) for m in modes_present]
        rows.append(f"| `{key}` | " + " | ".join(cells) + " |")

    report = f"""# IntelliTrack Four-Way Comparison Report

Generated from summary files in `{log_dir}`.

## Metrics table

{chr(10).join(rows)}

## Plots

### Tracking error over time

![Tracking error](plots/tracking_error.png)

### Mean latency

![Mean latency](plots/mean_latency.png)

### Mean FPS

![Mean FPS](plots/mean_fps.png)

## Interpretation

- Lowest tracking error (stability / std of pixel error from centre): **{best_stability_mode}** ({best_stability:.4f} px).
- Lowest target-loss event count: **{best_loss_mode}** ({int(best_loss)} events).

These results speak to the research question: whether predictive trajectory
estimation (LSTM / Transformer) improves tracking accuracy and effective
latency versus reactive PID and Kalman baselines on the recorded evaluation set.
If predictive modes underperform, limited training data (or missing checkpoints
falling back to constant-velocity) is a likely cause rather than a pipeline bug.
"""

    out = log_dir / "comparison_report.md"
    out.write_text(report, encoding="utf-8")
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
