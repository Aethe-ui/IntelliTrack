# IntelliTrack

**AI-driven pan-tilt vision platform for predictive object tracking.**

IntelliTrack detects, tracks, and *predicts* the future position of a moving object in real time, then drives a two-axis servo mount (via Arduino/ESP32) to keep the object centered in frame. The MVP compares four modes — reactive PID, Kalman, LSTM, and Transformer — to measure whether predictive trajectory estimation improves tracking accuracy and effective latency on low-cost hardware.

---

## Pipeline

```
Camera → YOLO Detection → Multi-Object Tracker → Target Selection
       → Trajectory Prediction → PID Controller
       → Arduino/ESP32 → Pan-Tilt Servos
```

## Tracking Modes

| Mode | Description |
|---|---|
| `reactive_pid` | YOLO + PID (pure reactive baseline) |
| `kalman` | YOLO + Kalman Filter prediction |
| `lstm` | YOLO + LSTM trajectory prediction |
| `transformer` | YOLO + Transformer trajectory prediction |

---

## Setup

```bash
cd intellitrack
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

---

## Live Tracking

```bash
python scripts/run_live.py --config configs/default.yaml --mode reactive_pid
```

Press `q` to quit. Modes: `reactive_pid`, `kalman`, `lstm`, `transformer`.

With no Arduino attached the pipeline logs servo commands in **mock mode** and continues normally. Set `hardware.enabled: true` in config to send real serial commands.

---

## Web Dashboard

```bash
uvicorn intellitrack.api.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 for the live MJPEG stream, metrics panel, and runtime mode/PID controls.

---

## Arduino Firmware

1. Flash `firmware/pan_tilt_controller/pan_tilt_controller.ino` with the Arduino IDE (or `arduino-cli`).
2. Pan servo → **pin 9**, tilt servo → **pin 10**.
3. Serial protocol: `PAN:<int> TILT:<int>\n` → firmware replies `ACK\n`.
4. Set in `configs/default.yaml`:

```yaml
hardware:
  enabled: true
  serial_port: "/dev/ttyUSB0"   # or COMx on Windows
  baud_rate: 115200
  mock_if_unavailable: true
```

---

## Record a Dataset & Train Predictors

```bash
python scripts/record_dataset.py --config configs/default.yaml --duration 120
python scripts/train_predictor.py --model lstm --dataset data/datasets/trajectories_<ts>.csv
python scripts/train_predictor.py --model transformer --dataset data/datasets/trajectories_<ts>.csv
```

Checkpoints are written to `data/models/`. If a checkpoint is missing, LSTM/Transformer modes fall back to constant-velocity extrapolation so the pipeline never crashes.

---

## Experiments & Comparison Report

Record a reference video (or use a live camera), then run each mode:

```bash
# Against a recorded video (fair four-way comparison)
python scripts/run_experiment.py --mode reactive_pid --source recorded --video-path data/recordings/demo.mp4
python scripts/run_experiment.py --mode kalman --source recorded --video-path data/recordings/demo.mp4
python scripts/run_experiment.py --mode lstm --source recorded --video-path data/recordings/demo.mp4
python scripts/run_experiment.py --mode transformer --source recorded --video-path data/recordings/demo.mp4

# Or live for N seconds
python scripts/run_experiment.py --mode kalman --duration-seconds 60

python scripts/generate_report.py --log-dir data/logs
```

Report output: `data/logs/comparison_report.md` (table + three plots under `data/logs/plots/`).

Set `camera.source: file` and `camera.file_path` in config as an alternative to CLI `--video-path`.

---

## Predictor training and validation

`scripts/train_predictor.py --model transformer --dataset data/datasets/archive_trajectories.csv --seed 42`
uses scene/video groups (the prefix before the last `:` in `track_id`) to hold out
whole scenes. All windows from a scene stay together. At least two groups with
usable windows are required; datasets without window provenance are rejected.
Plain IDs remain separate track groups, with a warning: they do not establish
scene isolation. Do not concatenate recordings with reused track IDs.

Both models retain 15 observations with `[x, y, vx, vy]` features and one target
at the fifth future observation. Positions and velocities use the configured
camera width/height for normalization. Training optimizes MSE; `val_fde_px` is
the mean Euclidean pixel error for that one point. Epoch losses are weighted by
sample count, and the checkpoint with lowest validation MSE is saved as a plain
state dictionary. Seed 42 controls scene selection, CLI initialization, and
training randomness; exact results across devices are not guaranteed.

CSV `frame_width`/`frame_height` columns, when present, must match configuration.
Otherwise training logs assumed dimensions and full raw coordinate ranges;
out-of-bounds values trigger a warning without clipping or automatic resizing.
Ranges alone cannot establish source resolution or detect a smaller-scale CSV.

The two local archive CSVs contain identical IDs, timestamps, and labels across
842,074 rows, but different coordinates: `archive_trajectories.csv` has x twice
and y 1.5 times those in `trajectories_archive.csv`. Their ranges respectively are
x=2.1875..1235.3125, y=1.0547..704.3555 and x=1.09375..617.65625,
y=0.7031..469.5703. This is consistent with 1280×720 versus 640×480 export scales;
neither file records authoritative dimensions. The original command selects
`archive_trajectories.csv`, checked with the configured 1280×720 assumption.

The archive timestamps advance by **10 source frames per observation**. Thus the
preserved five-observation horizon spans 50 source frames in these CSVs, rather
than five original camera frames. Match sampling cadence when assessing runtime
quality; this fix does not change dataset sampling or target indexing.

Local verification artifacts (logs, isolated smoke checkpoints, split IDs and
metrics) are under `data/logs/training_verification/`. The full-archive one-epoch
Transformer smoke run used 48 training scenes / 12 validation scenes and
497,869 / 149,485 windows with seed 42. It verifies the pipeline, not convergence.

## Tests

```bash
pytest
```

---

## Configuration

All runtime behaviour is controlled via `configs/default.yaml`. CLI scripts accept `--config` and `--mode` overrides.

---

## Known Limitations / Future Work

- Edge deployment (ONNX/TensorRT, Jetson/Pi) — out of MVP scope.
- Person re-identification and multi-target simultaneous servoing — future work.
- Sensor fusion, auto-zoom — future work.
- Dashboard authentication — future work.
