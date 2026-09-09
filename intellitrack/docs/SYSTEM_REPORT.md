# IntelliTrack — System Report

**What works, how it works, how to test it, and how the LSTM model fits in.**

---

## 1. What works today

The MVP is complete end-to-end. With no camera/Arduino attached, software modules still run via mocks; with hardware attached, the same pipeline drives real servos.

| Capability | Status | Notes |
|---|---|---|
| Webcam / video-file capture | Works | Live index or `camera.source: file` |
| YOLOv11 person detection | Works | Default `yolo11n.pt` |
| ByteTrack multi-object tracking | Works | Stable track IDs across frames |
| Target selection | Works | closest-to-center, largest bbox, highest confidence, manual ID lock |
| Reactive PID mode | Works | No prediction — uses current centroid |
| Kalman prediction mode | Works | Constant-velocity 2D Kalman |
| LSTM prediction mode | Works | Trained checkpoint **or** constant-velocity fallback |
| Transformer prediction mode | Works | Same interface / fallback as LSTM |
| PID → pan/tilt mapping | Works | Deadband + angle clamping |
| Serial bridge to Arduino/ESP32 | Works | Real port or mock logging |
| FastAPI live dashboard | Works | MJPEG stream, metrics, runtime config |
| Dataset recording + predictor training | Works | CSV trajectories → `.pt` checkpoints |
| Experiment harness + comparison report | Works | Four-mode metrics + plots |

**Research modes (switch via config or CLI `--mode`):**

1. `reactive_pid` — YOLO + PID (baseline)
2. `kalman` — YOLO + Kalman Filter
3. `lstm` — YOLO + LSTM trajectory prediction
4. `transformer` — YOLO + Transformer trajectory prediction

---

## 2. How it works

### 2.1 Pipeline

```
Camera
  → YoloDetector          (bboxes + class + confidence)
  → ByteTrackerWrapper    (stable track_id + centroid)
  → TargetSelector        (one active target)
  → Prediction stage      (mode-dependent future / current position)
  → PID × 2 (pan, tilt)   (pixel error from frame centre)
  → ServoMapper           (degrees, deadband, clamp)
  → SerialBridge          (PAN:<int> TILT:<int>  or mock log)
```

Each frame produces a `FrameMetrics` record: timestamp, mode, whether a target was found, raw vs predicted centroid, pan/tilt degrees, latency, and instantaneous FPS.

### 2.2 Prediction stages (the research core)

| Mode | What becomes the PID input |
|---|---|
| `reactive_pid` | Latest observed centroid (`NoPrediction`) |
| `kalman` | One-step Kalman `predict()` after `update(measurement)` |
| `lstm` | LSTM forecast of position **horizon_frames** ahead |
| `transformer` | Same horizon, Transformer encoder instead of LSTM |

**Why prediction matters:** reactive control only sees where the target *is*. Predictive modes estimate where it *will be*, so the servos can start moving earlier and reduce effective lag.

### 2.3 Control loop

1. Frame centre is `(width/2, height/2)`.
2. Error `(dx, dy) = predicted_centroid − centre`.
3. PID (gains from config) outputs degree deltas, clamped by `output_limit`.
4. Errors inside `deadband_px` are ignored (reduces jitter).
5. Absolute pan/tilt are clamped to configured min/max and sent to hardware.

### 2.4 Hardware path

Firmware (`firmware/pan_tilt_controller/pan_tilt_controller.ino`):

- Pan → pin 9, tilt → pin 10
- Command: `PAN:<int> TILT:<int>\n`
- Reply: `ACK\n` (boot: centres at 90° and prints `READY`)

If `hardware.enabled` is false, or the serial port cannot open and `mock_if_unavailable` is true, commands are **logged only** — the rest of the stack keeps running.

### 2.5 Dashboard

`uvicorn intellitrack.api.main:app --port 8000`

- `/` — HTML UI
- `/stream` — MJPEG annotated feed
- `/metrics` — latest FPS, latency, mode, pan/tilt
- `GET/POST /config` — change mode, PID gains, selection strategy live

### 2.6 Config as single source of truth

All tunables live in `configs/default.yaml` (camera, detection, tracking, prediction, PID, servo, hardware, API, logging). Scripts accept `--config` and usually `--mode`.

---

## 3. How to test

### 3.1 Automated unit / API tests

From `intellitrack/` with the venv active:

```bash
pytest
```

| Test file | What it covers |
|---|---|
| `tests/test_detection.py` | YOLO wrapper filtering (mocked model) |
| `tests/test_tracking.py` | Target selection strategies + lock/loss |
| `tests/test_kalman.py` | Linear trajectory convergence |
| `tests/test_pid.py` | Closed-loop trend + output clamp |
| `tests/test_serial_bridge.py` | Invalid port → mock; send_angles logs |
| `tests/test_prediction.py` | LSTM/Transformer train loss ↓; fallback without checkpoint |
| `tests/test_api.py` | `/config` GET/POST, `/metrics` schema (stub pipeline) |

Expected: **35 passed** (no camera/Arduino required).

### 3.2 Manual / integration checks

**Live tracking (webcam)**

```bash
python scripts/run_live.py --config configs/default.yaml --mode reactive_pid
# try: --mode kalman | lstm | transformer
```

Expect: annotated window, red selected target, FPS, pan/tilt overlay; `q` to quit.

**Dashboard**

```bash
uvicorn intellitrack.api.main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000 — switch mode in the form, watch metrics update
```

**Hardware (optional)**

1. Flash the `.ino` firmware.
2. Set `hardware.enabled: true` and the correct `serial_port`.
3. Run live or dashboard; moving a person should pan/tilt the mount.
4. With cable unplugged, logs should show mock mode without crashing.

**Record → train → run LSTM**

```bash
python scripts/record_dataset.py --duration 120
python scripts/train_predictor.py --model lstm --dataset data/datasets/trajectories_<ts>.csv
python scripts/run_live.py --mode lstm
```

Logs should show both **raw** and **predicted** centroids each frame.

**Experiment + report**

```bash
python scripts/run_experiment.py --mode kalman --duration-seconds 60
# repeat for reactive_pid, lstm, transformer (same video for a fair compare):
python scripts/run_experiment.py --mode lstm --source recorded --video-path data/recordings/demo.mp4

python scripts/generate_report.py --log-dir data/logs
```

Output: `data/logs/comparison_report.md` plus plots under `data/logs/plots/`.

---

## 4. The LSTM model (in depth)

### 4.1 Role in the system

The LSTM is the **core research differentiator**: instead of reacting to the current track centroid, it predicts where that centroid will be in `horizon_frames` (default **5**) using the last `sequence_length` (default **15**) observations. That predicted point is what the PID tries to centre.

Implementation: `src/intellitrack/prediction/lstm_predictor.py`

### 4.2 Features

Each history point becomes a 4-D feature:

| Feature | Meaning |
|---|---|
| `x`, `y` | Centroid position (normalised by frame width/height for the network) |
| `vx`, `vy` | Frame-to-frame deltas (first step = 0) |

So the model sees both position and instantaneous velocity.

### 4.3 Architecture

`LSTMTrajectoryPredictor` (`nn.Module`):

```
Input:  (batch, sequence_length, 4)
  → nn.LSTM(hidden_size=64, num_layers=2, batch_first=True)
  → take last timestep hidden state
  → Linear(hidden_size → 2)
Output: normalised (x, y) at horizon
```

Defaults (from `configs/default.yaml`):

- `hidden_size: 64`
- `num_layers: 2`
- Checkpoint: `data/models/lstm_predictor.pt`

### 4.4 Training data

`scripts/record_dataset.py` writes CSV rows:

`timestamp, track_id, centroid_x, centroid_y, frame_width, frame_height`

`TrajectoryDataset` builds sliding windows per track:

- **Input:** `sequence_length` feature frames  
- **Target:** true `(x, y)` at `horizon_frames` after the window end  

Shared loop in `prediction/train.py`: Adam + MSE, validation split, save best checkpoint by val loss.

```bash
python scripts/train_predictor.py --model lstm --dataset path/to.csv --epochs 20
```

### 4.5 Runtime wrapper (`LSTMPredictor`)

Public API (same as `NoPrediction` / Transformer):

```python
predict_next(history: list[tuple[float, float]]) -> tuple[float, float]
```

Behaviour:

1. If `lstm_predictor.pt` loads successfully → run the network, denormalise to pixels.
2. If the file is missing or load fails → **constant-velocity extrapolation** over `horizon_frames` (log a warning; never crash).
3. History shorter than 2 points → same CV fallback.

This keeps `prediction.mode: lstm` usable before any training.

### 4.6 How to verify the LSTM specifically

**Automated**

```bash
pytest tests/test_prediction.py -k lstm
```

Asserts: training loss decreases on a tiny synthetic linear trajectory; `predict_next` returns two finite floats with no checkpoint.

**Manual quality check**

1. Record ≥ ~2 minutes of people moving across frame.
2. Train until train/val loss plateaus.
3. Run live with `--mode lstm` and compare logged `raw=(…)` vs `predicted=(…)`.
4. On the same recorded video, run `run_experiment.py` for all four modes; inspect `prediction_rmse_px` in the LSTM/Transformer summaries (RMSE of prediction at `t` vs raw at `t + horizon`).

### 4.7 Limitations (honest)

- Quality depends on trajectory data diversity (speed, turns, occlusions). Sparse data → weak LSTM; fallback CV may dominate.
- Default YOLO class is `person` only; other objects need `detection.target_classes` changes.
- LSTM adds compute vs reactive/Kalman; expect slightly higher latency / lower FPS on CPU.
- Horizon is fixed in config; it is not adaptive per target speed.

---

## 5. Suggested verification checklist

- [ ] `pytest` — all green  
- [ ] `run_live.py` — boxes + FPS with webcam  
- [ ] Mode switch `reactive_pid` → `kalman` → `lstm` (config or dashboard)  
- [ ] Unplug serial / `hardware.enabled: false` — mock logs, no crash  
- [ ] (Optional) Flash Arduino — physical pan/tilt follows person  
- [ ] Record CSV → train LSTM → checkpoint appears under `data/models/`  
- [ ] `run_experiment.py` × 4 modes → `generate_report.py` produces table + 3 plots  

---

## 6. Key file map

| Area | Path |
|---|---|
| Pipeline | `src/intellitrack/pipeline/tracking_pipeline.py` |
| LSTM | `src/intellitrack/prediction/lstm_predictor.py` |
| Train loop | `src/intellitrack/prediction/train.py` |
| Serial | `src/intellitrack/control/serial_bridge.py` |
| API | `src/intellitrack/api/main.py` |
| Config | `configs/default.yaml` |
| Live demo | `scripts/run_live.py` |
| Experiments | `scripts/run_experiment.py`, `scripts/generate_report.py` |

---

*Generated for the IntelliTrack MVP (Phases 0–9). Sample quantitative outputs may live in `data/logs/comparison_report.md` after you run the experiment suite.*
