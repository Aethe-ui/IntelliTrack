# IntelliTrack

**AI-driven pan-tilt vision platform for predictive object tracking.**

IntelliTrack is a research-grade vision system that detects, tracks, and *predicts* the future position of a moving object in real time, then drives a two-axis servo mount (via Arduino/ESP32) to keep the object centered in frame. The core research contribution is a **predictive tracking** approach (LSTM/Transformer trajectory prediction) compared against purely reactive baselines (PID-only and Kalman filter), providing a measurable answer to: *Does predictive trajectory estimation improve tracking accuracy and reduce effective latency compared to reactive control on low-cost embedded hardware?*

---

## Pipeline

```
Camera → YOLO Detection → Multi-Object Tracker → Target Selection
       → Trajectory Prediction → Kalman Filter → PID Controller
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

### 1. Clone and enter the project

```bash
git clone <repo-url>
cd intellitrack
```

### 2. Create a virtual environment and install dependencies

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 3. Copy and configure the environment file

```bash
cp .env.example .env
# Edit .env as needed (CAMERA_INDEX, SERIAL_PORT, etc.)
```

---

## Running Live Tracking

```bash
python scripts/run_live.py --config configs/default.yaml --mode reactive_pid
```

Switch modes via the `--mode` flag or by editing `prediction.mode` in `configs/default.yaml`.

---

## Recording a Dataset

```bash
python scripts/record_dataset.py --config configs/default.yaml --duration 120
```

Saves trajectory CSV to `data/datasets/`.

---

## Training a Predictor

```bash
python scripts/train_predictor.py --model lstm --dataset data/datasets/<file>.csv --config configs/default.yaml
python scripts/train_predictor.py --model transformer --dataset data/datasets/<file>.csv --config configs/default.yaml
```

---

## Running the Experiment Suite

```bash
python scripts/run_experiment.py --mode kalman --duration-seconds 60
python scripts/run_experiment.py --mode lstm --duration-seconds 60
```

---

## Generating the Comparison Report

```bash
python scripts/generate_report.py
```

Output: `data/logs/comparison_report.md` with a four-way comparison table and plots.

---

## Arduino Firmware

Flash `firmware/pan_tilt_controller/pan_tilt_controller.ino` to your Arduino/ESP32.  
Connect pan servo to **pin 9**, tilt servo to **pin 10**.  
Set `hardware.enabled: true` and `hardware.serial_port` in `configs/default.yaml`.

---

## Running Tests

```bash
pytest
```

---

## Configuration

All runtime behaviour is controlled via `configs/default.yaml`.  
CLI scripts accept `--config path/to.yaml` and `--mode <mode>` overrides.  
See `configs/default.yaml` for all available keys and their default values.

---

## Known Limitations / Future Work

- Edge deployment (ONNX/TensorRT, Jetson Nano/Raspberry Pi) — not in MVP scope.
- Person re-identification and multi-target simultaneous servoing — future work.
- Sensor fusion (IMU, range sensors), auto-zoom — future work.
- Authentication/user accounts on the dashboard — future work.
