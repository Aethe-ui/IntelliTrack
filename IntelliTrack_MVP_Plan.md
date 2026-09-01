# IntelliTrack — MVP Build Plan for an AI Coding Agent

This document is a complete, phase-by-phase build spec for the IntelliTrack MVP. It is written to be handed to an AI coding agent (e.g. Claude Code) one phase at a time. Each phase is self-contained: it lists exact files to create, function signatures, config keys, and acceptance criteria the agent must satisfy before moving to the next phase. Do not skip phases and do not start a phase until the previous phase's acceptance criteria pass.

---

## 0. Project Snapshot

**What it is:** An AI-driven pan-tilt vision platform that detects, tracks, and predicts the future position of a moving object in real time, then drives a two-axis servo mount (via Arduino/ESP32) to keep the object centered in frame.

**Core research differentiator:** Predictive tracking (LSTM/Transformer trajectory prediction) instead of purely reactive tracking (PID only). The MVP must produce a working, measurable comparison between four tracking modes:
1. YOLO + PID (reactive baseline)
2. YOLO + Kalman Filter
3. YOLO + LSTM trajectory prediction
4. YOLO + Transformer trajectory prediction

**Research question the MVP must be able to answer with data:** Does predictive trajectory estimation improve tracking accuracy and reduce effective latency compared to reactive control, on low-cost embedded hardware?

**Pipeline (from concept doc):**
`Camera → YOLO Detection → Multi-Object Tracker → Target Selection → Trajectory Prediction → Kalman Filter → PID Controller → Arduino/ESP32 → Pan-Tilt Servos`

**Stack:** Python 3.10+, OpenCV, Ultralytics YOLOv11, PyTorch, FastAPI, ByteTrack, PySerial, Arduino/ESP32 firmware, optional Raspberry Pi/Jetson edge deployment (out of scope for MVP, noted as future work).

---

## 1. MVP Scope — In and Out

**In scope for MVP:**
- Single-camera, single-target tracking (multi-object detection/tracking, but only one active target at a time).
- All four tracking modes listed above, switchable via config.
- Real-time video pipeline running on a normal laptop CPU/GPU (no Jetson/Pi required to complete MVP).
- Arduino/ESP32 serial bridge with a mock/simulation fallback when no hardware is connected.
- A minimal FastAPI dashboard: live annotated video stream, current mode, live metrics, runtime config changes.
- A metrics/logging system and an experiment script that produces a comparison report across the four modes.
- Unit tests for every non-trivial module.

**Explicitly out of scope for MVP (future work, do not build now):**
- Person re-identification, activity recognition, reinforcement-learning adaptive control.
- Multi-target simultaneous tracking/servoing.
- Edge deployment / model quantization (ONNX/TensorRT export).
- Sensor fusion (IMU, range sensors), auto-zoom, camera stabilization.
- Authentication/user accounts on the dashboard.

If the agent is unsure whether something is in scope, it must default to leaving it out and noting it as a TODO in README rather than building it.

---

## 2. Repository Structure

Create exactly this structure in Phase 0. All later phases only fill in files that already have a placeholder here.

```
intellitrack/
  configs/
    default.yaml
  firmware/
    pan_tilt_controller/
      pan_tilt_controller.ino
  src/
    intellitrack/
      __init__.py
      capture/
        __init__.py
        camera.py
      detection/
        __init__.py
        yolo_detector.py
      tracking/
        __init__.py
        byte_tracker_wrapper.py
        target_selector.py
      prediction/
        __init__.py
        kalman.py
        lstm_predictor.py
        transformer_predictor.py
        no_prediction.py
        train.py
      control/
        __init__.py
        pid_controller.py
        servo_mapper.py
        serial_bridge.py
      pipeline/
        __init__.py
        modes.py
        tracking_pipeline.py
      metrics/
        __init__.py
        logger.py
        evaluator.py
      api/
        __init__.py
        main.py
        routes_stream.py
        routes_config.py
        routes_metrics.py
        schemas.py
        static/
          index.html
      utils/
        __init__.py
        config.py
        geometry.py
  scripts/
    run_live.py
    run_experiment.py
    generate_report.py
    record_dataset.py
    train_predictor.py
  data/
    recordings/
    logs/
    models/
    datasets/
  tests/
    test_detection.py
    test_tracking.py
    test_kalman.py
    test_pid.py
    test_prediction.py
    test_serial_bridge.py
    test_api.py
  notebooks/
    01_explore_trajectories.ipynb
  requirements.txt
  .env.example
  .gitignore
  README.md
```

---

## 3. Global Conventions the Agent Must Follow

- Python 3.10+, PEP8, type hints on every function signature, docstrings on every public class/function.
- No hardcoded magic numbers in pipeline code — everything tunable lives in `configs/default.yaml` and is loaded through `utils/config.py`.
- Use `dataclasses` for domain objects (`Detection`, `TrackedObject`, `FrameMetrics`, etc.), not raw dicts/tuples passed between modules.
- Use the `logging` module (not `print`) for runtime diagnostics; use the dedicated `metrics/logger.py` for structured per-frame data.
- Every loop that touches hardware (camera, serial) must degrade gracefully (log a warning and continue in mock/simulation mode) rather than crash if the device is unavailable — the whole pipeline must run end-to-end even with zero hardware attached.
- After finishing each phase: run the relevant tests, fix failures, then commit with message `Phase N: <short description>` before starting the next phase.
- Do not proceed to the next phase until the current phase's "Acceptance Criteria" section is fully satisfied.
- Config file is the single source of truth for runtime behavior; CLI scripts should accept `--config path/to.yaml` and `--mode {reactive_pid,kalman,lstm,transformer}` overrides.

---

## Phase 0 — Environment & Repo Scaffolding

**Goal:** A clean, installable, empty-but-structured repo.

**Tasks:**
1. Create the full directory/file tree from Section 2, with empty `__init__.py` files and placeholder files containing only a module docstring (e.g. `"""Camera capture module."""`).
2. Write `requirements.txt` pinning at least: `ultralytics`, `opencv-python`, `torch`, `torchvision`, `fastapi`, `uvicorn[standard]`, `pydantic`, `pyserial`, `numpy`, `pandas`, `matplotlib`, `PyYAML`, `pytest`, `python-multipart`, `websockets`.
3. Write `.env.example` with: `CAMERA_INDEX=0`, `SERIAL_PORT=/dev/ttyUSB0`, `SERIAL_BAUD=115200`.
4. Write `configs/default.yaml` with top-level sections (fill values with sane defaults; every later phase references keys from here — do not invent parallel config mechanisms):
   ```yaml
   camera:
     index: 0
     width: 640
     height: 480
     fps_target: 30
   detection:
     model_path: "yolo11n.pt"
     confidence_threshold: 0.4
     iou_threshold: 0.45
     target_classes: ["person"]
   tracking:
     tracker: "bytetrack"
     max_age_frames: 30
     selection_strategy: "closest_to_center"  # or largest_bbox | highest_confidence | manual_id_lock
   prediction:
     mode: "reactive_pid"  # reactive_pid | kalman | lstm | transformer
     sequence_length: 15
     horizon_frames: 5
     lstm:
       hidden_size: 64
       num_layers: 2
       checkpoint_path: "data/models/lstm_predictor.pt"
     transformer:
       d_model: 64
       nhead: 4
       num_layers: 2
       checkpoint_path: "data/models/transformer_predictor.pt"
   control:
     pid:
       kp: 0.05
       ki: 0.0
       kd: 0.01
       output_limit: 15  # max degrees change per update
     deadband_px: 10
   servo:
     pan_min_deg: 0
     pan_max_deg: 180
     tilt_min_deg: 0
     tilt_max_deg: 180
     pan_center_deg: 90
     tilt_center_deg: 90
   hardware:
     enabled: false
     serial_port: "/dev/ttyUSB0"
     baud_rate: 115200
     mock_if_unavailable: true
   api:
     host: "0.0.0.0"
     port: 8000
   logging:
     level: "INFO"
     log_dir: "data/logs"
   ```
5. Write `.gitignore` (Python defaults + `data/recordings/`, `data/logs/`, `data/models/*.pt`, `.env`, `__pycache__/`, `*.egg-info`).
6. Write `utils/config.py`: a `load_config(path: str) -> dict` function using `PyYAML`, plus a `get(config, dotted_key, default=None)` helper for nested lookups like `get(cfg, "control.pid.kp")`.
7. Write a skeleton `README.md` with project title, one-paragraph description (copy from Section 0 of this plan), and a "Setup" section documenting `pip install -r requirements.txt`.
8. `git init` the repo if not already a repo, and make an initial commit.

**Acceptance Criteria:**
- `pip install -r requirements.txt` completes without errors in a fresh virtualenv.
- `python -c "from intellitrack.utils.config import load_config; print(load_config('configs/default.yaml'))"` runs and prints the parsed dict.
- `pytest` runs (even with zero tests collected) without import errors.

---

## Phase 1 — Camera Capture & YOLO Detection

**Goal:** Pull frames from a webcam and run object detection on them.

**Tasks:**
1. `capture/camera.py`: implement class `Camera`:
   - `__init__(self, index: int, width: int, height: int)`
   - `start(self) -> None` — opens `cv2.VideoCapture`, runs a background thread that continuously grabs frames into a thread-safe single-slot buffer (always keep only the latest frame, to avoid lag).
   - `read(self) -> tuple[bool, np.ndarray | None]` — returns latest frame.
   - `release(self) -> None`.
   - If the camera index cannot be opened, log an error and raise a clear `CameraUnavailableError` (custom exception in the same file) rather than a raw OpenCV error.
2. `detection/yolo_detector.py`:
   - Define `@dataclass Detection`: `bbox_xyxy: tuple[float,float,float,float]`, `confidence: float`, `class_id: int`, `class_name: str`.
   - Class `YoloDetector`:
     - `__init__(self, model_path: str, confidence_threshold: float, iou_threshold: float, target_classes: list[str])` — loads `ultralytics.YOLO(model_path)`.
     - `detect(self, frame: np.ndarray) -> list[Detection]` — runs inference, filters by confidence/IoU/target class names, returns list of `Detection`.
3. `scripts/run_live.py`: CLI script (`argparse`, accepts `--config`) that:
   - Loads config, starts `Camera`, loads `YoloDetector`.
   - Loop: read frame → detect → draw bounding boxes + labels with `cv2.rectangle`/`cv2.putText` → compute and overlay rolling FPS → `cv2.imshow` → exit on `q`.
4. `tests/test_detection.py`: use a static test image (agent should generate/save one simple synthetic image with a solid rectangle, or use any bundled sample image) and assert `YoloDetector.detect` returns a `list` without raising, and that filtering logic (mock a fake ultralytics result object) correctly drops detections below `confidence_threshold` and outside `target_classes`. Mock the underlying `ultralytics.YOLO` call so the test does not require downloading real weights.

**Acceptance Criteria:**
- `python scripts/run_live.py` opens a webcam window showing live bounding boxes and an FPS counter.
- `pytest tests/test_detection.py` passes.
- No hardcoded confidence/IoU/class values inside `yolo_detector.py` — all sourced from constructor args populated from config.

---

## Phase 2 — Multi-Object Tracking & Target Selection

**Goal:** Assign stable IDs across frames and pick one target to follow.

**Tasks:**
1. `tracking/byte_tracker_wrapper.py`:
   - Define `@dataclass TrackedObject`: `track_id: int`, `bbox_xyxy: tuple[float,float,float,float]`, `centroid: tuple[float,float]`, `class_name: str`, `confidence: float`, `age_frames: int`.
   - Class `ByteTrackerWrapper` wrapping Ultralytics' built-in tracking (`model.track(frame, tracker="bytetrack.yaml", persist=True)`) OR a standalone ByteTrack implementation if built-in tracking is insufficient — pick built-in first for MVP speed.
   - `update(self, frame: np.ndarray, detections: list[Detection]) -> list[TrackedObject]`.
2. `tracking/target_selector.py`:
   - Class `TargetSelector`:
     - `__init__(self, strategy: str, frame_width: int, frame_height: int, max_lost_frames: int)`.
     - `select(self, tracks: list[TrackedObject]) -> TrackedObject | None`.
     - Strategies to implement: `"largest_bbox"` (max bbox area), `"closest_to_center"` (min distance of centroid to frame center), `"highest_confidence"`, `"manual_id_lock"` (locks onto a specific `track_id` passed via a `lock_id(self, track_id: int)` method; falls back to another strategy if that ID disappears).
     - Target-loss handling: if the currently selected `track_id` is missing for more than `max_lost_frames` consecutive calls, clear the lock and reselect using the configured strategy on the next call.
3. Update `scripts/run_live.py` to run detection → tracking → selection, and draw the selected target's box in a distinct color (e.g. red) with its `track_id` and `class_name` labeled, all other tracked boxes in a neutral color.
4. `tests/test_tracking.py`: build synthetic lists of `TrackedObject` (no real video needed) and assert:
   - `"largest_bbox"` picks the largest area.
   - `"closest_to_center"` picks the nearest centroid.
   - Losing the locked target for more than `max_lost_frames` triggers reselection on the next call, and staying lost for fewer frames keeps the previous lock (returns `None` for that call rather than switching prematurely).

**Acceptance Criteria:**
- `python scripts/run_live.py` visibly tracks a single moving object with a consistent ID as it moves across frame.
- `pytest tests/test_tracking.py` passes for all four strategies.

---

## Phase 3 — Baseline Reactive Control: Kalman Filter + PID (simulated actuator)

**Goal:** Convert a tracked target's position into a control signal, in software only (no hardware yet).

**Tasks:**
1. `prediction/kalman.py`:
   - Class `ConstantVelocityKalman2D` implementing a standard constant-velocity 2D Kalman filter (state `[x, y, vx, vy]`).
   - `predict(self) -> tuple[float, float]` — advances state, returns predicted `(x, y)`.
   - `update(self, measurement: tuple[float, float]) -> None` — corrects state with a new centroid measurement.
   - Must work with `numpy` matrix math directly (do not require an external Kalman library) so behavior is transparent and citable in a research write-up.
2. `prediction/no_prediction.py`: trivial `NoPrediction` class with `predict_next(self, history: list[tuple[float,float]]) -> tuple[float,float]` that just returns the most recent centroid — this is the "reactive_pid" baseline's prediction stage (i.e., no prediction at all).
3. `control/pid_controller.py`:
   - Class `PIDController(kp, ki, kd, output_limit)`.
   - `compute(self, error: float, dt: float) -> float` — standard PID with integral clamping (anti-windup) and output clamped to `±output_limit`.
   - `reset(self) -> None`.
4. `control/servo_mapper.py`:
   - Function/class translating a pixel-space error `(dx, dy)` from frame center into target pan/tilt degree deltas, respecting `deadband_px` (ignore tiny errors) and `pan_min/max_deg`, `tilt_min/max_deg` clamping.
5. `pipeline/modes.py`: `class TrackingMode(str, Enum)` with values `REACTIVE_PID = "reactive_pid"`, `KALMAN = "kalman"`, `LSTM = "lstm"`, `TRANSFORMER = "transformer"`.
6. `pipeline/tracking_pipeline.py`:
   - Class `TrackingPipeline(config: dict)` that wires together `Camera → YoloDetector → ByteTrackerWrapper → TargetSelector → (prediction stage selected by TrackingMode) → PIDController → ServoMapper → (hardware output stage, stubbed as a log line for now)`.
   - `run_once(self) -> FrameMetrics` — processes exactly one frame and returns a `FrameMetrics` dataclass (`timestamp`, `mode`, `target_found: bool`, `raw_centroid`, `predicted_centroid`, `pan_deg`, `tilt_deg`, `latency_ms`, `fps_instant`).
   - `run_loop(self, max_frames: int | None = None)` — calls `run_once` repeatedly, optionally bounded.
   - For `REACTIVE_PID` mode, the prediction stage is `NoPrediction`; for `KALMAN`, it's `ConstantVelocityKalman2D.predict()`; `LSTM`/`TRANSFORMER` are wired in Phase 5/6.
7. `tests/test_kalman.py`: feed a synthetic linear trajectory (`x(t) = t, y(t) = 2t`) through several `update`/`predict` cycles and assert predicted position converges close to the true next point (within a tolerance) after a warm-up period.
8. `tests/test_pid.py`: run `PIDController.compute` repeatedly against a fixed nonzero error and assert the cumulative output trends toward reducing error in a simple closed-loop simulation (simulate: `position += pid.compute(target - position, dt)`), and assert output never exceeds `output_limit`.

**Acceptance Criteria:**
- `TrackingPipeline` runs end-to-end for both `reactive_pid` and `kalman` modes with a live camera, logging computed pan/tilt degrees to console every frame (no hardware needed yet).
- `pytest tests/test_kalman.py tests/test_pid.py` passes.

---

## Phase 4 — Arduino/ESP32 Serial Bridge & Real Hardware Control

**Goal:** Actually move physical pan-tilt servos, with a safe mock fallback.

**Tasks:**
1. `firmware/pan_tilt_controller/pan_tilt_controller.ino`:
   - Uses the `Servo` library, two servos on configurable pins (document pins in comments, e.g. pan = pin 9, tilt = pin 10).
   - Reads newline-terminated serial commands of the form `PAN:<int> TILT:<int>\n`.
   - Clamps incoming angles to `0–180`, writes to servos, and replies `ACK\n` over serial after each successful command.
   - On boot, centers both servos at 90 degrees and prints `READY\n`.
2. `control/serial_bridge.py`:
   - Class `SerialBridge(port: str, baud_rate: int, mock_if_unavailable: bool)`:
     - `connect(self) -> bool` — tries to open the serial port; on failure, if `mock_if_unavailable`, logs a warning and switches to internal mock mode (all sends are logged, not transmitted) rather than raising.
     - `send_angles(self, pan_deg: float, tilt_deg: float) -> None` — formats and writes the `PAN:.. TILT:..` command; non-blocking with a short read timeout waiting for `ACK`; logs (not raises) on timeout.
     - `close(self) -> None`.
   - Provide `MockSerialBridge` (or the same class's mock mode) that fully implements the same interface purely in-memory/log-based, so tests never need a real device.
3. Wire `SerialBridge`/`MockSerialBridge` into `pipeline/tracking_pipeline.py` as the final stage: when `config.hardware.enabled` is true, actually send angles; the mode is selected automatically at connect-time based on hardware availability.
4. `tests/test_serial_bridge.py`: instantiate the bridge pointed at a clearly invalid port (e.g. `"/dev/does-not-exist"`), assert `connect()` falls back to mock mode without raising, and assert `send_angles` logs the expected values (capture via `caplog` or an injectable log sink) rather than throwing.

**Acceptance Criteria:**
- With no Arduino connected, the full pipeline still runs without crashing and clearly logs "hardware not available — running in mock mode."
- With an Arduino connected and the firmware flashed, moving a physical object in front of the camera visibly pans/tilts the mount to keep it centered.
- `pytest tests/test_serial_bridge.py` passes.

---

## Phase 5 — Trajectory Prediction: LSTM (the core research differentiator)

**Goal:** Predict a target's future position instead of only reacting to its current position, and integrate it as a selectable tracking mode.

**Tasks:**
1. `scripts/record_dataset.py`: CLI script that runs the pipeline (camera + detection + tracking, no control needed) for a configurable duration/frame count and writes one CSV row per frame to `data/datasets/trajectories_<timestamp>.csv` with columns: `timestamp, track_id, centroid_x, centroid_y, frame_width, frame_height`. This is how training data for the predictor gets collected.
2. `prediction/lstm_predictor.py`:
   - `class TrajectoryDataset(torch.utils.data.Dataset)`: given a CSV (or list of per-track centroid sequences), builds `(sequence_length, horizon_frames)` input/target windows of normalized `(x, y, vx, vy)` features per track.
   - `class LSTMTrajectoryPredictor(nn.Module)`: an `nn.LSTM` followed by a linear head predicting `(x, y)` at `horizon_frames` ahead, with constructor args matching `configs/default.yaml`'s `prediction.lstm` section (`hidden_size`, `num_layers`).
   - Wrapper class `LSTMPredictor` exposing the same interface as `NoPrediction`/Kalman: `predict_next(self, history: list[tuple[float,float]]) -> tuple[float,float]`, internally maintaining a rolling window of the last `sequence_length` centroids and running the trained model; loads weights from `checkpoint_path` at construction, and if no checkpoint file exists, logs a warning and falls back to constant-velocity extrapolation (so the pipeline never crashes for lack of a trained model).
3. `prediction/train.py`: shared training loop function `train_predictor(model, dataset, epochs, lr, checkpoint_path)` usable by both LSTM and Transformer (Phase 6), using MSE loss between predicted and true future `(x, y)`, saving best checkpoint by validation loss.
4. `scripts/train_predictor.py`: CLI script, `--model {lstm,transformer} --dataset path/to.csv --config configs/default.yaml`, trains and saves to the checkpoint path from config.
5. Wire `LSTMPredictor` into `pipeline/tracking_pipeline.py` as the prediction stage when `TrackingMode.LSTM` is selected — its predicted future centroid, not the raw current centroid, becomes the input to the PID error calculation (this is the "predictive" difference from the reactive baseline).
6. `tests/test_prediction.py` (LSTM portion): construct a tiny synthetic dataset of a simple linear or sinusoidal trajectory, train for a small number of epochs, and assert training loss decreases from epoch 1 to the final epoch. Also assert `LSTMPredictor.predict_next` returns a tuple of two floats and does not raise when no checkpoint exists (fallback path).

**Acceptance Criteria:**
- `python scripts/record_dataset.py --duration 120` produces a usable CSV with a moving object in frame.
- `python scripts/train_predictor.py --model lstm --dataset <file>` trains and writes a checkpoint to `data/models/lstm_predictor.pt`.
- Running the live pipeline with `prediction.mode: lstm` uses the trained model's predicted position to drive PID/servo output, and this is visibly logged (predicted vs raw position both printed per frame for verification).
- `pytest tests/test_prediction.py` passes.

---

## Phase 6 — Trajectory Prediction: Transformer Variant

**Goal:** Add the second predictive model so the four-way comparison from the concept doc is complete.

**Tasks:**
1. `prediction/transformer_predictor.py`:
   - `class TransformerTrajectoryPredictor(nn.Module)`: a small Transformer encoder (`nn.TransformerEncoder`) operating on the same `(sequence_length, features)` windows as the LSTM, with a linear head to `(x, y)` at `horizon_frames` ahead. Constructor args from `configs/default.yaml`'s `prediction.transformer` section (`d_model`, `nhead`, `num_layers`).
   - Wrapper class `TransformerPredictor` with the identical `predict_next(history) -> (x, y)` interface as `LSTMPredictor`, same checkpoint-missing fallback behavior.
2. Extend `scripts/train_predictor.py` to support `--model transformer` using the same shared `prediction/train.py` training loop and the same recorded dataset format.
3. Wire `TransformerPredictor` into `pipeline/tracking_pipeline.py` for `TrackingMode.TRANSFORMER`.
4. Extend `tests/test_prediction.py` with the Transformer equivalent of the LSTM tests (loss decreases on tiny synthetic data; safe fallback with no checkpoint).

**Acceptance Criteria:**
- All four modes (`reactive_pid`, `kalman`, `lstm`, `transformer`) are selectable purely via `configs/default.yaml`'s `prediction.mode` key (or a `--mode` CLI override) with no code changes required to switch.
- `pytest tests/test_prediction.py` passes for both model types.

---

## Phase 7 — FastAPI Dashboard & Monitoring

**Goal:** A minimal web UI to watch the live tracking feed and adjust settings without restarting the process.

**Tasks:**
1. `api/schemas.py`: pydantic models `ConfigUpdateRequest` (subset of tunable fields: `prediction.mode`, PID gains, `tracking.selection_strategy`) and `MetricsResponse` (current FPS, latency, mode, target_found, pan_deg, tilt_deg).
2. `api/main.py`: creates the `FastAPI` app, instantiates a single shared `TrackingPipeline` running in a background thread (started on app startup, stopped cleanly on shutdown), includes the three routers below, and mounts `api/static/` for the simple HTML page.
3. `api/routes_stream.py`: `GET /stream` returns an MJPEG multipart response of the annotated frames (reuse the same drawing logic as `run_live.py`, refactored into a shared `visualize_frame()` utility so it isn't duplicated).
4. `api/routes_config.py`: `GET /config` returns the current effective config; `POST /config` accepts a `ConfigUpdateRequest` and applies it to the running pipeline in-memory (e.g., swapping `TrackingMode`, updating PID gains) without a restart.
5. `api/routes_metrics.py`: `GET /metrics` returns the most recent `MetricsResponse` computed from the pipeline's last `FrameMetrics`.
6. `api/static/index.html`: a single simple page (no framework needed) with an `<img src="/stream">` tag for the live feed, a small form for mode/PID changes posting to `/config` via `fetch`, and a metrics panel polling `/metrics` every second.
7. `tests/test_api.py`: use FastAPI's `TestClient` to hit `/config` (GET and POST) and `/metrics`, asserting correct status codes and schema-valid JSON responses. Mock/stub the pipeline so tests don't require a real camera.

**Acceptance Criteria:**
- Running `uvicorn intellitrack.api.main:app --reload` and visiting `http://localhost:8000` shows a live annotated video stream and a metrics panel.
- Changing `prediction.mode` through the web form visibly changes tracking behavior without restarting the server.
- `pytest tests/test_api.py` passes.

---

## Phase 8 — Metrics, Structured Logging & Experiment Harness

**Goal:** Make every run produce comparable, quantitative data.

**Tasks:**
1. `metrics/logger.py`: `class MetricsLogger` that appends one JSON-lines record per frame to `data/logs/<mode>_<timestamp>.jsonl` with fields: `timestamp, mode, target_found, raw_x, raw_y, predicted_x, predicted_y, pan_deg, tilt_deg, latency_ms, fps_instant`. Buffered writes, flushed periodically or on close.
2. `metrics/evaluator.py`: `evaluate_run(log_path: str) -> dict` computing, from a `.jsonl` log:
   - `tracking_accuracy` = fraction of frames where `target_found` is true.
   - `target_loss_events` = count of transitions from found→lost.
   - `mean_latency_ms`, `p95_latency_ms`.
   - `mean_fps`.
   - `prediction_rmse_px` = RMSE between `predicted_(x,y)` at frame `t` and the actual `raw_(x,y)` observed at frame `t + horizon_frames` (only meaningful for `lstm`/`transformer` modes; return `None` for the others).
   - `tracking_stability` = standard deviation of frame-to-frame pixel error (distance between target centroid and frame center).
3. Wire `MetricsLogger` into `pipeline/tracking_pipeline.py` so every `run_once()` call also logs a record when logging is enabled.
4. `scripts/run_experiment.py`: CLI (`--mode`, `--duration-seconds` or `--max-frames`, `--source {live,recorded}`, `--video-path` if recorded) that runs `TrackingPipeline` for a bounded run, writes the `.jsonl` log, then immediately calls `evaluate_run` and writes a summary `data/logs/<mode>_<timestamp>_summary.json`.

**Acceptance Criteria:**
- `python scripts/run_experiment.py --mode kalman --duration-seconds 60` produces both a `.jsonl` per-frame log and a `_summary.json` with all metric fields populated (non-`None` where applicable).
- Re-running with `--mode lstm` populates `prediction_rmse_px` with a real number.

---

## Phase 9 — Four-Way Experimental Comparison & Report

**Goal:** Directly answer the project's research question with a reproducible report.

**Tasks:**
1. Record one or more short reference videos (people/objects moving through frame) and save under `data/recordings/`, so all four modes can be evaluated against identical input (fairness across the comparison) — extend `TrackingPipeline`/`Camera` to optionally read from a video file path instead of a live camera (add a `source: {live, file}` and `file_path` key to `configs/default.yaml`'s `camera` section).
2. Run `scripts/run_experiment.py` once per mode (`reactive_pid`, `kalman`, `lstm`, `transformer`) against the same recorded video(s), producing four summary JSON files.
3. `scripts/generate_report.py`: loads all four summary JSON files, and produces `data/logs/comparison_report.md` containing:
   - A markdown table comparing all four modes across every metric from Phase 8 (tracking accuracy, target loss rate, mean/p95 latency, mean FPS, prediction RMSE, tracking stability), plus CPU utilization sampled during the run via `psutil` (add `psutil` to `requirements.txt` and sample it inside `run_experiment.py`).
   - Matplotlib-generated plots saved as PNGs and embedded via markdown image links: (a) per-frame pixel tracking error over time, one line per mode, on one chart; (b) bar chart of mean latency per mode; (c) bar chart of mean FPS per mode.
   - A short "Interpretation" section auto-filled with which mode had the lowest tracking error and lowest target loss rate, generated programmatically from the computed numbers (not hardcoded text).

**Acceptance Criteria:**
- `python scripts/generate_report.py` produces `data/logs/comparison_report.md` with a complete table (no missing/`None` cells for metrics that apply to that mode) and three embedded PNG plots.
- The report is directly usable as a results section draft for the undergraduate research paper described in the project concept.

---

## 4. Definition of Done for the MVP

The MVP is complete when all of the following hold simultaneously:
- All nine phases above pass their individual acceptance criteria.
- `pytest` passes with zero failures across the whole `tests/` directory.
- The pipeline runs end-to-end with zero hardware attached (full mock mode) and, separately, with an Arduino/ESP32 and pan-tilt mount attached (real servo motion).
- All four tracking modes are selectable via config/CLI with no code edits.
- The FastAPI dashboard shows a live stream and lets you switch modes at runtime.
- `scripts/generate_report.py` produces a complete four-way comparison report from real recorded data.
- `README.md` is updated with: setup instructions, how to flash the Arduino firmware, how to run live tracking, how to record a dataset and train a predictor, how to run the full experiment suite, and where to find the final report.

## 5. Known Risks & Mitigations

- **No Arduino/ESP32 available during development:** mitigated by `SerialBridge`'s mock mode (Phase 4) — every phase after Phase 4 must remain fully testable without hardware.
- **Not enough recorded trajectory data to train LSTM/Transformer well:** mitigated by the checkpoint-missing fallback to constant-velocity extrapolation (Phase 5/6), so the pipeline is never blocked on model quality; note in the report if predictive modes underperform due to limited training data rather than treating it as a pipeline bug.
- **YOLO inference too slow on CPU for real-time FPS:** default to the smallest model variant (`yolo11n.pt`) and document in README how to switch to a larger model if a GPU is available.
- **Servo jitter from PID overshoot:** mitigated by `deadband_px` and `output_limit` in the PID/servo-mapper config; tune empirically once hardware is attached.

## 6. How to Hand This Plan to an Agent

Give the agent one phase at a time (paste that phase's section plus Sections 2 and 3 for context). After the agent reports a phase's acceptance criteria as met, review the diff, run the tests yourself, and only then provide the next phase's section. Do not paste the entire document as one giant task — phased handoff keeps each change reviewable and keeps the agent from making cross-cutting assumptions before earlier modules are verified.
