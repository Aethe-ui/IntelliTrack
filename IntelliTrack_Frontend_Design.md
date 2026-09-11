# IntelliTrack Frontend — Design & Feature Specification

Companion file: `intellitrack_dashboard.html` (self-contained HTML/CSS/JS mockup, no build step, no external dependencies besides two Google Fonts). Open it directly in a browser to review the design. It is a **visual and interaction mockup with simulated data** — every value moves and updates live, but the numbers come from client-side JS, not from a running pipeline. Section 6 below lists exactly what to swap for real data when this is wired into Phase 7.

This spec extends, and stays consistent with, Phase 7 (FastAPI Dashboard) and Phase 8–9 (metrics/experiment reporting) of the MVP build plan. It does not change anything about Phases 0–6.

---

## 1. Design rationale

**Concept:** a camera operator's control console, not a generic admin dashboard. The layout borrows from rangefinder/gimbal instrumentation — a viewfinder with corner brackets and a center reticle, semicircular servo gauges, fader-style sliders for PID gains — because the subject matter (an actual gimbal being driven by an actual control loop) supports it directly, not as decoration.

**Color carries meaning, not branding.** The four tracking modes form a deliberate gradient from neutral to warm, matching the project's own research narrative of "increasing predictive sophistication":

| Mode | Hex | Rationale |
|---|---|---|
| Reactive PID | `#7a8590` (steel grey) | No prediction — the neutral baseline |
| Kalman filter | `#4fd8e0` (cyan) | First-order motion estimate |
| LSTM | `#f2a93c` (amber) | Learned, data-driven prediction |
| Transformer | `#e8632f` (deep orange) | Most sophisticated / attention-based prediction |

This gradient is reused everywhere a mode needs identifying — mode buttons, dial needles, chart lines, comparison-table swatches — so a color always means the same thing across the whole page. Status/system colors (`--green #5fbf82` = found/ok, `--red #e8604a` = lost/error) are kept separate from the mode gradient so the two meanings never collide.

**Typography:** IBM Plex Sans for all UI chrome, labels, and buttons; IBM Plex Mono for every actual telemetry number (FPS, latency, angles, coordinates, table figures). This is a functional split — human-language labels versus machine-read numbers — not a decorative one. Labels are sentence case throughout; nothing is set in tracked-out uppercase.

**One motion moment:** switching tracking mode triggers a short blur-to-focus animation on the viewfinder (`.refocus`, 0.5s), evoking a camera racking focus. Everything else (hover states, slider drags) is a plain, fast transition. `prefers-reduced-motion` disables the refocus animation.

**Layout:** left-weighted asymmetric grid — a large viewfinder panel (≈61% width) paired with a telemetry rail on desktop, stacking to a single column under 900px. Panels are flat, hairline-bordered, 6px corner radius — closer to instrument housings than SaaS cards, with no drop shadows.

---

## 2. Page structure (DOM map)

```
header.top
  .brand                      → title + mark
  .status-row                 → pipeline / hardware / clock pills

.dashboard-grid
  .viewfinder-wrap            → live feed panel
    canvas#scope
    .stage-readout (x2)       → raw / predicted coords, target pill, track id
  .rail
    "Tracking mode" panel     → #modeGrid (4 buttons)
    "Telemetry" panel         → #fpsVal #latVal #panVal #tiltVal
    "Servo position" panel    → canvas#dialPan, canvas#dialTilt

.control-strip (panel)
  .control-grid                → 5 faders: #kp #ki #kd #ol #db
  .strategy-row                → #strategy select, #lockId, #lockBtn
  .response-chart              → canvas#stepChart
  details.config                → #yamlView (GET /config preview)

.comparison (panel)
  table#compareTable           → four-way metrics table
  #chartsRow                   → 4 sparkline cards
  #interpBox                   → auto-generated interpretation sentence

.log-panel (panel)
  #logBox                      → scrolling event console

footer
```

---

## 3. Feature catalog

### 3.1 Header / status bar
- **Brand mark**: a reticle glyph (circle + crosshair) whose accent color follows the active tracking mode — a small always-visible reminder of current mode even when scrolled past the mode selector.
- **Pipeline status pill**: green LED + "Pipeline running". Maps to whether the background `TrackingPipeline` thread (Phase 7, `api/main.py`) is alive.
- **Hardware status pill**: LED + text, e.g. "Hardware: mock serial" or "Hardware: connected (COM4)". Maps to `SerialBridge`/`MockSerialBridge` connection state (Phase 4). Amber LED = mock mode, green = real device connected, red = enabled but unreachable.
- **Clock**: local wall clock, decorative/orienting only, not tied to backend.

### 3.2 Viewfinder (`GET /stream`)
- Renders the annotated video feed. In the mockup, a canvas draws a synthetic moving target so the whole interaction model can be reviewed without a camera; in production this panel is the same MJPEG `<img src="/stream">` element specified in Phase 7, with the following **overlay elements added on top of it**, all driven by `/metrics` polling rather than baked into the MJPEG frame itself (keeps the video stream itself simple, per Phase 7's `visualize_frame()` responsibility):
  - Frame-center reticle (static, decorative — always at canvas center).
  - Corner brackets (static, decorative — framing device only).
  - **Raw detection box**: solid white/light box drawn around the current `raw_centroid`, labeled with class name and, in `manual_id_lock`, `"id {track_id} · locked"`.
  - **Predicted ghost box**: dashed box in the active mode's color, offset ahead of the raw box along the target's current velocity vector, shown only when the mode has a nonzero prediction horizon (Kalman, LSTM, Transformer — not Reactive PID). This is the single clearest visual expression of the project's core research differentiator, so it should be kept even if other overlay elements are simplified later.
  - **Coordinate readouts** (bottom-left, monospace): raw `(x, y)` and predicted `(x, y)`, updated every frame.
  - **Target-found pill** (top-right): green "target found" / red "target lost", swaps instantly on state change.
  - **Track ID readout** (top-right): current selected `track_id`.
  - **Lost-target frame**: when `target_found` is false, a dashed red border appears around the whole viewfinder as a hard-to-miss state change.
- **Mode chip** (panel header, top-right of this panel): shows the active mode's display name in its accent color, redundant with the mode selector rail but useful since the viewfinder is the largest, most-looked-at element on the page.

### 3.3 Tracking mode selector (`POST /config`, field `prediction.mode`)
- Four buttons, one per `TrackingMode` enum value, each showing: colored dot, mode name, one-line plain-language description of what that mode does (not the class name — see writing guidance in Section 1).
- Exactly one active at a time (`.mode-btn.active`), highlighted with the mode's accent color as both border and text color.
- Clicking a button is a complete action, not a staged draft: it should fire `POST /config` immediately with `{"prediction": {"mode": "<id>"}}` and only mark the button active once the response confirms — the mockup marks it active optimistically since there's no real backend, but production should reconcile from the actual server state so the UI never claims a mode is active that the pipeline rejected.
- Triggers the viewfinder refocus animation and updates every accent-colored element on the page (mode chip, dial needles, chart line, step-response line color) in one action, reinforcing that this is a single global switch, not a per-panel setting.

### 3.4 Telemetry panel (`GET /metrics`, polled — Phase 7 spec says the static page polls this every second; the mockup animates every frame purely for visual liveliness)
Four fields, each large monospace figure with a small unit and a muted label above it:
- **FPS, instantaneous** — from `MetricsResponse.fps` / `FrameMetrics.fps_instant`.
- **Latency** — from `MetricsResponse.latency_ms` / `FrameMetrics.latency_ms`, in ms.
- **Pan angle** — from `FrameMetrics.pan_deg`, in degrees.
- **Tilt angle** — from `FrameMetrics.tilt_deg`, in degrees.

This is deliberately a small, fixed set matching exactly what `MetricsResponse` already specifies in Phase 7 — no new backend fields required for this panel.

### 3.5 Servo position dials
- Two semicircular gauges (0–180°), one per axis, with a needle and a filled arc in the active mode's color.
- Purely a redundant, more legible visual encoding of the same `pan_deg` / `tilt_deg` numbers already shown in 3.4 — there for at-a-glance monitoring from across a room during a live demo, where reading a number is slower than reading a needle position.
- No new data required; same two fields as 3.4.

### 3.6 PID & selection control strip (`POST /config`)
- **Five sliders** bound 1:1 to existing `configs/default.yaml` keys, with the live numeric value shown beside each label as it's dragged:
  - `control.pid.kp` (0–0.3, step 0.005)
  - `control.pid.ki` (0–0.05, step 0.001)
  - `control.pid.kd` (0–0.08, step 0.001)
  - `control.pid.output_limit` (2–40°)
  - `control.deadband_px` (0–40 px)
- **Selection strategy dropdown**, the four values already defined in `TargetSelector` (Phase 2): `closest_to_center`, `largest_bbox`, `highest_confidence`, `manual_id_lock`.
- **Track-ID lock field + button**, shown only when `manual_id_lock` is selected. Calls `TargetSelector.lock_id(track_id)` conceptually; the button toggles between "Lock target" and "Locked ✓" and should reflect whether the lock actually succeeded (i.e., that `track_id` currently exists) rather than just the button's own click state — the mockup simplifies this to a pure UI toggle.
- **Behavior note for wiring**: sliders should debounce before firing `POST /config` (e.g., on `change`, not every `input` event, or debounced ~150–250ms) so dragging a slider doesn't flood the API with requests; the mockup updates its local preview on every `input` event since there's no network call to worry about.

### 3.7 Simulated step response chart
- A small line chart recomputed live as the PID sliders move, showing a simplified damped step response (a toy mass-spring-damper driven by the same Kp/Ki/Kd/output-limit values) so a person can feel the effect of a gain change — overshoot, oscillation, settling time — before touching real hardware.
- **This is explicitly illustrative**, not derived from the real camera/servo/latency chain, and is labeled as such in the UI caption. It needs no backend endpoint; it can remain a pure client-side convenience feature computed from whatever gain values are currently in the form fields, whether or not those have been submitted yet.

### 3.8 Effective config viewer (`GET /config`)
- A collapsed-by-default `<details>` panel that, when opened, shows the current effective config as read-only YAML-styled text: `prediction.mode`, the PID block, `deadband_px`, `selection_strategy`, and the `hardware.enabled` / `mock_if_unavailable` flags.
- Mirrors what `GET /config` already returns per the Phase 7 spec; this is a read-only debugging/verification view, not an editor — all edits happen through the dedicated controls in 3.3 and 3.6, not by typing into this panel.

### 3.9 Four-way comparison section (Phase 8–9 output)
- **Table**: one row per mode, columns for every metric `metrics/evaluator.py`'s `evaluate_run()` already computes — tracking accuracy, target loss events, mean latency, p95 latency, mean FPS, prediction RMSE (shown as `—` for modes where it's `None`, i.e. Reactive PID and Kalman), tracking stability, plus CPU utilization (sampled via `psutil` per the Phase 9 report spec). The best value in each column that has a clear "better" direction is highlighted.
- **Four sparkline cards**: one small per-frame-error-over-time trace per mode, colored with the same mode gradient, mirroring plot (a) from Phase 9's `generate_report.py` spec in a compact form for the live dashboard.
- **Auto-generated interpretation line**: one sentence naming the mode with the highest tracking accuracy and the mode with the fewest target-loss events, computed from the same numbers in the table — this is meant to read exactly like the "Interpretation" section `generate_report.py` already produces, just rendered inline instead of into the markdown report.
- **This entire section reads from the four `data/logs/<mode>_<timestamp>_summary.json` files**, not from the live pipeline — it's a report viewer embedded in the dashboard, not a live telemetry panel. The mockup fills it with clearly-labeled sample numbers and says so directly under the table; that disclaimer should stay in some form (e.g. "no experiment data yet — run scripts/run_experiment.py for all four modes") until real summary files exist, rather than ever showing fabricated numbers as if they were real results.

### 3.10 Event log console
- A scrolling, monospace, timestamped log — info in muted grey, warnings (e.g. "target lost") in amber, errors in red.
- Conceptually a live tail of the structured per-frame log Phase 8's `MetricsLogger` writes, or of the standard Python `logging` output, filtered to human-relevant events (mode changes, lock/unlock, target lost/reacquired) rather than every single frame — a raw per-frame tail at 20–30 lines/sec would be unreadable.
- Capped at 60 visible lines client-side (oldest dropped) to keep the DOM small during long sessions.

### 3.11 Footer
- Repo link and a one-line note that this is a console mockup pending Phase 7 wiring. Purely informational, no bound data.

---

## 4. Data contract additions for Phase 7

The existing `api/schemas.py` plan (Section 7 of the build doc) already covers `ConfigUpdateRequest` and `MetricsResponse`. To support everything in Section 3 above, `MetricsResponse` needs a few fields it doesn't yet list:

```python
class MetricsResponse(BaseModel):
    timestamp: float
    mode: str
    target_found: bool
    track_id: int | None
    raw_x: float | None
    raw_y: float | None
    predicted_x: float | None
    predicted_y: float | None
    pan_deg: float
    tilt_deg: float
    latency_ms: float
    fps_instant: float
```

This matches `FrameMetrics` (Phase 3) plus `track_id`, `raw_x/y`, and `predicted_x/y`, which the viewfinder overlay (3.2) and coordinate readouts need but the original `FrameMetrics` dataclass does not explicitly list as separate fields — worth confirming against the actual Phase 3 implementation and extending `FrameMetrics` itself if those fields aren't already broken out.

Two additions beyond what Phase 7 originally scoped, both optional and additive — nothing here should block or change Phases 0–6:

- **`GET /experiments/summary`**: reads the four most recent `data/logs/*_summary.json` files (one per mode) and returns them as a single JSON object keyed by mode, so Section 3.9 can be a real live view instead of a static mockup. Naturally implemented alongside `metrics/evaluator.py` in Phase 8, or as a small addition to `api/routes_metrics.py`.
- **Log tail for Section 3.10**: either a lightweight `GET /logs/recent?limit=60` polling endpoint, or a WebSocket if one is already planned elsewhere — a polling endpoint is the simpler match for the rest of Phase 7's polling-based design and avoids introducing a new transport just for this panel.

No changes are needed to `routes_stream.py` — the MJPEG stream stays exactly as specified; all overlay elements in 3.2 are drawn by the frontend from `/metrics`, not burned into the video frames server-side.

---

## 5. Responsive & accessibility notes

- Single breakpoint at 900px: the two-column dashboard grid and the five-column control grid both collapse to fewer columns; everything else already stacks naturally.
- All interactive elements (mode buttons, sliders, select, lock button, details/summary) are native HTML controls, so keyboard tabbing and screen readers work without extra ARIA — visible focus ring uses the active mode's accent color.
- `prefers-reduced-motion` disables the refocus animation and would similarly disable any future added transitions.
- No color-only signaling: the target-found/lost state pairs color with both an LED dot and text ("target found" / "target lost"), not color alone.

---

## 6. What's simulated in the mockup vs what needs real wiring

| Element | Mockup source | Real source once wired |
|---|---|---|
| Video feed | Canvas-drawn synthetic target | `GET /stream` MJPEG |
| Raw / predicted centroid | Computed from a sine-wave path | `FrameMetrics.raw_*` / `predicted_*` via `/metrics` |
| FPS, latency | Randomized around a per-mode baseline | `FrameMetrics.fps_instant` / `latency_ms` |
| Pan / tilt angles | A real PID loop running client-side against the synthetic target | `FrameMetrics.pan_deg` / `tilt_deg` from the actual pipeline |
| Target found / lost | Randomly toggled every ~10–15s | `FrameMetrics.target_found` |
| Mode switch | Local state only | `POST /config` round-trip |
| PID/selection sliders | Local state only, drives the local simulated PID loop | `POST /config` round-trip, debounced |
| Effective config viewer | Mirrors local slider/mode state | `GET /config` response |
| Step-response chart | Simplified client-side physics toy | Stays client-side by design (see 3.7) — not meant to become real |
| Four-way comparison table & sparklines | Hand-written sample numbers | Four `data/logs/*_summary.json` files, ideally via the new `/experiments/summary` route |
| Event log | Synthetic messages on synthetic events | Tail of `MetricsLogger` output or Python `logging`, filtered to notable events |

---

## 7. Suggested file placement

- `intellitrack_dashboard.html` → becomes `src/intellitrack/api/static/index.html` in Phase 7, once `visualize_frame()` and the metrics/config routes exist to feed it. The file is fully self-contained (inline CSS/JS, two Google Fonts links, no build step), matching Phase 7's "no framework needed" instruction.
- This spec (`IntelliTrack_Frontend_Design.md`) is meant to sit alongside the build plan — e.g. `docs/frontend_design.md` — as the Phase 7 reference an agent would read before implementing `routes_stream.py`, `routes_config.py`, and `routes_metrics.py`, in the same way the phase document already links features to acceptance criteria.
