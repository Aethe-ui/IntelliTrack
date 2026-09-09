"""API route tests with a stubbed TrackingPipeline (no camera required)."""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from intellitrack.api.main import create_app
from intellitrack.api.schemas import MetricsResponse
from intellitrack.pipeline.modes import TrackingMode
from intellitrack.pipeline.tracking_pipeline import FrameMetrics


class StubPipeline:
    """Minimal stand-in for TrackingPipeline used by API tests."""

    def __init__(self) -> None:
        self._mode = TrackingMode.REACTIVE_PID
        self._config: Dict[str, Any] = {
            "prediction": {"mode": "reactive_pid"},
            "control": {"pid": {"kp": 0.05, "ki": 0.0, "kd": 0.01, "output_limit": 15}},
            "tracking": {"selection_strategy": "closest_to_center"},
        }
        self._last = FrameMetrics(
            timestamp=0.0,
            mode=self._mode,
            target_found=True,
            raw_centroid=(320.0, 240.0),
            predicted_centroid=(325.0, 240.0),
            pan_deg=92.0,
            tilt_deg=88.0,
            latency_ms=12.5,
            fps_instant=30.0,
        )

    @property
    def mode(self) -> TrackingMode:
        return self._mode

    @property
    def config(self) -> dict:
        return self._config

    @property
    def last_metrics(self) -> Optional[FrameMetrics]:
        return self._last

    @property
    def last_annotated_frame(self):
        return None

    def set_mode(self, mode: TrackingMode) -> None:
        self._mode = mode
        self._config["prediction"]["mode"] = mode.value
        self._last.mode = mode

    def update_pid(self, **kwargs) -> None:
        pid = self._config["control"]["pid"]
        for k, v in kwargs.items():
            if v is not None and k in pid:
                pid[k] = v

    def set_selection_strategy(self, strategy: str) -> None:
        self._config["tracking"]["selection_strategy"] = strategy


@pytest.fixture()
def client() -> TestClient:
    stub = StubPipeline()
    app = create_app(pipeline=stub)  # type: ignore[arg-type]
    with TestClient(app) as c:
        yield c


def test_get_config(client: TestClient) -> None:
    res = client.get("/config")
    assert res.status_code == 200
    body = res.json()
    assert "prediction" in body
    assert body["prediction"]["mode"] == "reactive_pid"


def test_post_config_changes_mode(client: TestClient) -> None:
    res = client.post(
        "/config",
        json={"prediction.mode": "kalman", "control.pid.kp": 0.1},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["config"]["prediction"]["mode"] == "kalman"
    assert body["config"]["control"]["pid"]["kp"] == 0.1


def test_get_metrics_schema(client: TestClient) -> None:
    res = client.get("/metrics")
    assert res.status_code == 200
    data = res.json()
    parsed = MetricsResponse.model_validate(data)
    assert parsed.target_found is True
    assert parsed.fps == 30.0
    assert parsed.mode in {m.value for m in TrackingMode}
