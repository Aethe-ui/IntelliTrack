"""Pydantic request/response schemas for the IntelliTrack API."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ConfigUpdateRequest(BaseModel):
    """Subset of tunable fields that can be changed at runtime."""

    prediction_mode: Optional[str] = Field(
        default=None,
        description="One of: reactive_pid, kalman, lstm, transformer",
        alias="prediction.mode",
    )
    pid_kp: Optional[float] = Field(default=None, alias="control.pid.kp")
    pid_ki: Optional[float] = Field(default=None, alias="control.pid.ki")
    pid_kd: Optional[float] = Field(default=None, alias="control.pid.kd")
    pid_output_limit: Optional[float] = Field(
        default=None, alias="control.pid.output_limit"
    )
    selection_strategy: Optional[str] = Field(
        default=None, alias="tracking.selection_strategy"
    )
    target_class: Optional[str] = Field(
        default=None,
        description="YOLO class name to track, or 'all' to disable class filtering",
        alias="detection.target_class",
    )

    model_config = {"populate_by_name": True}


class MetricsResponse(BaseModel):
    """Latest pipeline metrics snapshot."""

    fps: float
    latency_ms: float
    mode: str
    target_found: bool
    pan_deg: float
    tilt_deg: float
    live_stats: Dict[str, Any] = Field(default_factory=dict)
