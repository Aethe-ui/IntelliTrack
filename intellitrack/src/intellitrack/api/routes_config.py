"""Runtime configuration GET/POST routes."""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Request

from intellitrack.api.schemas import ConfigUpdateRequest
from intellitrack.pipeline.modes import TrackingMode

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/config")
def get_config(request: Request) -> Dict[str, Any]:
    """Return the current effective pipeline configuration."""
    pipeline = request.app.state.pipeline
    return pipeline.config


@router.post("/config")
def post_config(body: ConfigUpdateRequest, request: Request) -> Dict[str, Any]:
    """Apply a subset of config changes to the running pipeline in-memory."""
    pipeline = request.app.state.pipeline
    cfg = pipeline.config

    if body.prediction_mode is not None:
        mode = TrackingMode(body.prediction_mode)
        pipeline.set_mode(mode)
        cfg.setdefault("prediction", {})["mode"] = mode.value

    pid_kwargs = {}
    if body.pid_kp is not None:
        pid_kwargs["kp"] = body.pid_kp
        cfg.setdefault("control", {}).setdefault("pid", {})["kp"] = body.pid_kp
    if body.pid_ki is not None:
        pid_kwargs["ki"] = body.pid_ki
        cfg.setdefault("control", {}).setdefault("pid", {})["ki"] = body.pid_ki
    if body.pid_kd is not None:
        pid_kwargs["kd"] = body.pid_kd
        cfg.setdefault("control", {}).setdefault("pid", {})["kd"] = body.pid_kd
    if body.pid_output_limit is not None:
        pid_kwargs["output_limit"] = body.pid_output_limit
        cfg.setdefault("control", {}).setdefault("pid", {})["output_limit"] = (
            body.pid_output_limit
        )
    if pid_kwargs:
        pipeline.update_pid(**pid_kwargs)

    if body.selection_strategy is not None:
        pipeline.set_selection_strategy(body.selection_strategy)

    logger.info("Config updated via API: %s", body.model_dump(by_alias=True, exclude_none=True))
    return {"ok": True, "config": cfg}
