"""Live metrics route."""

from __future__ import annotations

from fastapi import APIRouter, Request

from intellitrack.api.schemas import MetricsResponse

router = APIRouter()


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(request: Request) -> MetricsResponse:
    """Return the most recent FrameMetrics as a MetricsResponse."""
    pipeline = request.app.state.pipeline
    m = pipeline.last_metrics
    if m is None:
        return MetricsResponse(
            fps=0.0,
            latency_ms=0.0,
            mode=pipeline.mode.value,
            target_found=False,
            pan_deg=90.0,
            tilt_deg=90.0,
        )
    return MetricsResponse(
        fps=m.fps_instant,
        latency_ms=m.latency_ms,
        mode=m.mode.value,
        target_found=m.target_found,
        pan_deg=m.pan_deg,
        tilt_deg=m.tilt_deg,
    )
