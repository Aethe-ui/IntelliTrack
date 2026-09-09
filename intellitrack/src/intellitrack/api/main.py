"""FastAPI application entrypoint for the IntelliTrack dashboard."""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from intellitrack.api import routes_config, routes_metrics, routes_stream
from intellitrack.pipeline.tracking_pipeline import TrackingPipeline
from intellitrack.utils.config import get, load_config

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"
_DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"


def _pipeline_loop(pipeline: TrackingPipeline, stop_event: threading.Event) -> None:
    """Background worker that repeatedly calls ``run_once`` until stopped."""
    try:
        pipeline.start()
    except Exception as exc:  # noqa: BLE001
        logger.error("Pipeline failed to start in API worker: %s", exc)
        return
    while not stop_event.is_set():
        try:
            pipeline.run_once()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline run_once error: %s", exc)
            stop_event.wait(0.1)
    pipeline.stop()


def create_app(
    config_path: Optional[str] = None,
    pipeline: Optional[TrackingPipeline] = None,
) -> FastAPI:
    """Build the FastAPI app with a shared TrackingPipeline.

    Args:
        config_path: Optional path to YAML config (used when ``pipeline`` is None).
        pipeline: Optional pre-built pipeline (tests inject a stub/mock here).

    Returns:
        Configured :class:`FastAPI` application.
    """
    stop_event = threading.Event()
    worker: Optional[threading.Thread] = None
    owns_pipeline = pipeline is None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal worker, pipeline
        if pipeline is None:
            cfg_path = config_path or str(_DEFAULT_CONFIG)
            config = load_config(cfg_path)
            logging.basicConfig(level=getattr(logging, get(config, "logging.level", "INFO")))
            pipeline = TrackingPipeline(config)
        app.state.pipeline = pipeline
        app.state.stop_event = stop_event

        if owns_pipeline:
            worker = threading.Thread(
                target=_pipeline_loop,
                args=(pipeline, stop_event),
                daemon=True,
                name="pipeline-worker",
            )
            worker.start()
            logger.info("Pipeline background thread started.")

        yield

        stop_event.set()
        if worker is not None:
            worker.join(timeout=5.0)
        logger.info("API shutdown complete.")

    app = FastAPI(title="IntelliTrack", lifespan=lifespan)
    app.include_router(routes_stream.router)
    app.include_router(routes_config.router)
    app.include_router(routes_metrics.router)

    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    return app


# Default ASGI app for ``uvicorn intellitrack.api.main:app``
app = create_app()
