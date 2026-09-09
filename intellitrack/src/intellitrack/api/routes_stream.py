"""MJPEG live stream route."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Generator

import cv2
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

if TYPE_CHECKING:
    from intellitrack.pipeline.tracking_pipeline import TrackingPipeline

logger = logging.getLogger(__name__)

router = APIRouter()


def _mjpeg_generator(pipeline: "TrackingPipeline") -> Generator[bytes, None, None]:
    """Yield multipart JPEG frames from the pipeline's last annotated frame."""
    boundary = b"--frame"
    while True:
        frame = pipeline.last_annotated_frame
        if frame is None:
            time.sleep(0.05)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            time.sleep(0.01)
            continue
        jpg = buf.tobytes()
        yield (
            boundary
            + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
            + str(len(jpg)).encode()
            + b"\r\n\r\n"
            + jpg
            + b"\r\n"
        )
        time.sleep(0.03)


@router.get("/stream")
def stream(request: Request) -> StreamingResponse:
    """Return an MJPEG multipart response of annotated frames."""
    pipeline: "TrackingPipeline" = request.app.state.pipeline
    return StreamingResponse(
        _mjpeg_generator(pipeline),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
