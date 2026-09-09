"""Camera capture module.

Provides the :class:`Camera` class which wraps OpenCV's ``VideoCapture`` and
continuously grabs frames in a background thread so that the pipeline always
consumes the *latest* available frame without accumulating lag.

Supports both live device indices and recorded video files (Phase 9).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraUnavailableError(RuntimeError):
    """Raised when a camera device or video file cannot be opened."""


class Camera:
    """Thread-safe, single-slot frame buffer camera wrapper.

    The internal reader thread continuously grabs frames from
    ``cv2.VideoCapture`` and stores only the most recent one.  This prevents
    frame-queue build-up when downstream processing is slower than the camera
    frame rate.

    For file sources, frames are delivered sequentially (no skipping) and
    :attr:`exhausted` becomes ``True`` when the file ends.

    Args:
        index: ``cv2.VideoCapture`` device index, or a video file path.
        width: Requested capture width in pixels.
        height: Requested capture height in pixels.
        is_file: If ``True``, treat ``index`` as a file path and do not loop.
    """

    def __init__(
        self,
        index: Union[int, str],
        width: int,
        height: int,
        is_file: bool = False,
    ) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._is_file = is_file

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._exhausted = False
        self._new_frame_event = threading.Event()

    @classmethod
    def from_file(cls, path: str, width: int, height: int) -> "Camera":
        """Construct a :class:`Camera` that reads frames from a video file.

        Args:
            path: Filesystem path to a video file.
            width: Requested frame width (may be ignored by the decoder).
            height: Requested frame height.

        Returns:
            A file-backed :class:`Camera` instance.
        """
        return cls(index=path, width=width, height=height, is_file=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the capture device/file and start the background grabber thread.

        Raises:
            CameraUnavailableError: If the device/file cannot be opened.
        """
        cap = cv2.VideoCapture(self._index)
        if not cap.isOpened():
            cap.release()
            raise CameraUnavailableError(
                f"Cannot open camera/source at {self._index!r}. "
                "Check that the device is connected or the file path is correct."
            )

        if not self._is_file:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        self._cap = cap
        self._running = True
        self._exhausted = False

        self._thread = threading.Thread(
            target=self._grab_loop, daemon=True, name="camera-grabber"
        )
        self._thread.start()
        logger.info(
            "Camera %r started (requested %dx%d, file=%s)",
            self._index,
            self._width,
            self._height,
            self._is_file,
        )

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the most recently captured frame.

        For file sources, waits briefly for a new frame so sequential playback
        does not skip content when the consumer is faster than the grabber.

        Returns:
            A ``(success, frame)`` tuple.  ``success`` is ``False`` and
            ``frame`` is ``None`` if no frame is available (or the file ended).
        """
        if self._is_file:
            # Wait for a fresh frame or EOF
            self._new_frame_event.wait(timeout=0.5)
            self._new_frame_event.clear()
            with self._lock:
                if self._exhausted and self._frame is None:
                    return False, None
                if self._frame is None:
                    return False, None
                frame = self._frame
                self._frame = None  # consume so we don't re-emit
                return True, frame.copy()

        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def release(self) -> None:
        """Stop the grabber thread and release the underlying capture device."""
        self._running = False
        self._new_frame_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Camera %r released.", self._index)

    @property
    def exhausted(self) -> bool:
        """Whether a file source has reached end-of-file."""
        return self._exhausted

    @property
    def is_file_source(self) -> bool:
        """Whether this camera reads from a video file."""
        return self._is_file

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _grab_loop(self) -> None:
        """Background loop: continuously read frames and store the latest."""
        assert self._cap is not None
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                if self._is_file:
                    logger.info("Camera %r: end of video file.", self._index)
                    self._exhausted = True
                    with self._lock:
                        self._frame = None
                    self._new_frame_event.set()
                    break
                logger.warning("Camera %r: frame grab failed, retrying…", self._index)
                time.sleep(0.01)
                continue

            if self._is_file:
                # Pace roughly to source FPS when available
                fps = self._cap.get(cv2.CAP_PROP_FPS)
                delay = 1.0 / fps if fps and fps > 1 else 1.0 / 30.0
                with self._lock:
                    # Wait until previous frame is consumed to avoid skipping
                    while self._frame is not None and self._running:
                        self._lock.release()
                        time.sleep(0.001)
                        self._lock.acquire()
                    self._frame = frame
                self._new_frame_event.set()
                time.sleep(delay * 0.1)  # light pacing; consumer drives rate
            else:
                with self._lock:
                    self._frame = frame
