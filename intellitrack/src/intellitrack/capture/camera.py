"""Camera capture module.

Provides the :class:`Camera` class which wraps OpenCV's ``VideoCapture`` and
continuously grabs frames in a background thread so that the pipeline always
consumes the *latest* available frame without accumulating lag.
"""

import logging
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraUnavailableError(RuntimeError):
    """Raised when a camera device cannot be opened."""


class Camera:
    """Thread-safe, single-slot frame buffer camera wrapper.

    The internal reader thread continuously grabs frames from
    ``cv2.VideoCapture`` and stores only the most recent one.  This prevents
    frame-queue build-up when downstream processing is slower than the camera
    frame rate.

    Args:
        index: ``cv2.VideoCapture`` device index (or filename for video files).
        width: Requested capture width in pixels.
        height: Requested capture height in pixels.
    """

    def __init__(self, index: int, width: int, height: int) -> None:
        self._index = index
        self._width = width
        self._height = height

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the capture device and start the background grabber thread.

        Raises:
            CameraUnavailableError: If the device index cannot be opened.
        """
        cap = cv2.VideoCapture(self._index)
        if not cap.isOpened():
            cap.release()
            raise CameraUnavailableError(
                f"Cannot open camera at index {self._index!r}. "
                "Check that the device is connected and the index is correct."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap = cap
        self._running = True

        self._thread = threading.Thread(
            target=self._grab_loop, daemon=True, name="camera-grabber"
        )
        self._thread.start()
        logger.info(
            "Camera %r started (requested %dx%d)", self._index, self._width, self._height
        )

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the most recently captured frame.

        Returns:
            A ``(success, frame)`` tuple.  ``success`` is ``False`` and
            ``frame`` is ``None`` if no frame has been captured yet.
        """
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def release(self) -> None:
        """Stop the grabber thread and release the underlying capture device."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Camera %r released.", self._index)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _grab_loop(self) -> None:
        """Background loop: continuously read frames and store the latest."""
        assert self._cap is not None
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Camera %r: frame grab failed, retrying…", self._index)
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
