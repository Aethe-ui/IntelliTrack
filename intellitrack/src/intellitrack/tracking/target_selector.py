"""Target selection module.

Implements the :class:`TargetSelector` which picks one "active" target from a
list of tracked objects using one of four configurable strategies.
"""

import logging
from typing import List, Optional

from intellitrack.tracking.byte_tracker_wrapper import TrackedObject
from intellitrack.utils.geometry import bbox_area, euclidean_distance

logger = logging.getLogger(__name__)

# Sentinel representing "no target locked yet"
_NO_LOCK = -1


class TargetSelector:
    """Select a single target track from all active tracked objects.

    Supported strategies
    --------------------
    ``largest_bbox``
        Pick the track whose bounding box has the largest pixel area.
    ``closest_to_center``
        Pick the track whose centroid is closest to the frame center.
    ``highest_confidence``
        Pick the track with the highest detection confidence score.
    ``manual_id_lock``
        Lock onto a specific track ID set via :meth:`lock_id`.  Falls back to
        the ``closest_to_center`` strategy if that ID disappears for more than
        ``max_lost_frames`` consecutive frames.

    Args:
        strategy: One of the four strategy strings listed above.
        frame_width: Width of the video frame in pixels.
        frame_height: Height of the video frame in pixels.
        max_lost_frames: How many consecutive update calls without the locked
            ID before the lock is released and the fallback strategy is used.
    """

    def __init__(
        self,
        strategy: str,
        frame_width: int,
        frame_height: int,
        max_lost_frames: int = 30,
    ) -> None:
        valid = {"largest_bbox", "closest_to_center", "highest_confidence", "manual_id_lock"}
        if strategy not in valid:
            raise ValueError(f"Unknown selection strategy {strategy!r}. Must be one of {valid}.")

        self._strategy = strategy
        self._frame_center = (frame_width / 2.0, frame_height / 2.0)
        self._max_lost_frames = max_lost_frames

        self._locked_id: int = _NO_LOCK
        self._lost_frames: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lock_id(self, track_id: int) -> None:
        """Manually lock the selector onto a specific track ID.

        Args:
            track_id: The ``track_id`` of the :class:`TrackedObject` to follow.
        """
        self._locked_id = track_id
        self._lost_frames = 0
        logger.info("TargetSelector: locked onto track_id=%d", track_id)

    def select(self, tracks: List[TrackedObject]) -> Optional[TrackedObject]:
        """Pick the active target from the current list of tracked objects.

        Args:
            tracks: All currently active tracked objects this frame.

        Returns:
            The selected :class:`TrackedObject`, or ``None`` if no tracks are
            available or if the locked target has been lost for fewer than
            ``max_lost_frames`` frames (grace-period hold-off).
        """
        if not tracks:
            self._handle_empty()
            return None

        # ---------- manual_id_lock: try to honour the locked ID first ----------
        if self._strategy == "manual_id_lock" and self._locked_id != _NO_LOCK:
            found = self._find_by_id(tracks, self._locked_id)
            if found is not None:
                self._lost_frames = 0
                return found

            # Locked ID is missing this frame
            self._lost_frames += 1
            if self._lost_frames <= self._max_lost_frames:
                # Still within the grace period — report "no target yet"
                logger.debug(
                    "TargetSelector: locked id=%d missing for %d/%d frames",
                    self._locked_id,
                    self._lost_frames,
                    self._max_lost_frames,
                )
                return None

            # Grace period expired — drop the lock and fall through to default strategy
            logger.info(
                "TargetSelector: locked id=%d lost for %d frames (> %d), releasing lock.",
                self._locked_id,
                self._lost_frames,
                self._max_lost_frames,
            )
            self._locked_id = _NO_LOCK
            self._lost_frames = 0

        # ---------- all other strategies (or fallback after lock release) ----------
        selected = self._apply_strategy(tracks)
        if selected is not None and self._strategy == "manual_id_lock":
            # Auto-lock onto the newly selected track
            self._locked_id = selected.track_id
        return selected

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_empty(self) -> None:
        """Update lost-frame counter when no tracks are visible at all."""
        if self._strategy == "manual_id_lock" and self._locked_id != _NO_LOCK:
            self._lost_frames += 1
            if self._lost_frames > self._max_lost_frames:
                logger.info(
                    "TargetSelector: no tracks for %d frames, releasing lock.", self._lost_frames
                )
                self._locked_id = _NO_LOCK
                self._lost_frames = 0

    def _find_by_id(
        self, tracks: List[TrackedObject], track_id: int
    ) -> Optional[TrackedObject]:
        for t in tracks:
            if t.track_id == track_id:
                return t
        return None

    def _apply_strategy(self, tracks: List[TrackedObject]) -> Optional[TrackedObject]:
        if self._strategy == "largest_bbox":
            return max(tracks, key=lambda t: bbox_area(t.bbox_xyxy))  # type: ignore[arg-type]
        elif self._strategy in ("closest_to_center", "manual_id_lock"):
            return min(
                tracks,
                key=lambda t: euclidean_distance(t.centroid, self._frame_center),
            )
        elif self._strategy == "highest_confidence":
            return max(tracks, key=lambda t: t.confidence)
        return None
