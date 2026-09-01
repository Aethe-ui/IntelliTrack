"""Servo angle mapper.

Translates pixel-space error ``(dx, dy)`` from the frame centre into absolute
pan and tilt degree commands, applying a deadband and degree-range clamping.
"""

import logging
from dataclasses import dataclass
from typing import Tuple

logger = logging.getLogger(__name__)


@dataclass
class ServoCommand:
    """Resolved servo angles for one control cycle.

    Attributes:
        pan_deg: Absolute pan angle in degrees.
        tilt_deg: Absolute tilt angle in degrees.
    """

    pan_deg: float
    tilt_deg: float


class ServoMapper:
    """Convert pixel error into servo degree commands.

    The mapper keeps track of the current pan/tilt angles (starting at centre)
    and increments them by the PID output each frame.

    Args:
        pan_min_deg: Minimum allowed pan angle.
        pan_max_deg: Maximum allowed pan angle.
        tilt_min_deg: Minimum allowed tilt angle.
        tilt_max_deg: Maximum allowed tilt angle.
        pan_center_deg: Starting pan angle (also returned when no target).
        tilt_center_deg: Starting tilt angle.
        deadband_px: Pixel radius within which errors are ignored (avoids
            constant micro-adjustments when the target is nearly centred).
    """

    def __init__(
        self,
        pan_min_deg: float,
        pan_max_deg: float,
        tilt_min_deg: float,
        tilt_max_deg: float,
        pan_center_deg: float,
        tilt_center_deg: float,
        deadband_px: float,
    ) -> None:
        self._pan_min = pan_min_deg
        self._pan_max = pan_max_deg
        self._tilt_min = tilt_min_deg
        self._tilt_max = tilt_max_deg
        self._deadband = deadband_px

        # Current servo angles — start at configured centre
        self._pan_deg: float = pan_center_deg
        self._tilt_deg: float = tilt_center_deg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self, dx: float, dy: float, pan_delta: float, tilt_delta: float
    ) -> ServoCommand:
        """Apply PID deltas (if outside deadband) and return the new command.

        Args:
            dx: Signed horizontal pixel error (positive = target is right of centre).
            dy: Signed vertical pixel error (positive = target is below centre).
            pan_delta: PID output for the pan axis (degrees to add).
            tilt_delta: PID output for the tilt axis (degrees to add).

        Returns:
            :class:`ServoCommand` with clamped absolute angles.
        """
        # Apply deadband: if error is small, don't move
        if abs(dx) > self._deadband:
            self._pan_deg += pan_delta
        if abs(dy) > self._deadband:
            self._tilt_deg += tilt_delta

        # Clamp to valid servo range
        self._pan_deg = max(self._pan_min, min(self._pan_max, self._pan_deg))
        self._tilt_deg = max(self._tilt_min, min(self._tilt_max, self._tilt_deg))

        return ServoCommand(pan_deg=self._pan_deg, tilt_deg=self._tilt_deg)

    def center(self) -> ServoCommand:
        """Return the centre position without updating internal state."""
        return ServoCommand(
            pan_deg=(self._pan_min + self._pan_max) / 2.0,
            tilt_deg=(self._tilt_min + self._tilt_max) / 2.0,
        )

    @property
    def current_angles(self) -> Tuple[float, float]:
        """Current ``(pan_deg, tilt_deg)`` without advancing state."""
        return (self._pan_deg, self._tilt_deg)
