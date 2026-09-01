"""PID controller for servo axis error correction.

Provides a classic Proportional-Integral-Derivative controller with
integral anti-windup and symmetric output clamping.
"""

import logging

logger = logging.getLogger(__name__)


class PIDController:
    """Standard discrete PID controller with anti-windup and output clamping.

    Each axis (pan and tilt) uses its own :class:`PIDController` instance so
    that integral state is tracked independently per axis.

    Args:
        kp: Proportional gain.
        ki: Integral gain.
        kd: Derivative gain.
        output_limit: Maximum absolute value of the controller output
            (symmetric clamp: output is in ``[-output_limit, +output_limit]``).
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_limit: float,
    ) -> None:
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._output_limit = output_limit

        self._integral: float = 0.0
        self._prev_error: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, error: float, dt: float) -> float:
        """Compute the PID output for the given error and time step.

        Integral anti-windup: the integral accumulator is clamped to the range
        ``[-output_limit / ki, +output_limit / ki]`` (when ki > 0) so that it
        cannot grow beyond what the output limit would permit.

        Args:
            error: Signed error (target − current) in the control space
                (pixels for the pixel-error PID loop).
            dt: Time elapsed since the last call in seconds.  Must be > 0.

        Returns:
            Clamped controller output.
        """
        if dt <= 0:
            logger.warning("PIDController.compute: dt=%f is non-positive, using 1e-3.", dt)
            dt = 1e-3

        # Proportional term
        p_term = self._kp * error

        # Integral term with anti-windup clamping
        self._integral += error * dt
        if self._ki != 0.0:
            integral_limit = self._output_limit / self._ki
            self._integral = max(-integral_limit, min(integral_limit, self._integral))
        i_term = self._ki * self._integral

        # Derivative term
        d_term = self._kd * (error - self._prev_error) / dt
        self._prev_error = error

        raw_output = p_term + i_term + d_term
        return max(-self._output_limit, min(self._output_limit, raw_output))

    def reset(self) -> None:
        """Reset the integral accumulator and previous error to zero."""
        self._integral = 0.0
        self._prev_error = 0.0
