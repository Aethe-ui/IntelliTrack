"""Constant-velocity 2D Kalman filter for trajectory prediction.

Implements a standard linear Kalman filter with state vector
``[x, y, vx, vy]`` using only NumPy — no external Kalman library is required
so that the derivation is transparent and citable in a research write-up.
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ConstantVelocityKalman2D:
    """A 2D Kalman filter assuming constant velocity motion.

    **State vector:** ``[x, y, vx, vy]^T``

    **Observation vector:** ``[x, y]^T`` (centroid measurements)

    All matrices are constructed analytically from a single time-step ``dt``.

    Args:
        dt: Time step between frames in seconds (default 1/30 for 30 fps).
        process_noise: Scalar multiplier for the process noise covariance Q.
            Higher values make the filter trust measurements more.
        measurement_noise: Scalar multiplier for the measurement noise
            covariance R.  Higher values make the filter trust the model more.
    """

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        process_noise: float = 1.0,
        measurement_noise: float = 10.0,
    ) -> None:
        self._dt = dt

        # State transition matrix F (constant-velocity model)
        self._F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

        # Observation matrix H: we observe [x, y] only
        self._H = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]],
            dtype=np.float64,
        )

        # Process noise covariance Q
        self._Q = process_noise * np.eye(4, dtype=np.float64)

        # Measurement noise covariance R
        self._R = measurement_noise * np.eye(2, dtype=np.float64)

        # Initial state estimate — zero until first update
        self._x = np.zeros((4, 1), dtype=np.float64)

        # Initial error covariance — large diagonal → high uncertainty
        self._P = np.eye(4, dtype=np.float64) * 1000.0

        self._initialized = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, measurement: Tuple[float, float]) -> None:
        """Correct the state estimate with a new centroid measurement.

        If the filter has not been initialised yet (first call), the state is
        seeded directly from the measurement and the velocity is set to zero.

        Args:
            measurement: Observed centroid ``(x, y)`` in pixel coordinates.
        """
        z = np.array([[measurement[0]], [measurement[1]]], dtype=np.float64)

        if not self._initialized:
            self._x[0, 0] = measurement[0]
            self._x[1, 0] = measurement[1]
            self._initialized = True
            return

        # Innovation / residual
        y = z - self._H @ self._x

        # Innovation covariance
        S = self._H @ self._P @ self._H.T + self._R

        # Kalman gain
        K = self._P @ self._H.T @ np.linalg.inv(S)

        # Updated state estimate
        self._x = self._x + K @ y

        # Updated error covariance
        I = np.eye(4, dtype=np.float64)
        self._P = (I - K @ self._H) @ self._P

    def predict(self) -> Tuple[float, float]:
        """Advance the state one time step and return the predicted position.

        Returns:
            Predicted ``(x, y)`` centroid position for the next frame.
        """
        if not self._initialized:
            logger.warning(
                "ConstantVelocityKalman2D.predict() called before first update(); "
                "returning (0.0, 0.0)."
            )
            return (0.0, 0.0)

        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q

        return (float(self._x[0, 0]), float(self._x[1, 0]))

    def reset(self) -> None:
        """Reset the filter to its uninitialised state."""
        self._x = np.zeros((4, 1), dtype=np.float64)
        self._P = np.eye(4, dtype=np.float64) * 1000.0
        self._initialized = False
