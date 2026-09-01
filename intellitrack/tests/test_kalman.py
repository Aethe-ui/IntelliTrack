"""Unit tests for the ConstantVelocityKalman2D filter.

Feeds a synthetic linear trajectory (x(t) = t, y(t) = 2t) through update/predict
cycles and verifies that the predicted position converges close to the true next
point after a warm-up period.
"""

import sys
import math
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intellitrack.prediction.kalman import ConstantVelocityKalman2D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOLERANCE = 3.0  # pixels — predicted must be within this of true position after warm-up
WARMUP_STEPS = 20  # frames before we start checking accuracy


def linear_trajectory(n: int):
    """Generate n (x, y) points along x=t, y=2t."""
    return [(float(t), 2.0 * float(t)) for t in range(n)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_kalman_converges_on_linear_trajectory():
    """After warm-up, the Kalman prediction should be within TOLERANCE px of truth."""
    kf = ConstantVelocityKalman2D(dt=1.0)
    points = linear_trajectory(50)

    for i, (x, y) in enumerate(points[:-1]):
        kf.update((x, y))
        pred_x, pred_y = kf.predict()

        if i >= WARMUP_STEPS:
            true_next_x, true_next_y = points[i + 1]
            error_x = abs(pred_x - true_next_x)
            error_y = abs(pred_y - true_next_y)
            assert error_x < TOLERANCE, (
                f"At step {i}: predicted x={pred_x:.2f}, true={true_next_x:.2f}, "
                f"error={error_x:.2f} > tolerance={TOLERANCE}"
            )
            assert error_y < TOLERANCE, (
                f"At step {i}: predicted y={pred_y:.2f}, true={true_next_y:.2f}, "
                f"error={error_y:.2f} > tolerance={TOLERANCE}"
            )


def test_kalman_predict_before_update_returns_zero():
    """Calling predict() before any update() should return (0, 0) gracefully."""
    kf = ConstantVelocityKalman2D()
    result = kf.predict()
    assert result == (0.0, 0.0)


def test_kalman_reset_clears_state():
    """After reset(), the filter should behave as if freshly constructed."""
    kf = ConstantVelocityKalman2D(dt=1.0)
    for i in range(10):
        kf.update((float(i), float(i)))
        kf.predict()

    kf.reset()
    # Post-reset predict should return (0, 0)
    result = kf.predict()
    assert result == (0.0, 0.0)


def test_kalman_returns_tuple_of_floats():
    """predict() must always return a tuple of two floats."""
    kf = ConstantVelocityKalman2D()
    kf.update((100.0, 200.0))
    result = kf.predict()
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(v, float) for v in result)


def test_kalman_velocity_estimation():
    """After feeding a constant-velocity sequence, velocity should be estimated correctly."""
    kf = ConstantVelocityKalman2D(dt=1.0, process_noise=0.1, measurement_noise=0.1)
    # x increases by 5 per step, y by 3
    for i in range(30):
        kf.update((5.0 * i, 3.0 * i))
        kf.predict()

    # The filter's internal vx, vy should be close to 5 and 3
    vx = float(kf._x[2, 0])
    vy = float(kf._x[3, 0])
    assert abs(vx - 5.0) < 1.0, f"Expected vx≈5.0, got {vx:.3f}"
    assert abs(vy - 3.0) < 1.0, f"Expected vy≈3.0, got {vy:.3f}"
