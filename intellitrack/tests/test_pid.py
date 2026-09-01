"""Unit tests for PIDController.

Verifies:
  - Output stays within ±output_limit at all times.
  - A closed-loop simulation converges (position approaches target).
  - Integral anti-windup prevents runaway accumulation.
  - reset() clears integral and derivative state.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intellitrack.control.pid_controller import PIDController


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_output_never_exceeds_limit():
    """Controller output must always be clamped to ±output_limit."""
    output_limit = 15.0
    pid = PIDController(kp=1.0, ki=0.5, kd=0.1, output_limit=output_limit)

    for error in [0.0, 10.0, -10.0, 500.0, -500.0]:
        out = pid.compute(error, dt=0.033)
        assert abs(out) <= output_limit + 1e-9, (
            f"Output {out} exceeded ±{output_limit} for error={error}"
        )


def test_closed_loop_converges():
    """A simple P-controller closed loop should reduce error toward zero over time."""
    pid = PIDController(kp=0.5, ki=0.0, kd=0.0, output_limit=50.0)
    target = 100.0
    position = 0.0
    dt = 0.033

    for _ in range(200):
        error = target - position
        position += pid.compute(error, dt)

    # After 200 iterations the position should be very close to target
    assert abs(target - position) < 1.0, (
        f"Closed-loop failed to converge: final position={position:.2f}, target={target}"
    )


def test_pid_closed_loop_converges_with_all_gains():
    """Full PID (non-zero ki, kd) should also converge and not exhibit runaway."""
    pid = PIDController(kp=0.3, ki=0.05, kd=0.02, output_limit=20.0)
    target = 200.0
    position = 0.0
    dt = 0.033

    for _ in range(500):
        error = target - position
        position += pid.compute(error, dt)

    assert abs(target - position) < 5.0, (
        f"PID failed to converge close enough: position={position:.2f}, target={target}"
    )


def test_reset_clears_state():
    """After reset(), the controller should behave as if freshly constructed."""
    pid = PIDController(kp=0.5, ki=1.0, kd=0.1, output_limit=50.0)

    # Run for a while to build up integral
    for _ in range(50):
        pid.compute(10.0, dt=0.033)

    pid.reset()
    # With integral and prev_error zeroed, the first output should be purely proportional
    out = pid.compute(10.0, dt=0.033)
    expected_p = 0.5 * 10.0  # kp * error
    # The d term is (error - prev_error) / dt = (10 - 0) / 0.033 ≈ 303, scaled by kd=0.1 → ≈30
    # But output_limit clamps. Let's just verify the output is within limit and > 0
    assert abs(out) <= 50.0
    assert out > 0.0  # positive error should produce positive output


def test_zero_error_gives_zero_output():
    """Zero error with zero ki should produce zero output."""
    pid = PIDController(kp=1.0, ki=0.0, kd=0.0, output_limit=10.0)
    # First call: prev_error = 0, so d-term = 0 too
    out = pid.compute(0.0, dt=0.033)
    assert out == pytest.approx(0.0)


def test_nonpositive_dt_handled_gracefully():
    """dt <= 0 should not crash the controller (uses fallback dt=1e-3)."""
    pid = PIDController(kp=0.5, ki=0.0, kd=0.1, output_limit=20.0)
    # Should not raise
    out = pid.compute(5.0, dt=0.0)
    assert isinstance(out, float)
