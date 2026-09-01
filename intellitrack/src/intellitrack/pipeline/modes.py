"""Tracking mode enumeration."""

from enum import Enum


class TrackingMode(str, Enum):
    """Enumeration of supported tracking/prediction modes.

    Values match the ``prediction.mode`` key in ``configs/default.yaml``.
    """

    REACTIVE_PID = "reactive_pid"
    KALMAN = "kalman"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
