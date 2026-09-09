"""Tests for SerialBridge mock fallback behaviour."""

import logging

from intellitrack.control.serial_bridge import MockSerialBridge, SerialBridge


def test_connect_invalid_port_falls_back_to_mock(caplog) -> None:
    """connect() on a nonexistent port must enter mock mode without raising."""
    bridge = SerialBridge(
        port="/dev/does-not-exist",
        baud_rate=115200,
        mock_if_unavailable=True,
    )
    with caplog.at_level(logging.WARNING):
        ok = bridge.connect()

    assert ok is True
    assert bridge.is_mock is True
    assert any(
        "hardware not available" in r.message.lower()
        or "mock mode" in r.message.lower()
        or "failed to open" in r.message.lower()
        for r in caplog.records
    )
    bridge.close()


def test_send_angles_in_mock_mode_logs_values(caplog) -> None:
    """send_angles in mock mode must log pan/tilt and not raise."""
    bridge = SerialBridge(
        port="/dev/does-not-exist",
        baud_rate=115200,
        mock_if_unavailable=True,
    )
    bridge.connect()

    with caplog.at_level(logging.INFO):
        bridge.send_angles(45.4, 120.6)

    messages = " ".join(r.message for r in caplog.records)
    assert "PAN:45" in messages or "pan" in messages.lower()
    assert "TILT:121" in messages or "tilt" in messages.lower() or "120" in messages
    bridge.close()


def test_mock_serial_bridge_interface() -> None:
    """MockSerialBridge exposes the same connect/send/close interface."""
    mock = MockSerialBridge()
    assert mock.connect() is True
    assert mock.is_mock is True
    mock.send_angles(90.0, 90.0)
    assert mock.last_pan == 90.0
    assert mock.last_tilt == 90.0
    mock.close()


def test_connect_without_mock_returns_false_on_failure() -> None:
    """When mock_if_unavailable is False, connect() returns False on failure."""
    bridge = SerialBridge(
        port="/dev/does-not-exist",
        baud_rate=115200,
        mock_if_unavailable=False,
    )
    ok = bridge.connect()
    assert ok is False
    bridge.close()
