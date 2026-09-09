"""Serial bridge to Arduino/ESP32 pan-tilt firmware.

Speaks the newline-terminated protocol::

    PAN:<int> TILT:<int>\\n

and expects ``ACK\\n`` after each successful command.  When the serial port
cannot be opened and ``mock_if_unavailable`` is set, the bridge switches to an
in-memory mock mode so the rest of the pipeline can run without hardware.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Protocol

logger = logging.getLogger(__name__)

# Short timeout waiting for ACK from the firmware (seconds).
_ACK_TIMEOUT_S = 0.05


class SerialBridgeProtocol(Protocol):
    """Common interface for real and mock serial bridges."""

    def connect(self) -> bool:
        """Open the connection (or enter mock mode). Returns True on success."""

    def send_angles(self, pan_deg: float, tilt_deg: float) -> None:
        """Send a pan/tilt command."""

    def close(self) -> None:
        """Release the underlying resource."""

    @property
    def is_mock(self) -> bool:
        """Whether the bridge is operating in mock/log-only mode."""


class MockSerialBridge:
    """In-memory serial bridge that logs commands instead of transmitting them.

    Implements the same interface as :class:`SerialBridge` so tests and
    hardware-less runs never need a real device.
    """

    def __init__(self) -> None:
        self._connected = False
        self.last_pan: Optional[float] = None
        self.last_tilt: Optional[float] = None

    def connect(self) -> bool:
        """Enter mock mode. Always succeeds."""
        self._connected = True
        logger.warning(
            "hardware not available — running in mock mode."
        )
        return True

    def send_angles(self, pan_deg: float, tilt_deg: float) -> None:
        """Log the commanded angles without transmitting."""
        pan_i = int(round(pan_deg))
        tilt_i = int(round(tilt_deg))
        self.last_pan = float(pan_i)
        self.last_tilt = float(tilt_i)
        logger.info("MOCK SERVO CMD — PAN:%d TILT:%d", pan_i, tilt_i)

    def close(self) -> None:
        """Mark the mock bridge as closed."""
        self._connected = False
        logger.debug("MockSerialBridge closed.")

    @property
    def is_mock(self) -> bool:
        """Always ``True`` for the mock bridge."""
        return True


class SerialBridge:
    """Serial bridge to the pan-tilt Arduino/ESP32 firmware.

    Args:
        port: Serial device path (e.g. ``/dev/ttyUSB0`` or ``COM3``).
        baud_rate: Baud rate matching the firmware (default 115200).
        mock_if_unavailable: If ``True``, fall back to mock mode when the
            port cannot be opened instead of raising.
    """

    def __init__(
        self,
        port: str,
        baud_rate: int,
        mock_if_unavailable: bool = True,
    ) -> None:
        self._port = port
        self._baud_rate = baud_rate
        self._mock_if_unavailable = mock_if_unavailable

        self._ser = None  # serial.Serial | None
        self._mock = False
        self._mock_bridge: Optional[MockSerialBridge] = None

    def connect(self) -> bool:
        """Try to open the serial port; fall back to mock mode on failure.

        Returns:
            ``True`` if connected (real or mock).  Only returns ``False`` when
            the port fails *and* ``mock_if_unavailable`` is ``False``.
        """
        try:
            import serial  # pyserial

            self._ser = serial.Serial(
                port=self._port,
                baudrate=self._baud_rate,
                timeout=_ACK_TIMEOUT_S,
                write_timeout=_ACK_TIMEOUT_S,
            )
            # Give the MCU a moment after port open (USB-serial reset).
            time.sleep(0.5)
            # Drain any READY banner
            try:
                self._ser.reset_input_buffer()
            except Exception:  # noqa: BLE001
                pass
            self._mock = False
            logger.info(
                "SerialBridge connected to %s @ %d baud.",
                self._port,
                self._baud_rate,
            )
            return True
        except Exception as exc:  # noqa: BLE001 — port missing, permission, etc.
            logger.warning(
                "Failed to open serial port %s: %s", self._port, exc
            )
            if self._mock_if_unavailable:
                self._mock = True
                self._mock_bridge = MockSerialBridge()
                self._mock_bridge.connect()
                return True
            return False

    def send_angles(self, pan_deg: float, tilt_deg: float) -> None:
        """Format and write ``PAN:<int> TILT:<int>\\n``; wait briefly for ACK.

        In mock mode the command is logged only.  ACK timeouts are logged,
        never raised.

        Args:
            pan_deg: Desired pan angle in degrees.
            tilt_deg: Desired tilt angle in degrees.
        """
        if self._mock and self._mock_bridge is not None:
            self._mock_bridge.send_angles(pan_deg, tilt_deg)
            return

        if self._ser is None or not self._ser.is_open:
            logger.warning("SerialBridge.send_angles: port not open; ignoring.")
            return

        pan_i = int(round(max(0, min(180, pan_deg))))
        tilt_i = int(round(max(0, min(180, tilt_deg))))
        cmd = f"PAN:{pan_i} TILT:{tilt_i}\n"

        try:
            self._ser.write(cmd.encode("ascii"))
            self._ser.flush()
            # Non-blocking-ish ACK wait with short timeout already on the port
            deadline = time.perf_counter() + _ACK_TIMEOUT_S
            buf = b""
            while time.perf_counter() < deadline:
                chunk = self._ser.read(32)
                if chunk:
                    buf += chunk
                    if b"ACK" in buf:
                        return
                else:
                    break
            if b"ACK" not in buf:
                logger.warning(
                    "SerialBridge: ACK timeout after sending %r", cmd.strip()
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("SerialBridge.send_angles failed: %s", exc)

    def close(self) -> None:
        """Close the serial port or mock bridge."""
        if self._mock_bridge is not None:
            self._mock_bridge.close()
            self._mock_bridge = None
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Error closing serial port: %s", exc)
            self._ser = None
        logger.debug("SerialBridge closed.")

    @property
    def is_mock(self) -> bool:
        """Whether the bridge is currently in mock mode."""
        return self._mock
