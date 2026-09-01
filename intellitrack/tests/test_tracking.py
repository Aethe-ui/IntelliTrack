"""Unit tests for target selection strategies.

Tests operate entirely on synthetic :class:`TrackedObject` lists without any
real video or YOLO inference.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intellitrack.tracking.byte_tracker_wrapper import TrackedObject
from intellitrack.tracking.target_selector import TargetSelector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FRAME_W, FRAME_H = 640, 480


def make_track(
    track_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    conf: float = 0.9,
    cls: str = "person",
    age: int = 1,
) -> TrackedObject:
    """Build a synthetic :class:`TrackedObject`."""
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return TrackedObject(
        track_id=track_id,
        bbox_xyxy=(x1, y1, x2, y2),
        centroid=(cx, cy),
        class_name=cls,
        confidence=conf,
        age_frames=age,
    )


# ---------------------------------------------------------------------------
# Strategy: largest_bbox
# ---------------------------------------------------------------------------


def test_largest_bbox_picks_biggest_area():
    sel = TargetSelector("largest_bbox", FRAME_W, FRAME_H, max_lost_frames=5)
    tracks = [
        make_track(1, 0, 0, 50, 50),    # area = 2500
        make_track(2, 0, 0, 100, 200),  # area = 20000  ← should be chosen
        make_track(3, 0, 0, 30, 30),    # area = 900
    ]
    result = sel.select(tracks)
    assert result is not None
    assert result.track_id == 2


# ---------------------------------------------------------------------------
# Strategy: closest_to_center
# ---------------------------------------------------------------------------


def test_closest_to_center_picks_nearest():
    sel = TargetSelector("closest_to_center", FRAME_W, FRAME_H, max_lost_frames=5)
    # Frame center is (320, 240)
    tracks = [
        make_track(1, 0, 0, 40, 40),      # centroid (20, 20) — far
        make_track(2, 300, 220, 340, 260), # centroid (320, 240) — exact centre ← should win
        make_track(3, 500, 400, 600, 480), # centroid (550, 440) — far
    ]
    result = sel.select(tracks)
    assert result is not None
    assert result.track_id == 2


# ---------------------------------------------------------------------------
# Strategy: highest_confidence
# ---------------------------------------------------------------------------


def test_highest_confidence_picks_max_conf():
    sel = TargetSelector("highest_confidence", FRAME_W, FRAME_H, max_lost_frames=5)
    tracks = [
        make_track(1, 0, 0, 50, 50, conf=0.5),
        make_track(2, 0, 0, 50, 50, conf=0.95),  # ← highest
        make_track(3, 0, 0, 50, 50, conf=0.7),
    ]
    result = sel.select(tracks)
    assert result is not None
    assert result.track_id == 2


# ---------------------------------------------------------------------------
# Strategy: manual_id_lock
# ---------------------------------------------------------------------------


def test_manual_id_lock_follows_locked_id():
    sel = TargetSelector("manual_id_lock", FRAME_W, FRAME_H, max_lost_frames=5)
    tracks = [
        make_track(1, 0, 0, 50, 50),
        make_track(7, 300, 200, 400, 300),
    ]
    sel.lock_id(7)
    result = sel.select(tracks)
    assert result is not None
    assert result.track_id == 7


def test_manual_id_lock_within_grace_period_returns_none():
    """Lost for < max_lost_frames → returns None (grace-period hold-off)."""
    max_lost = 5
    sel = TargetSelector("manual_id_lock", FRAME_W, FRAME_H, max_lost_frames=max_lost)
    sel.lock_id(99)  # ID 99 does not exist in any tracks

    tracks = [make_track(1, 0, 0, 50, 50)]

    # Call fewer than max_lost_frames times — should return None each time
    for _ in range(max_lost - 1):
        result = sel.select(tracks)
        assert result is None, "Expected None during grace period"


def test_manual_id_lock_released_after_max_lost_frames():
    """After max_lost_frames, the selector should reselect from available tracks."""
    max_lost = 3
    sel = TargetSelector("manual_id_lock", FRAME_W, FRAME_H, max_lost_frames=max_lost)
    sel.lock_id(99)  # ID 99 never appears

    tracks = [make_track(5, 300, 220, 340, 260)]  # closest to center

    # Exhaust the grace period
    for _ in range(max_lost):
        sel.select(tracks)

    # On the next call, the lock should be released and track 5 selected
    result = sel.select(tracks)
    assert result is not None
    assert result.track_id == 5


def test_select_returns_none_on_empty_tracks():
    """All strategies must return None when no tracks are present."""
    for strategy in ["largest_bbox", "closest_to_center", "highest_confidence", "manual_id_lock"]:
        sel = TargetSelector(strategy, FRAME_W, FRAME_H, max_lost_frames=5)
        result = sel.select([])
        assert result is None, f"Expected None for strategy '{strategy}' with empty tracks"
