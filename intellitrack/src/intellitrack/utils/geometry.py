"""Geometry utility functions for IntelliTrack."""

from typing import Tuple


def bbox_centroid(bbox_xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Compute the centroid of a bounding box given in xyxy format.

    Args:
        bbox_xyxy: Bounding box as ``(x1, y1, x2, y2)``.

    Returns:
        Centroid as ``(cx, cy)``.
    """
    x1, y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_area(bbox_xyxy: Tuple[float, float, float, float]) -> float:
    """Compute the area of a bounding box given in xyxy format.

    Args:
        bbox_xyxy: Bounding box as ``(x1, y1, x2, y2)``.

    Returns:
        Area in pixels squared.
    """
    x1, y1, x2, y2 = bbox_xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def euclidean_distance(
    p1: Tuple[float, float], p2: Tuple[float, float]
) -> float:
    """Compute Euclidean distance between two 2D points.

    Args:
        p1: First point as ``(x, y)``.
        p2: Second point as ``(x, y)``.

    Returns:
        Euclidean distance.
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
