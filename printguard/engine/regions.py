"""Overlapping views for offline evaluation of a single camera frame."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from printguard.engine import vision


@dataclass(frozen=True)
class Region:
    """A square source view and the source pixels retained by preprocessing."""

    name: str
    source_box: tuple[int, int, int, int]
    visible_box: tuple[int, int, int, int]


def _starts(length: int, visible: int, overlap: float) -> list[int]:
    last = max(0, length - visible)
    stride = max(1, math.floor(visible * (1 - overlap)))
    positions = list(range(0, last + 1, stride))
    if positions[-1] != last:
        positions.append(last)
    return positions


def plan_regions(width: int, height: int, tile_size: int = 320, overlap: float = 0.25) -> list[Region]:
    """Plans square views whose retained centers cover every source pixel.

    Args:
        width: Source frame width.
        height: Source frame height.
        tile_size: Source view side, a positive multiple of 16.
        overlap: Minimum overlap between retained centers, from zero to below one.

    Returns:
        Regions in row-major order, including boundary views with edge padding.

    Raises:
        ValueError: Dimensions or overlap are invalid.
    """
    if width <= 0 or height <= 0 or tile_size < 16 or tile_size % 16:
        raise ValueError("positive dimensions and a tile size divisible by 16 are required")
    if not math.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("overlap must be finite and between zero and one")
    visible = tile_size * vision.INPUT_SIZE // vision.RESIZE_SHORTEST
    margin = (tile_size - visible) // 2
    return [
        Region(
            f"r{row}c{column}",
            (x - margin, y - margin, x - margin + tile_size, y - margin + tile_size),
            (x, y, min(width, x + visible), min(height, y + visible)),
        )
        for row, y in enumerate(_starts(height, visible, overlap))
        for column, x in enumerate(_starts(width, visible, overlap))
    ]


def region_image(rgb: np.ndarray, region: Region) -> np.ndarray:
    """Extracts one view, replicating edge pixels where it crosses the frame.

    Args:
        rgb: Source RGB frame.
        region: View planned for this frame's dimensions.

    Returns:
        Square RGB image ready for the unchanged model preprocessing.
    """
    height, width = rgb.shape[:2]
    left, top, right, bottom = region.source_box
    view = rgb[max(0, top):min(height, bottom), max(0, left):min(width, right)]
    return np.pad(view, ((max(0, -top), max(0, bottom - height)), (max(0, -left), max(0, right - width)), (0, 0)), mode="edge")
