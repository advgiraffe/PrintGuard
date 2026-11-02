"""Coverage and boundary preservation for experimental overlapping views."""

from __future__ import annotations

import numpy as np
import pytest

from printguard.engine import regions, vision


@pytest.mark.parametrize("shape,tile", [((360, 640), 320), ((360, 640), 192), ((640, 360), 320), ((73, 91), 192), ((1, 1), 16)])
def test_every_source_pixel_is_in_a_retained_center(shape: tuple[int, int], tile: int) -> None:
    height, width = shape
    coverage = np.zeros(shape, dtype=np.uint8)
    rgb = np.arange(height * width * 3, dtype=np.uint32).reshape(height, width, 3).astype(np.uint8)
    before = rgb.copy()
    margin = tile // 16
    for region in regions.plan_regions(width, height, tile):
        left, top, right, bottom = region.visible_box
        coverage[top:bottom, left:right] += 1
        view = regions.region_image(rgb, region)
        assert view.shape == (tile, tile, 3)
        np.testing.assert_array_equal(view[margin:margin+bottom-top, margin:margin+right-left], rgb[top:bottom, left:right])
    assert coverage.min() >= 1
    np.testing.assert_array_equal(rgb, before)


def test_edge_marker_survives_real_preprocessing() -> None:
    rgb = np.zeros((360, 640, 3), dtype=np.uint8)
    rgb[:32, :32] = 255
    assets = vision.Assets(mean=(0, 0, 0), std=(1, 1, 1), prototypes={})
    baseline = vision.preprocess(rgb, assets)
    first = regions.plan_regions(640, 360)[0]
    tiled = vision.preprocess(regions.region_image(rgb, first), assets)
    assert baseline.max() == 0
    assert tiled[0, 0, :10, :10].min() > 0.99


@pytest.mark.parametrize("tile,overlap", [(15, .25), (321, .25), (320, -0.1), (320, 1), (320, float("nan"))])
def test_invalid_grid_is_rejected(tile: int, overlap: float) -> None:
    with pytest.raises(ValueError):
        regions.plan_regions(640, 360, tile, overlap)
