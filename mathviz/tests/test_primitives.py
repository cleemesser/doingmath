"""Scene primitives normalize their inputs to float NumPy arrays of the right shape."""

import numpy as np

from mathviz import primitives as P


def test_view_defaults():
    v = P.View()
    assert v.extent == 3.0
    assert v.size == (640, 640)


def test_segment_normalizes_points():
    s = P.Segment([0, 0], (1, 2))
    assert s.p0.shape == (2,) and s.p1.shape == (2,)
    assert s.p0.dtype == float
    assert np.allclose(s.p1, [1, 2])


def test_arrow_normalizes():
    a = P.Arrow([0, 0], [3, 4])
    assert a.tail.shape == (2,) and a.head.shape == (2,)


def test_polyline_and_points_shape():
    pl = P.Polyline([[0, 0], [1, 1], [2, 0]])
    assert pl.pts.shape == (3, 2)
    pts = P.Points([[0, 0], [1, 1]])
    assert pts.pts.shape == (2, 2)


def test_raster_holds_image_and_extent():
    img = np.zeros((8, 8, 3))
    r = P.Raster(img, (-1, 1, -1, 1))
    assert r.rgb.shape == (8, 8, 3)
    assert r.extent == (-1, 1, -1, 1)


def test_scene_collects_primitives():
    scene = P.Scene(P.View(), [P.Segment([0, 0], [1, 1])])
    assert len(scene.primitives) == 1
