"""The Plane facade accumulates the right primitives; drawing calls are chainable."""

import numpy as np

from mathviz import Plane
from mathviz import primitives as P


def _types(plane):
    return [type(p) for p in plane.primitives]


def test_default_plane_has_grid_and_axes():
    p = Plane(extent=3)
    # one Grid + two axis Segments
    assert _types(p).count(P.Grid) == 1
    assert _types(p).count(P.Segment) == 2


def test_bare_plane_is_empty():
    p = Plane(grid=False, axes=False)
    assert p.primitives == []


def test_chainable_returns_self():
    p = Plane(grid=False, axes=False)
    assert p.vector([1, 1]) is p
    assert p.curve([[0, 0], [1, 1]]) is p


def test_primitive_methods_add_expected_types():
    p = Plane(grid=False, axes=False)
    p.vector([1, 0]).basis().segment([0, 0], [1, 1]).curve([[0, 0], [1, 1]]).points(
        [[0, 0]]
    ).text([0, 0], "x")
    t = _types(p)
    assert t.count(P.Arrow) == 3  # vector + basis(2)
    assert t.count(P.Segment) == 1
    assert t.count(P.Polyline) == 1
    assert t.count(P.Points) == 1
    assert t.count(P.Text) == 1


def test_line_clips_to_view_as_segment():
    p = Plane(grid=False, axes=False, extent=3)
    p.line([0, 0], [1, 0])
    seg = p.primitives[-1]
    assert isinstance(seg, P.Segment)
    assert abs(seg.p0[0]) >= 3 and abs(seg.p1[0]) >= 3  # extends past the view


def test_apply_matrix_adds_image_grid_and_basis_arrows():
    p = Plane(grid=False, axes=False, extent=3)
    p.apply_matrix([[1, 1], [0, 1]], faint=False)
    t = _types(p)
    assert t.count(P.Arrow) == 2  # M e1, M e2
    assert t.count(P.Polyline) > 0  # the warped lattice


def test_apply_complex_adds_polylines():
    p = Plane(grid=False, axes=False, extent=2)
    n_before = len(p.primitives)
    p.apply_complex(lambda z: z**2, faint=False, step=1.0)
    assert sum(isinstance(x, P.Polyline) for x in p.primitives) > 0
    assert len(p.primitives) > n_before


def test_raster_adds_raster_primitive():
    p = Plane(grid=False, axes=False, extent=1)
    p.raster(np.zeros((4, 4, 3)))
    assert isinstance(p.primitives[-1], P.Raster)
    assert p.primitives[-1].extent == (-1, 1, -1, 1)


def test_polygon_adds_filled_polygon():
    p = Plane(grid=False, axes=False)
    p.polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    assert isinstance(p.primitives[-1], P.Polygon)


def test_parallelogram_vertices():
    poly = (
        Plane(grid=False, axes=False)
        .parallelogram([0, 0], [2, 0], [0, 1])
        .primitives[-1]
    )
    assert isinstance(poly, P.Polygon)
    assert np.allclose(poly.pts, [[0, 0], [2, 0], [2, 1], [0, 1]])  # o, o+u, o+u+v, o+v


def test_field_draws_one_arrow_per_nonzero_sample():
    p = Plane(grid=False, axes=False, extent=2)
    p.field([[0, -1], [1, 0]], n=5)  # rotational field: zero only at the origin
    arrows = [x for x in p.primitives if isinstance(x, P.Arrow)]
    assert len(arrows) == 5 * 5 - 1  # 25 samples minus the origin


def test_field_normalize_gives_equal_lengths():
    p = Plane(grid=False, axes=False, extent=2)
    p.field([[1, 0], [0, 1]], n=4, normalize=True)  # n=4 avoids sampling the origin
    lens = [
        np.hypot(*(a.head - a.tail)) for a in p.primitives if isinstance(a, P.Arrow)
    ]
    assert len(lens) == 16
    assert np.allclose(lens, lens[0])


def test_field_complex_adds_arrows():
    p = Plane(grid=False, axes=False, extent=2)
    p.field_complex(lambda z: z, n=4)  # the identity field (Re z, Im z)
    assert any(isinstance(x, P.Arrow) for x in p.primitives)
