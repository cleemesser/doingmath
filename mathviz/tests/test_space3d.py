"""The 3D scene: primitives, field sampling, analytic landscape, and rendering on both backends."""

import numpy as np
import pytest
from PIL import Image

from mathviz import Space3D
from mathviz import primitives as P
from mathviz import space3d


def test_3d_primitives_normalize():
    a = P.Arrow3D([0, 0, 0], [1, 2, 3])
    assert a.tail.shape == (3,) and a.head.shape == (3,)
    assert P.Line3D([[0, 0, 0], [1, 1, 1]]).pts.shape == (2, 3)
    assert P.Points3D([[0, 0, 0]]).pts.shape == (1, 3)


def test_surface_holds_grid():
    X = np.zeros((4, 5))
    assert P.Surface(X, X, X).Z.shape == (4, 5)


def test_as_field3_matrix_and_callable():
    f = space3d._as_field3(np.eye(3))
    assert np.allclose(f([[1, 2, 3]]), [[1, 2, 3]])
    g = lambda p: p
    assert space3d._as_field3(g) is g


def test_field_draws_arrow_per_nonzero_sample():
    # rotational+z field (−y, x, z) vanishes only at the origin
    s = Space3D(bounds=2).field([[0, -1, 0], [1, 0, 0], [0, 0, 1]], n=3)
    arrows = [x for x in s.primitives if isinstance(x, P.Arrow3D)]
    assert len(arrows) == 3**3 - 1


def test_field_normalize_gives_equal_lengths():
    s = Space3D(bounds=2).field(np.eye(3), n=2)  # n=2 avoids the origin; source field
    lens = [
        np.linalg.norm(a.head - a.tail)
        for a in s.primitives
        if isinstance(a, P.Arrow3D)
    ]
    assert len(lens) == 8
    assert np.allclose(lens, lens[0])


def test_landscape_is_a_colored_capped_surface():
    s = Space3D(bounds=2).landscape(lambda z: z, res=30, zmax=2)
    surf = [x for x in s.primitives if isinstance(x, P.Surface)][-1]
    assert surf.colors.shape == (30, 30, 3)
    assert np.isfinite(surf.Z).all()
    assert surf.Z.max() <= 2 + 1e-9  # clipped to zmax


def test_landscape_pole_is_capped_not_infinite():
    # 1/z blows up at 0; the height must stay finite (clipped)
    s = Space3D(bounds=1).landscape(lambda z: 1 / z, res=40, zmax=3)
    surf = s.primitives[-1]
    assert np.isfinite(surf.Z).all() and surf.Z.max() <= 3 + 1e-9


@pytest.mark.parametrize("be", ["vedo", "mpl"])
def test_render_3d_nonblank(tmp_path, be):
    out = tmp_path / f"s3_{be}.png"
    Space3D(bounds=2, backend=be).field([[0, -1, 0], [1, 0, 0], [0, 0, 1]], n=3).save(
        str(out)
    )
    arr = np.asarray(Image.open(out).convert("RGB"))
    assert int((arr > 20).sum()) > 500
