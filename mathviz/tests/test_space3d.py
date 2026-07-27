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


def test_riemann_root_builds_colored_surface():
    s = Space3D(bounds=2).riemann_root(2, nr=20, ntheta=41)
    surf = s.primitives[-1]
    assert isinstance(surf, P.Surface)
    assert surf.Z.shape == (20, 41) and surf.colors.shape == (20, 41, 3)
    assert np.isfinite(surf.Z).all()


def test_riemann_root_spans_two_sheets():
    # height = Im(w) over the full w-disk → both sheets (negative and positive heights)
    surf = Space3D(bounds=2).riemann_root(2, nr=30, ntheta=120).primitives[-1]
    assert surf.Z.min() < 0 < surf.Z.max()


def test_riemann_log_is_a_rising_helicoid():
    surf = (
        Space3D(bounds=2)
        .riemann_log(sheets=2, nr=20, ntheta=61, height_scale=1.0)
        .primitives[-1]
    )
    assert isinstance(surf, P.Surface)
    assert np.isclose(surf.Z.min(), 0.0) and surf.Z.max() > 2 * np.pi  # ~2 full turns


def test_parallelepiped_is_a_six_face_mesh():
    m = (
        Space3D(bounds=2)
        .parallelepiped([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])
        .primitives[-1]
    )
    assert isinstance(m, P.Mesh3D)
    assert m.verts.shape == (8, 3) and len(m.faces) == 6
    assert np.allclose(m.verts.min(0), [0, 0, 0]) and np.allclose(
        m.verts.max(0), [1, 1, 1]
    )


def test_parallelogram3d_single_quad_face():
    m = Space3D(bounds=2).parallelogram([0, 0, 0], [1, 0, 0], [0, 1, 0]).primitives[-1]
    assert isinstance(m, P.Mesh3D)
    assert m.verts.shape == (4, 3) and m.faces == [[0, 1, 2, 3]]


@pytest.mark.parametrize("be", ["vedo", "mpl"])
def test_render_parallelepiped_nonblank(tmp_path, be):
    out = tmp_path / f"pp_{be}.png"
    Space3D(bounds=2, backend=be).parallelepiped(
        [0, 0, 0], [1.2, 0.2, 0], [0.3, 1.3, 0], [0.2, 0.4, 1.1]
    ).save(str(out))
    arr = np.asarray(Image.open(out).convert("RGB"))
    assert int((arr > 20).sum()) > 500


@pytest.mark.parametrize("be", ["vedo", "mpl"])
def test_render_3d_nonblank(tmp_path, be):
    out = tmp_path / f"s3_{be}.png"
    Space3D(bounds=2, backend=be).field([[0, -1, 0], [1, 0, 0], [0, 0, 1]], n=3).save(
        str(out)
    )
    arr = np.asarray(Image.open(out).convert("RGB"))
    assert int((arr > 20).sum()) > 500


# ── analytic anti-aliasing of the surface contour bands ───────────────
def _value(colors):
    import matplotlib.colors as mcolors

    return mcolors.rgb_to_hsv(np.clip(colors, 0, 1))[..., 2]


def _hard_edges(colors, thresh=0.25):
    """Count cells where brightness *jumps* between neighbours — the visible staircase.

    Deliberately not the mean of |diff|: anti-aliasing does not lower the *height* of a step edge,
    it gives the single cell straddling the edge an intermediate value proportional to its coverage.
    The mean over all cells therefore barely moves (0.0198 -> 0.0180) while the count of hard,
    un-graded edges collapses (557 -> 116). Only the second measures jaggedness.
    """
    return int((np.abs(np.diff(_value(colors), axis=1)) > thresh).sum())


F = lambda z: (z**2 - 1) / (z**2 + 1)  # noqa: E731


def test_landscape_anti_aliases_the_contour_bands_by_default():
    aliased = Space3D(bounds=2.5).landscape(F, res=140, aa=False).primitives[-1].colors
    smooth = Space3D(bounds=2.5).landscape(F, res=140, aa=True).primitives[-1].colors
    assert not np.allclose(aliased, smooth)
    assert _hard_edges(smooth) < _hard_edges(aliased) / 2  # the staircase collapses


def test_landscape_aa_only_touches_the_cells_straddling_a_band_edge():
    """Correct AA is local: interior-of-band cells already hold the exact average."""
    aliased = Space3D(bounds=2.5).landscape(F, res=140, aa=False).primitives[-1].colors
    smooth = Space3D(bounds=2.5).landscape(F, res=140, aa=True).primitives[-1].colors
    changed = (np.abs(aliased - smooth).max(axis=-1) > 0.02).mean()
    assert 0.005 < changed < 0.15  # a thin skin of edge cells, not a global blur


def test_landscape_aa_is_on_by_default():
    a = Space3D(bounds=2).landscape(F, res=60).primitives[-1].colors
    b = Space3D(bounds=2).landscape(F, res=60, aa=True).primitives[-1].colors
    assert np.allclose(a, b)


def test_landscape_aa_leaves_the_hue_alone():
    import matplotlib.colors as mcolors

    hues = [
        mcolors.rgb_to_hsv(
            np.clip(
                Space3D(bounds=2).landscape(F, res=60, aa=x).primitives[-1].colors, 0, 1
            )
        )[..., 0]
        for x in (False, True)
    ]
    assert np.allclose(*hues)  # AA filters brightness only


def test_landscape_kwargs_no_longer_clobber_the_scheme():
    """Passing `steps` used to drop into `colorize(w, **kw)` and silently lose `scheme`."""
    plane = (
        Space3D(bounds=2)
        .landscape(F, res=60, scheme="plane", steps=8)
        .primitives[-1]
        .colors
    )
    enhanced = (
        Space3D(bounds=2)
        .landscape(F, res=60, scheme="enhanced", steps=8)
        .primitives[-1]
        .colors
    )
    assert _value(plane).std() < 1e-9  # 'plane' = hue only, brightness flat at 1
    assert _value(enhanced).std() > 0.05  # 'enhanced' really has contour bands


@pytest.mark.parametrize("aa", [False, True])
def test_riemann_surfaces_accept_aa_and_stay_finite(aa):
    for surf in (
        Space3D(bounds=2).riemann_root(2, nr=20, ntheta=41, aa=aa),
        Space3D(bounds=2).riemann_log(sheets=2, nr=20, ntheta=41, aa=aa),
    ):
        colors = surf.primitives[-1].colors
        assert np.isfinite(colors).all() and colors.min() >= 0 and colors.max() <= 1


def test_riemann_log_aa_reduces_the_staircase():
    rough = (
        Space3D(bounds=2.5)
        .riemann_log(sheets=2, nr=60, ntheta=241, aa=False)
        .primitives[-1]
        .colors
    )
    smooth = (
        Space3D(bounds=2.5)
        .riemann_log(sheets=2, nr=60, ntheta=241, aa=True)
        .primitives[-1]
        .colors
    )
    assert _hard_edges(smooth) < _hard_edges(rough)


def test_riemann_log_uses_a_different_width_per_contour_family():
    """On the (u, v) grid the phase and modulus footprints genuinely differ; one width over-blurs.

    Forcing the holomorphic shortcut b = i*a onto this grid must give a different (wrong) result.
    """
    from mathviz.phase import _d_logw, colorize

    u = np.linspace(np.log(0.12), np.log(2.5), 60)
    v = np.linspace(0, 4 * np.pi, 241)
    U, V = np.meshgrid(u, v, indexing="ij")
    z = np.exp(U + 1j * V)
    a, b = _d_logw(z)
    correct = colorize(z, d_logw=(a, b))
    forced = colorize(z, d_logw=(a, 1j * a))
    assert not np.allclose(correct, forced)


def test_riemann_root_branch_point_does_not_produce_nans():
    """w = 0 at the branch point makes log w singular; the AA width must not leak a NaN."""
    colors = (
        Space3D(bounds=2)
        .riemann_root(2, nr=30, ntheta=61, aa=True)
        .primitives[-1]
        .colors
    )
    assert np.isfinite(colors).all()
