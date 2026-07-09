"""Frame rendering, matrix interpolation paths, holomorphic sweeps, and the ipywidgets scrubbers."""

import io

import numpy as np
import pytest

from mathviz import Plane
from mathviz import maps as mp
from mathviz.animate import (
    animate_matrix,
    classify_mobius,
    geodesic_path,
    homotopy,
    lerp_path,
    matrix_path,
    mobius,
    mobius_path,
    polar_path,
    real_logm,
    render_frames,
    render_grid,
    scrubber,
    scrubber2,
    to_gif,
    to_png,
)

ID = np.eye(2)
SHEAR = np.array([[1.0, 1.0], [0.0, 1.0]])
HALF_TURN = -np.eye(2)  # rotation by π
SCALE = np.array([[2.0, 0.0], [0.0, 0.5]])
REFLECT = np.array([[1.0, 0.0], [0.0, -1.0]])  # det < 0
KINDS = ["lerp", "polar", "geodesic"]


# ── frames ────────────────────────────────────────────────────────────
def test_to_png_returns_png_bytes(backend):
    data = to_png(Plane(extent=2, backend=backend).basis())
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_frames_samples_the_unit_interval():
    ts, pngs = render_frames(lambda t: Plane(extent=1), n=5)
    assert len(pngs) == 5
    assert ts[0] == 0.0 and ts[-1] == 1.0


def test_render_frames_passes_t_to_the_builder():
    seen = []
    render_frames(lambda t: (seen.append(t), Plane(extent=1))[1], n=4)
    assert seen == [0.0, pytest.approx(1 / 3), pytest.approx(2 / 3), 1.0]


def test_ping_pong_mirrors_the_interior_frames():
    """0 → 1 → back, without repeating the endpoints, so a looping Play has no stutter."""
    ts, pngs = render_frames(lambda t: Plane(extent=1), n=4, ping_pong=True)
    assert len(pngs) == len(ts) == 6  # 4 forward + 2 interior back
    assert list(ts) == pytest.approx([0, 1 / 3, 2 / 3, 1, 2 / 3, 1 / 3])


def test_render_frames_needs_at_least_two():
    with pytest.raises(ValueError, match="at least 2"):
        render_frames(lambda t: Plane(extent=1), n=1)


def test_to_gif_writes_an_animated_gif(tmp_path):
    # the frames must actually differ — PIL's GIF writer drops duplicate consecutive frames
    path = to_gif(
        lambda t: Plane(extent=1).basis().vector([t, t]), tmp_path / "a.gif", n=3
    )
    assert path.exists()
    from PIL import Image

    with Image.open(path) as im:
        assert im.format == "GIF" and im.n_frames == 3


# ── the real matrix logarithm ─────────────────────────────────────────
@pytest.mark.parametrize("M", [SHEAR, HALF_TURN, SCALE, -2 * ID, ID])
def test_real_logm_is_real_and_exponentiates_back(M):
    from scipy.linalg import expm

    L = real_logm(M)
    assert L.dtype.kind == "f"  # real, not complex
    assert np.allclose(expm(L), M, atol=1e-8)


def test_real_logm_handles_the_principal_branch_failure():
    """logm(-I) = iπI (complex); the real generator is the π-rotation [[0,-π],[π,0]]."""
    from scipy.linalg import logm

    assert not np.allclose(logm(HALF_TURN).imag, 0)  # principal branch is complex
    assert np.allclose(real_logm(HALF_TURN), [[0, -np.pi], [np.pi, 0]])


def test_real_logm_rejects_orientation_reversing():
    with pytest.raises(ValueError, match="det M"):
        real_logm(REFLECT)


def test_real_logm_rejects_negative_eigenvalues_without_a_real_log():
    """diag(-1,-2): det > 0, but one Jordan block per negative eigenvalue ⇒ no real log."""
    with pytest.raises(ValueError, match="no real logarithm"):
        real_logm(np.diag([-1.0, -2.0]))


# ── interpolation paths ───────────────────────────────────────────────
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("M", [SHEAR, HALF_TURN, SCALE])
def test_every_path_joins_the_identity_to_M(kind, M):
    at = matrix_path(M, kind)
    assert np.allclose(at(0.0), ID)
    assert np.allclose(at(1.0), M)


@pytest.mark.parametrize("kind", ["polar", "geodesic"])
@pytest.mark.parametrize("M", [SHEAR, HALF_TURN, SCALE])
def test_principled_paths_never_degenerate(kind, M):
    at = matrix_path(M, kind)
    dets = [np.linalg.det(at(t)) for t in np.linspace(0, 1, 21)]
    assert min(dets) > 1e-6


def test_lerp_collapses_through_the_zero_matrix_on_a_half_turn():
    """The whole reason polar/geodesic exist: the straight line hits det = 0 at t = 1/2."""
    at = lerp_path(HALF_TURN)
    assert np.allclose(at(0.5), np.zeros((2, 2)))
    assert np.linalg.det(at(0.5)) == pytest.approx(0.0)


def test_polar_and_geodesic_agree_on_a_pure_rotation():
    """For M = R (no stretch), the polar split is R·I, so the polar path IS exp(t·log R)."""
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert np.allclose(polar_path(HALF_TURN)(t), geodesic_path(HALF_TURN)(t))


def test_geodesic_is_a_one_parameter_subgroup():
    """M(s)·M(t) = M(s+t): the defining property of exp(t·L)."""
    at = geodesic_path(SHEAR)
    assert np.allclose(at(0.3) @ at(0.4), at(0.7))


def test_polar_path_rejects_a_reflection():
    with pytest.raises(ValueError, match="orientation-reversing"):
        polar_path(REFLECT)


def test_lerp_accepts_a_reflection():
    """lerp is the only path that can reach det < 0 — by passing through a singular matrix."""
    assert np.allclose(lerp_path(REFLECT)(1.0), REFLECT)


def test_matrix_path_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown kind"):
        matrix_path(SHEAR, "spline")


# ── the widget layer ──────────────────────────────────────────────────
def test_scrubber_builds_a_play_slider_over_cached_frames():
    import ipywidgets as widgets

    box = scrubber(lambda t: Plane(extent=1), n=4)
    play = box.children[1].children[0]
    slider = box.children[1].children[1]
    assert isinstance(box.children[0], widgets.Image)
    assert isinstance(play, widgets.Play) and isinstance(slider, widgets.IntSlider)
    assert play.max == slider.max == 3  # n - 1


def test_scrubber_slider_swaps_the_cached_frame_and_updates_the_readout():
    box = scrubber(lambda t: Plane(extent=1 + t), n=3, label="s")
    img, (_, slider, readout) = box.children[0], box.children[1].children
    first = img.value
    slider.value = 2
    assert img.value != first  # a different pre-rendered PNG
    assert readout.value == "s = 1.00"


def test_animate_matrix_returns_a_widget():
    box = animate_matrix(SHEAR, kind="geodesic", n=3, size=(120, 120))
    assert len(box.children) == 2


def test_animate_matrix_propagates_the_path_error():
    with pytest.raises(ValueError, match="orientation-reversing"):
        animate_matrix(REFLECT, kind="polar", n=2, size=(80, 80))


def test_to_png_does_not_display_anything(monkeypatch):
    """Frame rendering must stay offscreen — otherwise every frame spams the notebook."""
    import IPython.display as ipd

    monkeypatch.setattr(
        ipd, "display", lambda *a, **k: pytest.fail("displayed a frame")
    )
    assert to_png(Plane(extent=1)).startswith(b"\x89PNG")


def test_to_png_leaves_no_file_behind(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    to_png(Plane(extent=1))
    assert list(tmp_path.iterdir()) == []


def test_to_png_accepts_a_bytesio_via_save():
    buf = io.BytesIO()
    assert Plane(extent=1).save(buf) is buf and buf.getvalue()[:4] == b"\x89PNG"


# ── the fundamental operations (maps constructors) ────────────────────
PROJ = np.array([[1.0, 0.0], [0.0, 0.0]])  # det = 0


def test_rotation_is_orthogonal_with_unit_determinant():
    R = mp.rotation(np.pi / 3)
    assert np.allclose(R.T @ R, ID) and np.isclose(np.linalg.det(R), 1.0)


def test_scaling_defaults_to_uniform_and_squeeze_preserves_area():
    assert np.allclose(mp.scaling(2.0), np.diag([2.0, 2.0]))
    assert np.isclose(np.linalg.det(mp.scaling(3.0, 1 / 3)), 1.0)  # the squeeze


def test_shear_fixes_its_axis_and_is_not_diagonalizable():
    S = mp.shear(1.5)
    assert np.allclose(S @ [1, 0], [1, 0])  # the x-axis is fixed pointwise
    assert np.isclose(np.linalg.det(S), 1.0)
    assert np.linalg.matrix_rank(S - ID) == 1  # one eigendirection for eigenvalue 1


def test_shear_axis_y_is_the_transpose():
    assert np.allclose(mp.shear(0.7, axis="y"), mp.shear(0.7).T)


def test_shear_rejects_a_bad_axis():
    with pytest.raises(ValueError, match="axis must be"):
        mp.shear(1.0, axis="z")


@pytest.mark.parametrize("theta", [0.0, 0.4, np.pi / 3])
def test_reflection_is_an_involution_reversing_orientation(theta):
    F = mp.reflection(theta)
    assert np.allclose(F @ F, ID)  # F² = I
    assert np.isclose(np.linalg.det(F), -1.0)
    mirror = [np.cos(theta), np.sin(theta)]
    assert np.allclose(F @ mirror, mirror)  # the mirror line is fixed


@pytest.mark.parametrize("theta", [0.0, 0.4, np.pi / 3])
def test_projection_is_idempotent_and_singular(theta):
    P = mp.projection(theta)
    assert np.allclose(P @ P, P)  # P² = P
    assert np.linalg.matrix_rank(P) == 1
    line = [np.cos(theta), np.sin(theta)]
    assert np.allclose(P @ line, line)  # fixed on the line
    assert np.allclose(P @ [-np.sin(theta), np.cos(theta)], [0, 0])  # kills the normal


# ── only lerp reaches det <= 0 ────────────────────────────────────────
@pytest.mark.parametrize("kind", ["polar", "geodesic"])
@pytest.mark.parametrize("theta", [0.0, np.pi / 6, 1.1])
def test_principled_paths_reject_a_projection_at_any_angle(kind, theta):
    """det(projection(π/6)) is 1.6e-17, not 0 — an exact sign test would let it slip through."""
    with pytest.raises(ValueError, match="singular"):
        matrix_path(mp.projection(theta), kind)


@pytest.mark.parametrize("kind", ["polar", "geodesic"])
def test_principled_paths_reject_a_reflection(kind):
    with pytest.raises(ValueError, match="orientation-reversing"):
        matrix_path(mp.reflection(0.3), kind)


def test_tiny_but_well_conditioned_scaling_is_not_rejected():
    """scaling(1e-7) has det = 1e-14 yet is perfectly invertible — a det threshold would kill it."""
    at = matrix_path(mp.scaling(1e-7), "geodesic")
    assert np.allclose(at(1.0), mp.scaling(1e-7))


def test_lerp_reaches_a_projection_by_degenerating_at_the_endpoint():
    at = lerp_path(PROJ)
    dets = [np.linalg.det(at(t)) for t in (0.0, 0.5, 1.0)]
    assert dets == pytest.approx([1.0, 0.5, 0.0])  # only hits 0 at t = 1


# ── two-parameter composition ─────────────────────────────────────────
def test_render_grid_covers_the_unit_square():
    ss, tt, pngs = render_grid(lambda s, t: Plane(extent=1), n=(3, 4))
    assert (len(ss), len(tt)) == (3, 4)
    assert len(pngs) == 3 and len(pngs[0]) == 4
    assert ss[0] == tt[0] == 0.0 and ss[-1] == tt[-1] == 1.0


def test_scrubber2_has_two_independent_axes():
    import ipywidgets as widgets

    box = scrubber2(lambda s, t: Plane(extent=1 + s + t), n=(3, 4), labels=("a", "b"))
    img, row_a, row_b = box.children
    assert isinstance(img, widgets.Image)
    assert row_a.children[1].max == 2 and row_b.children[1].max == 3


def test_scrubber2_each_slider_swaps_the_frame_and_updates_its_own_readout():
    box = scrubber2(
        lambda s, t: Plane(extent=1 + s + 2 * t), n=(3, 3), labels=("a", "b")
    )
    img, row_a, row_b = box.children
    first = img.value
    row_b.children[1].value = 2  # move only the second axis
    assert img.value != first
    assert row_b.children[2].value == "b = 1.00"
    assert row_a.children[2].value == "a = 0.00"  # untouched


def test_composition_is_order_dependent():
    """The reason two sliders are interesting: R∘S ≠ S∘R."""
    R, S = mp.rotation(np.pi / 2), mp.shear(1.0)
    assert not np.allclose(R @ S, S @ R)


# ── holomorphic sweeps ────────────────────────────────────────────────
def test_homotopy_interpolates_between_identity_and_g():
    z = np.array([0.3 + 0.4j, -1.1 + 0.2j])
    g = lambda w: w**2  # noqa: E731
    assert np.allclose(homotopy(g)(0.0)(z), z)
    assert np.allclose(homotopy(g)(1.0)(z), g(z))
    assert np.allclose(homotopy(g)(0.5)(z), 0.5 * z + 0.5 * g(z))


def test_homotopy_stays_holomorphic():
    """A convex combination of holomorphic maps is holomorphic: check Cauchy–Riemann numerically."""
    gt = homotopy(lambda w: w**2)(0.4)
    z0, h = 0.3 + 0.5j, 1e-6
    d_dx = (gt(z0 + h) - gt(z0)) / h
    d_dy = (gt(z0 + 1j * h) - gt(z0)) / (1j * h)
    assert np.allclose(d_dx, d_dy, atol=1e-4)


def test_homotopy_respects_a_custom_base():
    z = np.array([0.5 + 0.1j])
    gt = homotopy(lambda w: w**2, base=lambda w: 1 / w)
    assert np.allclose(gt(0.0)(z), 1 / z)


def test_mobius_applies_the_fractional_linear_map():
    z = np.array([0.3 + 0.4j, 2.0 + 0.0j])
    A = np.array([[1, 2], [3, 4]], dtype=complex)
    assert np.allclose(mobius(A)(z), (z + 2) / (3 * z + 4))


def test_mobius_sends_its_pole_to_infinity():
    A = np.array([[1, 0], [1, 0]], dtype=complex)  # pole at z = 0
    assert not np.isfinite(mobius(A)(np.array([0.0 + 0j]))[0])


def test_mobius_is_scale_invariant():
    """A and λA give the same map — Möbius transformations live in PSL(2, ℂ)."""
    z = np.array([0.3 + 0.4j])
    A = np.array([[1, 2], [3, 4]], dtype=complex)
    assert np.allclose(mobius(A)(z), mobius(2.5j * A)(z))


@pytest.mark.parametrize(
    "A,expected",
    [
        (np.eye(2), "identity"),
        (np.diag([np.exp(0.7j), np.exp(-0.7j)]), "elliptic"),  # z ↦ e^{iθ}z
        (np.array([[1, 1], [0, 1]]), "parabolic"),  # z ↦ z + 1
        (np.diag([np.sqrt(2), 1 / np.sqrt(2)]), "hyperbolic"),  # z ↦ 2z
        (np.diag([np.sqrt(2j), 1 / np.sqrt(2j)]), "loxodromic"),  # z ↦ 2i·z
    ],
)
def test_classify_mobius_by_trace_squared(A, expected):
    assert classify_mobius(np.asarray(A, dtype=complex)) == expected


def test_classify_mobius_rejects_a_degenerate_matrix():
    with pytest.raises(ValueError, match="not a Möbius"):
        classify_mobius(np.array([[1, 1], [1, 1]], dtype=complex))


def test_mobius_path_joins_the_identity_to_A():
    z = np.array([0.3 + 0.4j, -0.8 + 1.2j])
    A = np.array([[2, 1], [1, 1]], dtype=complex)  # det = 1, hyperbolic
    at = mobius_path(A)
    assert np.allclose(at(0.0)(z), z)
    assert np.allclose(at(1.0)(z), mobius(A)(z))


def test_mobius_path_is_a_one_parameter_subgroup():
    """g_s ∘ g_t = g_{s+t} — the same structure as geodesic_path, one field up."""
    z = np.array([0.3 + 0.4j, -0.8 + 1.2j])
    at = mobius_path(np.array([[2, 1], [1, 1]], dtype=complex))
    assert np.allclose(at(0.3)(at(0.4)(z)), at(0.7)(z))


def test_mobius_path_of_a_parabolic_is_a_translation_flow():
    """z ↦ z + 1 generated by t: g_t(z) = z + t."""
    z = np.array([0.3 + 0.4j])
    at = mobius_path(np.array([[1, 1], [0, 1]], dtype=complex))
    assert np.allclose(at(0.5)(z), z + 0.5)
