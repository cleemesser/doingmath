"""Wegert phase portraits: the domain-coloring math (backend-independent)."""

import numpy as np
import pytest

from mathviz import phase


def test_domain_shape_and_range():
    z = phase.domain(2.0, res=50)
    assert z.shape == (50, 50)
    assert np.isclose(z.real.min(), -2) and np.isclose(z.real.max(), 2)


def test_colorize_shape_and_range():
    rgb = phase.colorize(phase.domain(1.0, 40))
    assert rgb.shape == (40, 40, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0


def test_plane_hue_red_at_arg_zero():
    # w = 1 has arg 0 → hue 0 → pure red in the plane scheme
    rgb = phase.colorize(
        np.array([[1 + 0j]]), phase_contours=False, modulus_contours=False
    )
    assert rgb[0, 0] == pytest.approx([1.0, 0.0, 0.0])


def test_quarter_turn_hue():
    # w = i has arg π/2 → hue 0.25
    rgb = phase.colorize(np.array([[1j]]), phase_contours=False, modulus_contours=False)
    # not red, and green channel dominates the red at hue 0.25
    assert not np.allclose(rgb[0, 0], [1, 0, 0])


def test_poles_render_white():
    rgb = phase.colorize(np.array([[np.inf + 0j, 1 + 0j]]))
    assert rgb[0, 0] == pytest.approx([1.0, 1.0, 1.0])  # pole → white


def test_enhanced_is_darker_than_plane():
    z = phase.domain(2.0, 80)
    plane = phase.colorize(z, phase_contours=False, modulus_contours=False)
    enhanced = phase.colorize(z, phase_contours=True, modulus_contours=True)
    # contours only ever darken (multiply value ≤ 1), same hue
    assert (enhanced <= plane + 1e-9).all()
    assert enhanced.mean() < plane.mean()


def test_phase_portrait_scheme_dispatch():
    plane = phase.phase_portrait(lambda z: z, extent=2, res=60, scheme="plane")
    enhanced = phase.phase_portrait(lambda z: z, extent=2, res=60, scheme="enhanced")
    assert plane.shape == enhanced.shape == (60, 60, 3)
    assert not np.allclose(plane, enhanced)


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        phase.phase_portrait(lambda z: z, scheme="rainbow")


def test_zeros_do_not_produce_nans():
    # (z² − 1) has zeros at ±1; the portrait must stay finite everywhere
    rgb = phase.phase_portrait(lambda z: z**2 - 1, extent=2, res=120, scheme="enhanced")
    assert np.isfinite(rgb).all()


# ── analytic anti-aliasing of the contour bands ───────────────────────
def test_frac_integral_is_the_antiderivative_of_frac():
    xs = np.linspace(-3, 3, 20001)
    num = np.gradient(phase._frac_integral(xs), xs[1] - xs[0])
    inner = np.abs(xs - np.round(xs)) > 1e-2  # skip the kinks at integers
    assert np.allclose(num[inner], xs[inner] % 1.0, atol=1e-2)


@pytest.mark.parametrize("t", [-2.3, 0.0, 0.37, 1.5, 4.9])
@pytest.mark.parametrize("width", [1e-3, 0.25, 1.0, 3.0])
def test_sawtooth_box_filter_equals_numerical_integration(t, width):
    u = np.linspace(t - width / 2, t + width / 2, 200001)
    brute = 0.65 + 0.35 * np.trapezoid(u % 1.0, u) / width
    assert phase._sawtooth(t, 0.65, width) == pytest.approx(brute, abs=1e-5)


def test_sawtooth_without_width_is_the_plain_point_sample():
    t = np.linspace(-2, 2, 50)
    assert np.allclose(phase._sawtooth(t, 0.65), 0.65 + 0.35 * (t % 1.0))


def test_sawtooth_narrow_width_reduces_to_the_point_sample():
    t = np.array([0.3, 1.7, -0.2])
    assert np.allclose(phase._sawtooth(t, 0.65, 1e-12), phase._sawtooth(t, 0.65))


def test_sawtooth_wide_width_fades_to_the_flat_mean():
    """Bands finer than a pixel must average out, not moiré. Mean of the ramp is low+(1-low)/2."""
    assert phase._sawtooth(np.array([0.3]), 0.65, np.inf) == pytest.approx(0.825)
    assert phase._sawtooth(np.array([0.3]), 0.65, 1e5) == pytest.approx(0.825, abs=1e-4)


def test_sawtooth_survives_non_finite_width():
    assert np.isfinite(
        phase._sawtooth(np.array([0.3, 0.9]), 0.65, np.array([np.nan, np.inf]))
    ).all()


def test_sawtooth_stays_within_the_ramp_range():
    v = phase._sawtooth(np.linspace(-5, 5, 500), 0.65, 0.4)
    assert v.min() >= 0.65 - 1e-12 and v.max() <= 1.0 + 1e-12


def test_d_logw_recovers_the_logarithmic_derivative():
    """d(log w) per cell, from the samples alone. For f = e^{kz} it is exactly k·pixel along x."""
    extent, res, k = 1.0, 401, 3.0
    w = np.exp(k * phase.domain(extent, res))
    pixel = 2 * extent / (res - 1)
    a, b = (d[2:-2, 2:-2] for d in phase._d_logw(w))  # trim one-sided borders
    assert np.allclose(a, k * pixel, rtol=1e-3)  # along x: f'/f = k
    assert np.allclose(b, 1j * k * pixel, rtol=1e-3)  # along y: i·f'/f


def test_d_logw_second_axis_is_i_times_the_first_on_a_holomorphic_grid():
    """b = i·a is what makes both contour families share the width |f'/f| in 2D."""
    w = np.asarray((lambda z: (z**2 - 1) / (z**2 + 1))(phase.domain(2.0, 300)), complex)
    a, b = phase._d_logw(w)
    ok = np.zeros_like(w, bool)
    ok[3:-3, 3:-3] = True
    ok &= np.isfinite(a) & np.isfinite(b) & (np.abs(a) < 0.05) & (np.abs(a) > 1e-9)
    assert ok.sum() > 1000
    assert np.abs(b[ok] - 1j * a[ok]).max() / np.abs(a[ok]).max() < 1e-2


def test_riemann_log_grid_separates_the_two_contour_widths():
    """On the (u, v) grid of log z, axis 1 carries phase only and axis 0 modulus only.

    Using one axis for both (the holomorphic shortcut) would over-blur the modulus contours.
    """
    u = np.linspace(np.log(0.12), np.log(3), 60)
    v = np.linspace(0, 6 * np.pi, 241)
    U, V = np.meshgrid(u, v, indexing="ij")
    a, b = phase._d_logw(np.exp(U + 1j * V))
    inner = (slice(2, -2), slice(2, -2))
    assert np.abs(a[inner].real).max() < 1e-6  # axis 1 (v): purely imaginary
    assert np.abs(b[inner].imag).max() < 1e-6  # axis 0 (u): purely real
    width_phase = np.hypot(a.imag, b.imag)[inner]
    width_mod = np.hypot(a.real, b.real)[inner]
    assert width_phase.mean() > 1.3 * width_mod.mean()  # genuinely different footprints


def test_aa_smooths_the_contour_edges_without_moving_the_hue():
    f = lambda z: (z**2 - 1) / (z**2 + 1)  # noqa: E731
    raw = phase.phase_portrait(f, extent=2, res=200, aa=False)
    smooth = phase.phase_portrait(f, extent=2, res=200, aa=True)
    import matplotlib.colors as mcolors

    hsv_raw, hsv_aa = mcolors.rgb_to_hsv(raw), mcolors.rgb_to_hsv(smooth)
    assert np.allclose(
        hsv_raw[..., 0], hsv_aa[..., 0]
    )  # hue untouched: AA only filters brightness
    assert not np.allclose(raw, smooth)  # but the value channel did change
    # the band edges are the high-frequency content; AA must reduce it
    edge = lambda a: np.abs(np.diff(a[..., 2], axis=1)).mean()  # noqa: E731
    assert edge(hsv_aa) < edge(hsv_raw)


def test_aa_kills_moire_where_contours_are_sub_pixel():
    """exp(1/z) packs bands below a pixel near 0; brightness must go flat, not noisy."""
    f = lambda z: np.exp(1 / z)  # noqa: E731
    extent, res = 0.35, 200
    w = np.asarray(f(phase.domain(extent, res)), dtype=complex)
    a, b = phase._d_logw(w)
    width = np.hypot(a.imag, b.imag) * 6 / phase.TWO_PI
    sub = np.isfinite(w) & (width > 1.0)
    assert sub.sum() > 50  # the test region is genuinely sub-pixel

    import matplotlib.colors as mcolors

    raw = mcolors.rgb_to_hsv(phase.phase_portrait(f, extent=extent, res=res, aa=False))[
        ..., 2
    ]
    aa = mcolors.rgb_to_hsv(phase.phase_portrait(f, extent=extent, res=res, aa=True))[
        ..., 2
    ]
    assert aa[sub].std() < raw[sub].std() / 5  # an order of magnitude less noise
    assert aa[sub].mean() == pytest.approx(
        0.825**2, abs=0.02
    )  # two sawtooths, each averaging to 0.825


def test_phase_portrait_aa_is_on_by_default():
    f = lambda z: (z**2 - 1) / (z**2 + 1)  # noqa: E731
    assert np.allclose(
        phase.phase_portrait(f, extent=2, res=80),
        phase.phase_portrait(f, extent=2, res=80, aa=True),
    )


def test_plane_scheme_is_unaffected_by_aa():
    """No contours, no bands to filter — the hue-only portrait is identical either way."""
    f = lambda z: z**2 - 1  # noqa: E731
    a = phase.phase_portrait(f, extent=2, res=60, scheme="plane", aa=True)
    b = phase.phase_portrait(f, extent=2, res=60, scheme="plane", aa=False)
    assert np.allclose(a, b)


def test_aa_portrait_has_no_nans_at_zeros_and_poles():
    f = lambda z: (z**2 - 1) / (z**2 + 1)  # noqa: E731
    assert np.isfinite(phase.phase_portrait(f, extent=2, res=120)).all()


def test_plane_phase_portrait_defaults_res_to_the_view_pixel_width():
    from mathviz import Plane

    p = Plane(extent=1, size=(320, 320), grid=False, axes=False).phase_portrait(
        lambda z: z
    )
    raster = p.primitives[-1]
    assert raster.rgb.shape == (320, 320, 3)


def test_colorize_accepts_d_logw_and_dims_nothing_it_should_not():
    w = np.asarray((lambda z: z**2 - 1)(phase.domain(2.0, 60)), dtype=complex)
    rgb = phase.colorize(w, d_logw=phase._d_logw(w))
    assert np.isfinite(rgb).all() and rgb.min() >= 0 and rgb.max() <= 1


def test_sawtooth_handles_infinite_t_at_zeros_without_warnings():
    """log|f| = -inf at an exact zero; the closed form must not evaluate inf - inf."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        v = phase._sawtooth(
            np.array([-np.inf, np.inf, 0.3]), 0.65, np.array([0.5, 0.5, 0.5])
        )
    assert np.isfinite(v).all()
    assert v[0] == pytest.approx(0.825) and v[1] == pytest.approx(
        0.825
    )  # flat mid-tone


def test_phase_portrait_at_an_exact_zero_is_warning_free():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        rgb = phase.phase_portrait(
            lambda z: z, extent=1.0, res=101
        )  # z = 0 is a sample point
    assert np.isfinite(rgb).all()
