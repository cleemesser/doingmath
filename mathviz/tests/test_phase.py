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


def test_plain_hue_red_at_arg_zero():
    # w = 1 has arg 0 → hue 0 → pure red in the plain scheme
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


def test_enhanced_is_darker_than_plain():
    z = phase.domain(2.0, 80)
    plain = phase.colorize(z, phase_contours=False, modulus_contours=False)
    enhanced = phase.colorize(z, phase_contours=True, modulus_contours=True)
    # contours only ever darken (multiply value ≤ 1), same hue
    assert (enhanced <= plain + 1e-9).all()
    assert enhanced.mean() < plain.mean()


def test_phase_portrait_scheme_dispatch():
    plain = phase.phase_portrait(lambda z: z, extent=2, res=60, scheme="plain")
    enhanced = phase.phase_portrait(lambda z: z, extent=2, res=60, scheme="enhanced")
    assert plain.shape == enhanced.shape == (60, 60, 3)
    assert not np.allclose(plain, enhanced)


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        phase.phase_portrait(lambda z: z, scheme="rainbow")


def test_zeros_do_not_produce_nans():
    # (z² − 1) has zeros at ±1; the portrait must stay finite everywhere
    rgb = phase.phase_portrait(lambda z: z**2 - 1, extent=2, res=120, scheme="enhanced")
    assert np.isfinite(rgb).all()
