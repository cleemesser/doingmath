"""The push-forward machinery: matrices, complex functions, grid sampling, and pole handling."""

import numpy as np
import pytest

from mathviz import maps


def test_from_matrix_applies_linear_map():
    f = maps.from_matrix([[2, 1], [0, 3]])
    # columns of M are the images of the basis vectors
    assert np.allclose(f([[1, 0], [0, 1]]), [[2, 0], [1, 3]])


def test_from_matrix_row_vector_convention():
    R = np.array([[0, -1], [1, 0]])  # 90° rotation
    assert np.allclose(maps.from_matrix(R)([[1, 0]]), [[0, 1]])


def test_from_complex_square():
    f = maps.from_complex(lambda z: z**2)
    assert np.allclose(f([[0, 1]]), [[-1, 0]])  # i² = −1
    assert np.allclose(f([[2, 0]]), [[4, 0]])


def test_from_complex_pole_becomes_nan():
    f = maps.from_complex(lambda z: 1 / z)
    out = f([[0, 0], [1, 0]])
    assert not np.isfinite(out[0]).all()  # 1/0 → NaN
    assert np.allclose(out[1], [1, 0])  # 1/1 = 1


def test_as_pointmap_accepts_matrix_and_callable():
    assert np.allclose(maps.as_pointmap([[1, 0], [0, 1]])([[5, 7]]), [[5, 7]])
    g = lambda pts: np.asarray(pts) * 2
    assert maps.as_pointmap(g) is g


def test_domain_gridlines_structure():
    lines = maps.domain_gridlines(2.0, step=1.0, samples=50)
    ticks = np.arange(-2, 2 + 1e-9, 1.0)
    assert len(lines) == 2 * len(ticks)  # one horizontal + one vertical per tick
    assert all(ln.shape == (50, 2) for ln in lines)


def test_finite_runs_splits_at_nan():
    pts = np.array([[0, 0], [1, 1], [np.nan, 0], [2, 2], [3, 3]])
    runs = maps._finite_runs(pts)
    assert len(runs) == 2
    assert len(runs[0]) == 2 and len(runs[1]) == 2


def test_finite_runs_drops_singletons():
    pts = np.array([[0, 0], [np.nan, np.nan], [1, 1]])  # each finite piece length 1
    assert maps._finite_runs(pts) == []


def test_analytic_map_is_conformal():
    """A small circle maps to a near-circle under an analytic map (constant local scaling)."""
    z0, r = 0.7 + 0.4j, 1e-2
    c = z0 + r * np.exp(1j * np.linspace(0, 2 * np.pi, 200, endpoint=False))
    w = c**3
    d = np.abs(w - w.mean())
    assert (d.max() - d.min()) / d.mean() < 0.1
