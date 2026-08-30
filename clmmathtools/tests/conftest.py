"""Shared pytest fixtures/config for the clmmathtools suite.

Force matplotlib's non-interactive Agg backend so render tests never try to open a window.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest


@pytest.fixture
def circle():
    t = np.linspace(0, 2 * np.pi, 64)
    return np.c_[np.cos(t), np.sin(t)]


@pytest.fixture(params=["mpl", "vedo"])
def backend(request):
    """Every render test runs once per backend."""
    return request.param
