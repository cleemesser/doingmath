"""Backend registry: resolution, caching, default selection, and error handling."""

import pytest

from mathviz import backends


def test_get_mpl_and_vedo():
    assert backends.get_backend("mpl").name == "mpl"
    assert backends.get_backend("vedo").name == "vedo"


def test_aliases():
    assert backends.get_backend("matplotlib").name == "mpl"
    assert backends.get_backend("vtk").name == "vedo"


def test_backend_instances_are_cached():
    assert backends.get_backend("mpl") is backends.get_backend("mpl")


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        backends.get_backend("svg")


def test_set_backend_changes_default(monkeypatch):
    monkeypatch.setattr(backends, "_DEFAULT", "mpl")
    backends.set_backend("vedo")
    assert backends.get_backend().name == "vedo"
    backends.set_backend("mpl")
    assert backends.get_backend().name == "mpl"


def test_set_backend_rejects_unknown():
    with pytest.raises(ValueError):
        backends.set_backend("nope")
