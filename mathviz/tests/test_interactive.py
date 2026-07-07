"""Interactive-vs-static resolution, notebook detection, and the live vedo paths."""

from mathviz import Plane, backends
from mathviz.backends.vedo_backend import VedoBackend


def test_in_notebook_false_in_script():
    assert backends.in_notebook() is False


def test_resolve_interactive_default_and_override():
    assert backends.resolve_interactive(None) is False  # default: static
    assert backends.resolve_interactive(True) is True
    assert backends.resolve_interactive(False) is False


def test_resolve_auto_follows_notebook_detection(monkeypatch):
    monkeypatch.setattr(backends, "in_notebook", lambda: True)
    assert backends.resolve_interactive("auto") is True
    monkeypatch.setattr(backends, "in_notebook", lambda: False)
    assert backends.resolve_interactive("auto") is False


def test_set_interactive_global(monkeypatch):
    monkeypatch.setattr(backends, "_INTERACTIVE", False)
    backends.set_interactive(True)
    assert backends.resolve_interactive(None) is True
    backends.set_interactive(False)
    assert backends.resolve_interactive(None) is False


def test_resolve_vedo_display(monkeypatch):
    monkeypatch.setattr(backends, "_VEDO_DISPLAY", "k3d")
    assert backends.resolve_vedo_display(None) == "k3d"
    assert backends.resolve_vedo_display("trame") == "trame"


def test_interactive_in_notebook_returns_a_live_widget(monkeypatch):
    # in a notebook, interactive → an inline k3d widget object (not None)
    monkeypatch.setattr(backends, "in_notebook", lambda: True)
    widget = (
        Plane(extent=1, backend="vedo")
        .basis()
        .display(interactive=True, vedo_display="k3d")
    )
    assert widget is not None


def test_interactive_in_script_routes_to_native_window(monkeypatch):
    # in a plain script, interactive → the native VTK window path (which would block on a real display),
    # so we stub it and just assert it's the branch taken (no window opened in the test).
    monkeypatch.setattr(backends, "in_notebook", lambda: False)
    monkeypatch.setattr(VedoBackend, "_live_window", lambda self, o, v, k: "WINDOW")
    result = Plane(extent=1, backend="vedo").basis().display(interactive=True)
    assert result == "WINDOW"


def test_static_display_does_not_open_a_window():
    # the default (non-interactive) path never routes to a live window; it returns None (displays a PNG)
    assert Plane(extent=1, backend="vedo").basis().display() is None
