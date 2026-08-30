"""Backend registry and selection."""

from __future__ import annotations

from .base import Backend

_REGISTRY = {}
_DEFAULT = "mpl"


def _load(name: str) -> Backend:
    if name in ("mpl", "matplotlib"):
        from .mpl import MatplotlibBackend

        return MatplotlibBackend()
    if name in ("vedo", "vtk"):
        from .vedo_backend import VedoBackend

        return VedoBackend()
    raise ValueError(f"unknown backend {name!r}; choose 'mpl' or 'vedo'")


def get_backend(name: str | None = None) -> Backend:
    """Resolve a backend by name, caching one instance per name. ``None`` → the global default."""
    name = name or _DEFAULT
    if name not in _REGISTRY:
        _REGISTRY[name] = _load(name)
    return _REGISTRY[name]


def set_backend(name: str) -> None:
    """Set the global default backend ('mpl' or 'vedo')."""
    global _DEFAULT
    _load(name)  # validate eagerly
    _DEFAULT = "mpl" if name in ("mpl", "matplotlib") else "vedo"


# ── interactivity (currently honored by the vedo backend) ─────────────────────────
_INTERACTIVE = (
    False  # False | True | "auto" (auto → interactive only inside a notebook)
)
_VEDO_DISPLAY = (
    "k3d"  # vedo's live display backend: 'k3d' | 'trame' | 'ipyvtklink' | ...
)


def in_notebook() -> bool:
    """True when running inside a Jupyter/IPython kernel (as opposed to a script / headless run)."""
    try:
        from IPython import get_ipython

        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def set_interactive(value=True) -> None:
    """Global default: a static image (``False``), a live widget (``True``), or ``"auto"``
    (live only in a notebook). Honored by the vedo backend, whose static path is PNG; the
    matplotlib backend is always static and chooses SVG or PNG via ``format``."""
    global _INTERACTIVE
    _INTERACTIVE = value


def set_vedo_display(name: str) -> None:
    """Choose vedo's live display backend ('k3d', 'trame', 'ipyvtklink', ...) for interactive rendering."""
    global _VEDO_DISPLAY
    _VEDO_DISPLAY = name


def resolve_interactive(interactive) -> bool:
    """Per-call override → global default → (for 'auto') notebook detection."""
    val = _INTERACTIVE if interactive is None else interactive
    return in_notebook() if val == "auto" else bool(val)


def resolve_vedo_display(name) -> str:
    return name or _VEDO_DISPLAY
