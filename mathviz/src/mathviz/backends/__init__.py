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
