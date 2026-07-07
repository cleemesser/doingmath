"""The backend interface.

A backend turns a ``Scene`` (a ``View`` plus a list of primitives) into pixels — either displayed
inline in a notebook or written to a file. Concrete backends (matplotlib, vedo) implement ``render``.
"""

from __future__ import annotations

from ..primitives import Scene


class Backend:
    name = "base"

    def render(self, scene: Scene, *, save: str | None = None):
        """Draw the scene. If ``save`` is a path, write an image there; otherwise display inline."""
        raise NotImplementedError
