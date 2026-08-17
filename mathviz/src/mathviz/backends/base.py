"""The backend interface.

A backend turns a ``Scene`` (a ``View`` plus a list of primitives) into pixels — either displayed
inline in a notebook or written to a file. Concrete backends (matplotlib, vedo) implement ``render``.
"""

from __future__ import annotations

from ..primitives import Scene


class Backend:
    name = "base"
    # should this really be
    # render to {file, inline in notebook, or to a window outside the running noteobook}
    def render(
        self,
        scene: Scene,
        *,
        save: str | None = None,
        format: str | None = None,
        interactive=None,
        vedo_display=None,
    ):
        """Draw the scene. If ``save`` is a path, write an image there; otherwise display inline.

        ``format`` ("png", "svg", ...) overrides the format inferred from the ``save`` filename —
        needed when ``save`` is an in-memory buffer, which carries no suffix to infer from.

        ``interactive`` / ``vedo_display`` request a live widget instead of a static image (honored by
        the vedo backend; ignored by static backends).
        """
        raise NotImplementedError
