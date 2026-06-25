# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: doingmath (3.14.3)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import k3d

# Named palette — keep illustrations visually consistent.
GREY   = 0x888888
RED    = 0xff3333
GREEN  = 0x33ff33
BLUE   = 0x3388ff
ORANGE = 0xff9933
YELLOW = 0xffdd33


class Plane2D:
    """A top-down k3d view of the x-y plane for illustrating 2D linear algebra.

    Everything lives at z = 0, so all helpers take 2D inputs (x, y) and the
    class handles embedding into 3D and the fixed top-down camera. Add new
    methods here to illustrate vectors, matrix actions, spans, etc.
    """

    def __init__(self, extent=5, eye_height=7, grid_color=GREY, grid_opacity=0.25):
        self.extent = extent
        self.plot = k3d.plot(grid_visible=False, camera_auto_fit=False)
        self._add_grid(grid_color, grid_opacity)
        # Top-down view: eye on +z, looking at origin, +y is screen-up.
        # Must be set before display() or k3d's first render overrides it.
        self.plot.camera = [0, 0, eye_height, 0, 0, 0, 0, 1, 0]

    # ── internals ────────────────────────────────────────────
    @staticmethod
    def _pad(xy):
        """Embed an (N, 2) array of plane points into (N, 3) at z = 0."""
        xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
        return np.column_stack([xy, np.zeros(len(xy), dtype=np.float32)])

    def _add_grid(self, color, opacity):
        N = self.extent
        ticks = np.arange(-N, N + 1, dtype=np.float32)
        verts, segs = [], []
        for t in ticks:
            i = len(verts)
            verts += [[t, -N, 0], [t, N, 0]]      # vertical   x = t
            verts += [[-N, t, 0], [N, t, 0]]      # horizontal y = t
            segs  += [[i, i + 1], [i + 2, i + 3]]
        self.grid = k3d.lines(
            np.array(verts, dtype=np.float32),
            np.array(segs, dtype=np.uint32),
            indices_type='segment',
            color=color, opacity=opacity, width=0.01, shader='simple',
        )
        self.plot += self.grid

    # ── drawing helpers ──────────────────────────────────────
    def add_vectors(self, vecs, origins=None, colors=None, labels=None,
                    head_size=2.0, width=0.03):
        """Draw one or more 2D vectors as arrows.

        vecs    : (N, 2) array of vector components.
        origins : (N, 2) tails; defaults to the origin for all.
        colors  : list of N hex colors (one per arrow).
        labels  : list of N strings.
        """
        v3 = self._pad(vecs)
        n = len(v3)
        o3 = np.zeros((n, 3), dtype=np.float32) if origins is None else self._pad(origins)
        if colors is None:
            colors = [RED] * n
        # k3d wants TWO color entries per arrow (tail vertex, head vertex).
        pair_colors = [c for c in colors for _ in range(2)]
        obj = k3d.vectors(
            o3, v3,
            colors=pair_colors, head_size=head_size, line_width=width,
            labels=list(labels) if labels else [],
        )
        self.plot += obj
        return obj

    def add_vector(self, vec, origin=(0, 0), color=RED, label=None, **kw):
        """Draw a single 2D vector (convenience wrapper around add_vectors)."""
        return self.add_vectors(
            [vec], origins=[origin], colors=[color],
            labels=[label] if label else None, **kw,
        )

    def add_basis(self, colors=(RED, GREEN), labels=('e1', 'e2')):
        """Draw the standard basis e1, e2."""
        return self.add_vectors([[1, 0], [0, 1]],
                                colors=list(colors), labels=list(labels))

    def display(self):
        self.plot.display()
        return self.plot



# %%
# Reproduce the original figure: faint grid + standard basis, top-down.
view = Plane2D(extent=5, eye_height=7)
view.add_basis()
view.display()


# %%
