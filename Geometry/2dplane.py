# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: doingmath (3.14.3.final.0)
#     language: python
#     name: python3
# ---

# %%

# %%

# Named palette — keep illustrations visually consistent.
GREY   = 0x888888
RED    = 0xff3333
GREEN  = 0x33ff33
BLUE   = 0x3388ff
ORANGE = 0xff9933
YELLOW = 0xffdd33



# %%
import numpy as np
import k3d


class Plane2D:
    """A top-down k3d view of the x-y plane for illustrating 2D linear algebra.

    Everything lives at z = 0, so all helpers take 2D inputs (x, y) and the
    class handles embedding into 3D and the fixed top-down camera. Add new
    methods here to illustrate vectors, matrix actions, spans, etc.

    Labels are rendered by k3d with KaTeX, so they accept LaTeX math:
    'e_1' -> a subscript, 'e_{12}' for multi-char, 'v\\,'... etc.
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

    @staticmethod
    def _grid_segments(N):
        """Vertices (M, 2) and segment indices (K, 2) for an N x N gridline mesh.

        Shared by the faint background grid and the warped grid, so both have
        identical topology — applying a matrix just moves the vertices.
        """
        ticks = np.arange(-N, N + 1, dtype=np.float32)
        verts, segs = [], []
        for t in ticks:
            i = len(verts)
            verts += [[t, -N], [t, N]]      # vertical   x = t
            verts += [[-N, t], [N, t]]      # horizontal y = t
            segs  += [[i, i + 1], [i + 2, i + 3]]
        return (np.array(verts, dtype=np.float32),
                np.array(segs, dtype=np.uint32))

    def _add_grid(self, color, opacity):
        verts, segs = self._grid_segments(self.extent)
        self.grid = k3d.lines(
            self._pad(verts), segs,
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
        labels  : list of N LaTeX strings (KaTeX-rendered).
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

    def add_basis(self, colors=(RED, GREEN), labels=('e_1', 'e_2')):
        """Draw the standard basis e1, e2 (labels are LaTeX -> subscripts)."""
        return self.add_vectors([[1, 0], [0, 1]],
                                colors=list(colors), labels=list(labels))

    def apply_matrix(self, M, color=ORANGE, opacity=0.6, width=0.015,
                     basis=True, basis_colors=(RED, GREEN),
                     basis_labels=(r'M\,e_1', r'M\,e_2')):
        """Draw the image of the grid (and basis) under the 2x2 matrix M.

        Each background gridline vertex p is mapped to M @ p, so the square
        grid warps into the parallelogram lattice that M defines. The columns
        of M are exactly the images of e1 and e2, drawn as arrows by default.

        Returns (warped_grid_obj, basis_obj | None).
        """
        M = np.asarray(M, dtype=np.float32).reshape(2, 2)
        verts, segs = self._grid_segments(self.extent)
        # Row-vector convention: (M @ p) for every point p is verts @ M.T.
        warped = verts @ M.T
        grid_obj = k3d.lines(
            self._pad(warped), segs,
            indices_type='segment',
            color=color, opacity=opacity, width=width, shader='simple',
        )
        self.plot += grid_obj

        basis_obj = None
        if basis:
            # Columns of M = M @ e1, M @ e2 = images of the basis vectors.
            basis_obj = self.add_vectors(
                [M[:, 0], M[:, 1]],
                colors=list(basis_colors), labels=list(basis_labels),
            )
        return grid_obj, basis_obj

    def display(self):
        self.plot.display()
        return self.plot



# %%
# A matrix acting on the plane: faint original grid + warped grid + image basis.
M = [[1.0, 1.0],
     [0.0, 1.0]]          # a shear; try [[2,0],[0,0.5]] (scale) or a rotation

view = Plane2D(extent=5, eye_height=7)
view.apply_matrix(M)       # orange warped lattice + columns of M as arrows
view.display()


# %% [markdown]
# ## vedo port
#
# Same scene, rendered with [vedo](https://vedo.embl.es/) (VTK) instead of k3d. Identical method
# names, plus a `save(path)` for **headless** static PNG/PDF export — the thing k3d can't do
# without a live browser. Built-in `Grid`/`Arrow` primitives remove most of the k3d boilerplate
# (no manual segment arrays, no two-colors-per-arrow).
#

# %%
import numpy as np 
from vedo import Grid, Lines, Arrow, Latex, Plotter
import vedo.settings
vedo.settings.default_backend = "ipyvtklink" # vtk or k3d or 2d or ipyvtklink or trame
# Reuse the same palette constants (GREY, RED, ...) defined in the k3d cell.


class Plane2DVedo:
    """A top-down vedo (VTK) view of the x-y plane — a drop-in twin of Plane2D.

    Same method names as the k3d version (add_vectors / add_vector / add_basis /
    apply_matrix / display) plus save(path) for headless PNG/PDF export. Objects
    accumulate in self.objects; display() opens an interactive window and
    save() renders offscreen. Labels are LaTeX via mathtext ('e_1' -> subscript).
    """

    def __init__(self, extent=5, eye_height=7, grid_color=GREY, grid_opacity=0.3,
                 bg='black'):
        self.extent = extent
        self.bg = bg
        self.objects = []
        # Top-down camera: same [eye, target, up] as the k3d version.
        self.camera = {'pos': (0, 0, eye_height),
                       'focal_point': (0, 0, 0),
                       'viewup': (0, 1, 0)}
        ticks = np.arange(-extent, extent + 1)
        self.grid = (Grid(s=(ticks, ticks)).wireframe(True)
                     .c(self._hex(grid_color)).alpha(grid_opacity))
        self.objects.append(self.grid)

    # ── internals ────────────────────────────────────────────
    @staticmethod
    def _hex(c):
        """Accept either a 0xRRGGBB int (our palette) or a vedo color string."""
        return c if isinstance(c, str) else f'#{c:06x}'

    @staticmethod
    def _pad(xy):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        return np.column_stack([xy, np.zeros(len(xy))])

    @staticmethod
    def _grid_segments(N):
        ticks = np.arange(-N, N + 1, dtype=float)
        verts, segs = [], []
        for t in ticks:
            i = len(verts)
            verts += [[t, -N], [t, N]]      # vertical   x = t
            verts += [[-N, t], [N, t]]      # horizontal y = t
            segs  += [[i, i + 1], [i + 2, i + 3]]
        return np.array(verts), np.array(segs)

    # ── drawing helpers ──────────────────────────────────────
    def add_vectors(self, vecs, origins=None, colors=None, labels=None,
                    shaft_radius=0.02, head_radius=0.06, head_length=0.18,
                    label_size=0.4):
        """Draw 2D vectors as arrows (one Arrow per vector, so colors are easy)."""
        v = np.asarray(vecs, dtype=float).reshape(-1, 2)
        n = len(v)
        o = (np.zeros((n, 2)) if origins is None
             else np.asarray(origins, dtype=float).reshape(-1, 2))
        if colors is None:
            colors = [RED] * n
        objs = []
        for i in range(n):
            start = self._pad(o[i])[0]
            end = self._pad(o[i] + v[i])[0]
            col = self._hex(colors[i])
            arrow = Arrow(start, end, c=col, shaft_radius=shaft_radius,
                          head_radius=head_radius, head_length=head_length)
            self.objects.append(arrow)
            objs.append(arrow)
            if labels and labels[i]:
                # Offset the label just past the arrow tip.
                lab = Latex(labels[i], pos=end + np.array([0.15, 0.15, 0]),
                            s=label_size, c=col)
                self.objects.append(lab)
        return objs

    def add_vector(self, vec, origin=(0, 0), color=RED, label=None, **kw):
        return self.add_vectors([vec], origins=[origin], colors=[color],
                                labels=[label] if label else None, **kw)

    def add_basis(self, colors=(RED, GREEN), labels=('e_1', 'e_2')):
        return self.add_vectors([[1, 0], [0, 1]],
                                colors=list(colors), labels=list(labels))

    def apply_matrix(self, M, color=ORANGE, opacity=0.6, lw=2, basis=True,
                     basis_colors=(RED, GREEN),
                     basis_labels=(r'M\,e_1', r'M\,e_2')):
        """Draw the image of the grid (and basis) under the 2x2 matrix M."""
        M = np.asarray(M, dtype=float).reshape(2, 2)
        verts, segs = self._grid_segments(self.extent)
        warped = verts @ M.T          # (M @ p) for every point, row-vector form
        starts = self._pad(warped[segs[:, 0]])
        ends = self._pad(warped[segs[:, 1]])
        grid_obj = Lines(starts, ends, c=self._hex(color), lw=lw, alpha=opacity)
        self.objects.append(grid_obj)
        basis_obj = None
        if basis:
            basis_obj = self.add_vectors([M[:, 0], M[:, 1]],
                                         colors=list(basis_colors),
                                         labels=list(basis_labels))
        return grid_obj, basis_obj

    # ── output ───────────────────────────────────────────────
    def display(self, size=(800, 800), axes=0):
        """Open an interactive window (backend depends on vedo settings)."""
        plt = Plotter(size=size, bg=self.bg)
        # resetcam=False is essential: otherwise vedo refits and ignores our
        # top-down camera (same lesson as k3d's camera_auto_fit).
        plt.show(self.objects, camera=self.camera, resetcam=False, axes=axes)
        return plt

    def save(self, path, size=(800, 800), axes=0, offscreen=True):
        """Render offscreen and write a static image — works fully headless."""
        plt = Plotter(offscreen=offscreen, size=size, bg=self.bg)
        plt.show(self.objects, camera=self.camera, resetcam=False, axes=axes)
        plt.screenshot(path)
        plt.close()
        return path



# %%
# Same scenes as the k3d cells, now in vedo.
vv = Plane2DVedo(extent=5, eye_height=7)
vv.apply_matrix([[1.0, 1.0],
                 [0.0, 1.0]])       # shear; warped lattice + image basis arrows

# Headless static export (no browser needed) — writes a committable figure:
#vv.save('shear.png')

# Interactive window (backend depends on your vedo/Jupyter setup):
plot = vv.display()


# %%
vv.save('shear2.png')

# %%

# %%
pwd

# %%

"""Render offscreen and write a static image — works fully headless."""
plt = Plotter(offscreen=False, size=(800,800))
plt.show(self.objects, camera=self.camera, resetcam=False, axes=axes)
        plt.screenshot(path)
        plt.close()
    
