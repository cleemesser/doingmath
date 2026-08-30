"""Vedo (VTK) backend — 2D parity with the ``Geometry/`` notebooks, plus a path to 3D (analytic
landscapes, Riemann surfaces). Renders offscreen to a static PNG that is embedded inline via IPython,
so figures survive a headless ``jupytext --execute`` and show on GitHub.
"""

from __future__ import annotations

import numpy as np

from .. import primitives as P
from ..palette import hexstr
from .base import Backend


def _pad(xy):
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    return np.column_stack([xy, np.zeros(len(xy))])


def _take_zup(show_kw):
    """Pull the private z-up flag back out of the show kwargs (it is ours, not vedo's)."""
    return bool(show_kw.pop("zup", False))


def _aim_zup(plt, view):
    """Aim a Plotter's camera from ``(elev, azim)`` with world **+z** up (matplotlib's convention).

    VTK's default camera is y-up, and ``show(elevation=, azimuth=)`` only rotates *relative* to it —
    which leaves z pointing sideways. Setting the camera up front instead keeps z vertical. Only the
    *direction* is set here: the following ``show(resetcam=True)`` slides the camera along it until
    the scene fits, so no distance has to be computed.

    No-op under a renderer-less display backend (k3d renders in JS, so ``Plotter.camera`` — a property
    over ``renderer.GetActiveCamera()`` — is ``None`` until ``show()``; assigning to it is a no-op too).
    """
    cam = plt.camera
    if cam is None:
        return
    e = np.radians(
        np.clip(view.elev, -89.0, 89.0)
    )  # ±90° would align the view with the up vector
    a = np.radians(view.azim)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    focus = np.array(cam.GetFocalPoint(), dtype=float)
    cam.SetPosition(*(focus + d))
    cam.SetViewUp(0, 0, 1)


class VedoBackend(Backend):
    name = "vedo"

    def render(
        self,
        scene: P.Scene,
        *,
        save: str | None = None,
        format: str | None = None,  # noqa: ARG002  (vedo infers from the filename; kept for interface parity)
        interactive=None,
        vedo_display=None,
    ):
        from . import resolve_interactive, resolve_vedo_display

        interactive = resolve_interactive(interactive)
        vedo_display = resolve_vedo_display(vedo_display)

        if isinstance(scene.view, P.View3D):
            return self._render3d(
                scene, save=save, interactive=interactive, vedo_display=vedo_display
            )

        from vedo import Arrow, Grid, Line, Points, Text3D, utils

        view = scene.view
        E = view.extent
        objects = []
        for prim in scene.primitives:
            if isinstance(prim, P.Grid):
                ticks = np.arange(-E, E + 1e-9, prim.step)
                objects.append(
                    Grid(s=(ticks, ticks))
                    .wireframe(True)
                    .c(hexstr(prim.color))
                    .alpha(prim.alpha)
                )
            elif isinstance(prim, P.Segment):
                objects.append(
                    Line(
                        _pad(prim.p0)[0],
                        _pad(prim.p1)[0],
                        c=hexstr(prim.color),
                        lw=prim.width,
                    ).alpha(prim.alpha)
                )
            elif isinstance(prim, P.Polyline):
                pts = prim.pts
                if prim.closed:
                    pts = np.vstack([pts, pts[0]])
                objects.append(
                    Line(_pad(pts), c=hexstr(prim.color), lw=prim.width).alpha(
                        prim.alpha
                    )
                )
            elif isinstance(prim, P.Arrow):
                objects.append(
                    Arrow(
                        _pad(prim.tail)[0],
                        _pad(prim.head)[0],
                        c=hexstr(prim.color),
                        shaft_radius=0.008 * prim.width,
                        head_radius=0.03 * prim.width,
                        head_length=0.09 * prim.width,
                    ).alpha(prim.alpha)
                )
                if prim.label:
                    mid = prim.head + 0.08 * (prim.head - prim.tail)
                    objects.append(
                        Text3D(
                            prim.label,
                            pos=_pad(mid)[0] + np.array([0.05, 0.05, 0]),
                            s=0.24,
                            c=hexstr(prim.color),
                        )
                    )
            elif isinstance(prim, P.Points):
                objects.append(
                    Points(_pad(prim.pts), r=prim.size, c=hexstr(prim.color)).alpha(
                        prim.alpha
                    )
                )
            elif isinstance(prim, P.Text):
                objects.append(
                    Text3D(
                        prim.text, pos=_pad(prim.pos)[0], s=0.24, c=hexstr(prim.color)
                    )
                )
            elif isinstance(prim, P.Polygon):
                from vedo import Mesh

                v = _pad(prim.pts)
                mesh = (
                    Mesh([v, [list(range(len(prim.pts)))]])
                    .c(hexstr(prim.facecolor))
                    .alpha(prim.alpha)
                    .lighting("off")
                )
                objects.append(mesh)
                if prim.edgecolor is not None:
                    objects.append(
                        Line(
                            np.vstack([v, v[0]]),
                            c=hexstr(prim.edgecolor),
                            lw=prim.edgewidth,
                        )
                    )
            elif isinstance(prim, P.Raster):
                from vedo import Image

                rgb = np.asarray(prim.rgb)
                if rgb.dtype != np.uint8:
                    rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
                xmin, xmax, ymin, ymax = prim.extent
                ny, nx = rgb.shape[:2]
                im = Image(np.flipud(rgb))  # array row 0 is the top; keep it up-top
                im.scale([(xmax - xmin) / nx, (ymax - ymin) / ny, 1.0])
                im.pos(xmin, ymin, -0.05)  # sit just behind the z=0 vector layer
                if prim.alpha < 1.0:
                    im.alpha(prim.alpha)
                objects.insert(0, im)

        camera = {
            "pos": (0, 0, 2 * E),
            "focal_point": (0, 0, 0),
            "viewup": (0, 1, 0),
            "parallel_scale": E,
        }
        cam = utils.camera_from_dict(camera)
        cam.SetParallelProjection(True)

        return self._show(
            objects,
            view,
            save,
            interactive,
            vedo_display,
            camera=cam,
            resetcam=False,
            zoom=1,
            axes=0,
        )

    # ── 3D scenes ─────────────────────────────────────────────
    def _render3d(self, scene, *, save=None, interactive=False, vedo_display="k3d"):
        from vedo import Arrow, Line, Mesh, Points

        view = scene.view
        objects = []
        for prim in scene.primitives:
            if isinstance(prim, P.Arrow3D):
                objects.append(
                    Arrow(prim.tail, prim.head, c=hexstr(prim.color)).alpha(prim.alpha)
                )
            elif isinstance(prim, P.Line3D):
                objects.append(
                    Line(prim.pts, c=hexstr(prim.color), lw=prim.width).alpha(
                        prim.alpha
                    )
                )
            elif isinstance(prim, P.Points3D):
                objects.append(
                    Points(prim.pts, r=prim.size, c=hexstr(prim.color)).alpha(
                        prim.alpha
                    )
                )
            elif isinstance(prim, P.Surface):
                objects.append(self._surface_mesh(prim, Mesh))
            elif isinstance(prim, P.Mesh3D):
                m = Mesh([prim.verts, prim.faces]).alpha(prim.alpha).lighting("plastic")
                if prim.colors is not None:
                    rgba = (np.clip(np.asarray(prim.colors), 0, 1) * 255).astype(
                        np.uint8
                    )
                    rgba = np.column_stack(
                        [rgba[:, :3], np.full(len(rgba), 255, np.uint8)]
                    )
                    m.cellcolors = rgba
                else:
                    m.c(hexstr(prim.facecolor))
                if prim.edgecolor is not None:
                    m.linewidth(prim.edgewidth).linecolor(hexstr(prim.edgecolor))
                objects.append(m)

        # z-up aims the camera before the scene is shown (see _aim_zup); y-up keeps VTK's default
        # camera and rotates relative to it.
        aim = (
            {"zup": True}
            if view.up == "z"
            else {"elevation": view.elev, "azimuth": view.azim}
        )
        return self._show(
            objects,
            view,
            save,
            interactive,
            vedo_display,
            **aim,
            axes=1,
            resetcam=True,
        )

    # ── offscreen (static PNG) vs. live (interactive) ────────────────────────────
    def _show(self, objects, view, save, interactive, vedo_display, **show_kw):
        """Route to: a live notebook widget, a live desktop window, or an offscreen PNG."""
        from . import in_notebook

        if interactive and save is None:
            if in_notebook():
                return self._live_widget(objects, view, show_kw, vedo_display)
            return self._live_window(
                objects, view, show_kw
            )  # plain script → native VTK window

        import vedo
        import vedo.settings
        from vedo import Plotter

        vedo.settings.default_backend = "vtk"  # offscreen → static PNG
        plt = Plotter(offscreen=True, size=view.size, bg=hexstr(view.bg))
        if _take_zup(show_kw):
            _aim_zup(plt, view)
        plt.show(objects, **show_kw)
        arr = plt.screenshot(asarray=True)
        plt.close()
        return self._emit(arr, save)

    def _live_widget(self, objects, view, show_kw, vedo_display):
        """Inline live widget in a Jupyter notebook (k3d/trame/…); returns the widget to display."""
        import vedo
        import vedo.settings
        from vedo import Plotter

        vedo.settings.default_backend = vedo_display or "k3d"
        zup = _take_zup(show_kw)
        live_kw = {
            k: v for k, v in show_kw.items() if k != "camera"
        }  # k3d ignores vtkCamera
        plt = Plotter(size=view.size, bg=hexstr(view.bg))
        if zup:
            _aim_zup(
                plt, view
            )  # honored by the VTK-backed widgets; k3d has its own camera
        return plt.show(objects, **live_kw)

    def _live_window(self, objects, view, show_kw):
        """A native, orbitable VTK window for a plain ``python script.py`` run (blocks until closed)."""
        import vedo
        import vedo.settings
        from vedo import Plotter

        vedo.settings.default_backend = "vtk"
        plt = Plotter(size=view.size, bg=hexstr(view.bg), title="clmmathtools")
        if _take_zup(show_kw):
            _aim_zup(plt, view)
        plt.show(
            objects, **{**show_kw, "interactive": True}
        )  # blocks: drag to orbit, close to continue
        plt.close()
        return None

    @staticmethod
    def _surface_mesh(prim, Mesh):
        m, n = prim.Z.shape
        verts = np.column_stack([prim.X.ravel(), prim.Y.ravel(), prim.Z.ravel()])
        faces = []
        for i in range(m - 1):
            for j in range(n - 1):
                a = i * n + j
                faces.append([a, a + 1, a + n + 1, a + n])  # quad, CCW
        mesh = Mesh([verts, faces]).alpha(prim.alpha).lighting("plastic")
        if prim.colors is not None:
            rgba = (np.clip(prim.colors.reshape(-1, 3), 0, 1) * 255).astype(np.uint8)
            rgba = np.column_stack([rgba, np.full(len(rgba), 255, np.uint8)])
            mesh.pointcolors = rgba
        else:
            mesh.c(hexstr(prim.color))
        if prim.wireframe:
            mesh.wireframe(True)
        return mesh

    @staticmethod
    def _emit(arr, save):
        import os

        from PIL import Image

        if save is not None:
            # PIL infers the format from a path's extension, but an in-memory buffer
            # (io.BytesIO, as animate.to_png passes) has no name — say PNG explicitly.
            fmt = None if isinstance(save, (str, os.PathLike)) else "PNG"
            Image.fromarray(arr).save(save, format=fmt)
            return save
        from IPython.display import display

        display(Image.fromarray(arr))
