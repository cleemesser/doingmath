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


class VedoBackend(Backend):
    name = "vedo"

    def render(self, scene: P.Scene, *, save: str | None = None):
        import vedo
        import vedo.settings

        vedo.settings.default_backend = "vtk"
        from vedo import Arrow, Grid, Line, Plotter, Points, Text3D, utils

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

        plt = Plotter(offscreen=True, size=view.size, bg=hexstr(view.bg))
        plt.show(objects, camera=cam, resetcam=False, zoom=1, axes=0)
        arr = plt.screenshot(asarray=True)
        plt.close()

        if save is not None:
            from PIL import Image

            Image.fromarray(arr).save(save)
            return save
        from IPython.display import display
        from PIL import Image

        display(Image.fromarray(arr))
