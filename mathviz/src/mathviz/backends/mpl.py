"""Matplotlib backend — 2D, raster-native (the natural home for phase portraits), with static PNG
output that embeds in notebooks and renders on GitHub. The default backend."""

from __future__ import annotations

import numpy as np

from .. import primitives as P
from ..palette import rgb01
from .base import Backend


class MatplotlibBackend(Backend):
    name = "mpl"

    def render(self, scene: P.Scene, *, save: str | None = None):
        import matplotlib

        if save is not None:
            matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        view = scene.view
        w, h = view.size
        fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
        bg = rgb01(view.bg)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        E = view.extent
        ax.set_xlim(-E, E)
        ax.set_ylim(-E, E)
        ax.set_aspect("equal")
        ax.axis("off")

        for prim in scene.primitives:
            self._draw(ax, prim)

        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        if save is not None:
            fig.savefig(save, facecolor=bg, dpi=100)
            plt.close(fig)
            return save
        try:  # explicit display so it works regardless of position in a cell
            from IPython.display import display

            display(fig)
            plt.close(fig)
        except Exception:
            return fig

    def _draw(self, ax, prim):
        if isinstance(prim, P.Grid):
            E, step = prim.extent, prim.step
            ticks = np.arange(-np.floor(E / step) * step, E + 1e-9, step)
            for t in ticks:
                ax.plot(
                    [-E, E],
                    [t, t],
                    color=rgb01(prim.color),
                    lw=prim.width,
                    alpha=prim.alpha,
                    zorder=1,
                )
                ax.plot(
                    [t, t],
                    [-E, E],
                    color=rgb01(prim.color),
                    lw=prim.width,
                    alpha=prim.alpha,
                    zorder=1,
                )
        elif isinstance(prim, P.Raster):
            ax.imshow(
                prim.rgb,
                extent=prim.extent,
                origin="lower",
                alpha=prim.alpha,
                zorder=0,
                interpolation="bilinear",
            )
        elif isinstance(prim, P.Segment):
            ax.plot(
                [prim.p0[0], prim.p1[0]],
                [prim.p0[1], prim.p1[1]],
                color=rgb01(prim.color),
                lw=prim.width,
                alpha=prim.alpha,
                zorder=2,
            )
        elif isinstance(prim, P.Polyline):
            pts = prim.pts
            if prim.closed:
                pts = np.vstack([pts, pts[0]])
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=rgb01(prim.color),
                lw=prim.width,
                alpha=prim.alpha,
                zorder=3,
            )
        elif isinstance(prim, P.Arrow):
            ax.annotate(
                "",
                xy=tuple(prim.head),
                xytext=tuple(prim.tail),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=rgb01(prim.color),
                    lw=prim.width,
                    alpha=prim.alpha,
                    shrinkA=0,
                    shrinkB=0,
                ),
                zorder=4,
            )
            if prim.label:
                mid = prim.head + 0.08 * (prim.head - prim.tail)
                ax.text(
                    mid[0],
                    mid[1],
                    prim.label,
                    color=rgb01(prim.color),
                    fontsize=11,
                    zorder=5,
                )
        elif isinstance(prim, P.Points):
            ax.scatter(
                prim.pts[:, 0],
                prim.pts[:, 1],
                s=prim.size**2,
                c=[rgb01(prim.color)],
                alpha=prim.alpha,
                zorder=5,
            )
        elif isinstance(prim, P.Text):
            ax.text(
                prim.pos[0],
                prim.pos[1],
                prim.text,
                color=rgb01(prim.color),
                fontsize=prim.size,
                zorder=6,
            )
