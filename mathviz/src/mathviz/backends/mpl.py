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

        if isinstance(scene.view, P.View3D):
            return self._render3d(scene, plt, save=save)

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
        return self._emit(fig, plt, save)

    def _emit(self, fig, plt, save):
        if save is not None:
            fig.savefig(save, facecolor=fig.get_facecolor(), dpi=100)
            plt.close(fig)
            return save
        try:  # explicit display so it works regardless of position in a cell
            from IPython.display import display

            display(fig)
            plt.close(fig)
        except Exception:
            return fig

    # ── 3D scenes (mplot3d) ──────────────────────────────────
    def _render3d(self, scene, plt, *, save=None):
        view = scene.view
        w, h = view.size
        bg = rgb01(view.bg)
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        ax.view_init(elev=view.elev, azim=view.azim)

        tails, dirs, cols = [], [], []
        for prim in scene.primitives:
            if isinstance(prim, P.Arrow3D):
                tails.append(prim.tail)
                dirs.append(prim.head - prim.tail)
                cols.append(rgb01(prim.color))
            elif isinstance(prim, P.Line3D):
                ax.plot(
                    prim.pts[:, 0],
                    prim.pts[:, 1],
                    prim.pts[:, 2],
                    color=rgb01(prim.color),
                    lw=prim.width,
                    alpha=prim.alpha,
                )
            elif isinstance(prim, P.Points3D):
                ax.scatter(
                    prim.pts[:, 0],
                    prim.pts[:, 1],
                    prim.pts[:, 2],
                    s=prim.size**2,
                    c=[rgb01(prim.color)],
                    alpha=prim.alpha,
                )
            elif isinstance(prim, P.Surface):
                if prim.colors is not None:
                    fc = np.clip(np.asarray(prim.colors, float), 0, 1)
                    if fc.shape[-1] == 3:  # plot_surface wants RGBA facecolors
                        fc = np.concatenate(
                            [fc, np.ones(fc.shape[:-1] + (1,))], axis=-1
                        )
                    ax.plot_surface(
                        prim.X,
                        prim.Y,
                        prim.Z,
                        facecolors=fc,
                        rstride=1,
                        cstride=1,
                        linewidth=0,
                        antialiased=False,
                        shade=False,
                        alpha=prim.alpha,
                    )
                else:
                    ax.plot_surface(
                        prim.X,
                        prim.Y,
                        prim.Z,
                        color=rgb01(prim.color),
                        rstride=1,
                        cstride=1,
                        linewidth=0,
                        antialiased=False,
                        alpha=prim.alpha,
                    )
        if tails:
            t = np.array(tails)
            d = np.array(dirs)
            ax.quiver(
                t[:, 0],
                t[:, 1],
                t[:, 2],
                d[:, 0],
                d[:, 1],
                d[:, 2],
                colors=cols,
                arrow_length_ratio=0.35,
                linewidth=1.5,
            )

        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_pane_color((0, 0, 0, 0))
        ax.grid(False)
        return self._emit(fig, plt, save)

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
