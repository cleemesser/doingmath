"""Matplotlib backend — 2D, raster-native (the natural home for phase portraits), with static PNG
output that embeds in notebooks and renders on GitHub. The default backend."""

from __future__ import annotations
import re
import numpy as np

from .. import primitives as P
from ..palette import rgb01
from .base import Backend

def svg_inline_mpl(scene, width=430):
    """A mathviz scene -> inline SVG (see 01_..._marimo.py for why not .display()).

    Inline SVG rather than `mo.image()`: mo.image serves a PNG whose *filename is a content
    hash*, so every widget tick mints a fresh URL and the browser tears down the old <img> to
    re-fetch it. That blank gap — plus an <img> with no reserved height collapsing the row —
    is what made these cells flash while dragging. Inline SVG is DOM, not an asset, so it
    swaps in the same paint as the rest of the cell output. It also renders in about half the
    time and ships ~3x smaller than the PNG did.

    This may need to have some checking on backend because it assumes scene is going to work
    """

    buf = io.StringIO()
    scene.save(buf, format="svg")
    body = buf.getvalue()
    body = body[
        body.index("<svg") :
    ]  # the XML declaration + DOCTYPE are illegal inline
    body = re.sub(  # let the wrapper size it, not matplotlib's fixed pt dimensions
        r'(<svg\b[^>]*?)\s*width="[\d.]+pt"\s*height="[\d.]+pt"',
        r'\1 width="100%" height="100%" style="display:block"',
        body,
        count=1,
    )
    w, h = scene.view.size  # a fixed box => no reflow between frames
    # return mo.Html(  # from marimo adapter
    return  f'<div style="width:{width}px;height:{round(width * h / w)}px;flex:0 0 auto">{body}</div>'



class MatplotlibBackend(Backend):
    name = "mpl"

    @staticmethod
    def _figure(w, h):
        """A pyplot-free figure with an Agg canvas — never touches the kernel's active backend
        (so it can't turn into an ipympl/inline widget). Renders to a static PNG every time."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(w / 100, h / 100), dpi=100)
        FigureCanvasAgg(fig)
        return fig

    def render(
        self,
        scene: P.Scene,
        *,
        save: str | None = None,
        format: str | None = None,
        interactive=None,
        vedo_display=None,
    ):
        # matplotlib renders static images; the interactive flags are for the vedo backend.
        if isinstance(scene.view, P.View3D):
            return self._render3d(scene, save=save, format=format)

        view = scene.view
        w, h = view.size
        fig = self._figure(w, h)
        ax = fig.add_subplot(111)
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
        return self._emit(fig, save, format)

    def _emit(self, fig, save, format=None):
        if save is not None:
            # `format` is required when `save` is a buffer: savefig infers from the filename
            # suffix, and a BytesIO/StringIO has none, so it would silently fall back to PNG.
            fig.savefig(save, format=format, facecolor=fig.get_facecolor(), dpi=100)
            print(f"about to return buffer in format={format}")
            return save
        try:  # display the rendered PNG bytes — a static image regardless of the active
            import io  # matplotlib backend (inline / ipympl-widget / Agg) or cell position

            from IPython.display import Image, display, SVG
            if format=='png' or not format: # default to png image
                print("try default png path")
                buf = io.BytesIO()
                fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), dpi=100)
                display(Image(data=buf.getvalue()))
            if format=='svg':
                print("try the svg path")
                buf = io.BytesIO()
                fig.savefig(buf, format="svg", facecolor=fig.get_facecolor(), dpi=100)
                display(SVG(data=buf.getvalue()))
            # should any of this play return something?
        except Exception:
            return fig

    # ── 3D scenes (mplot3d) ──────────────────────────────────
    def _render3d(self, scene, *, save=None, format=None):
        import mpl_toolkits.mplot3d  # noqa: F401  (registers the '3d' projection)

        view = scene.view
        w, h = view.size
        bg = rgb01(view.bg)
        fig = self._figure(w, h)
        ax = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        # mplot3d is z-up by default; vertical_axis mirrors Space3D(up=...) for the other choices
        ax.view_init(elev=view.elev, azim=view.azim, vertical_axis=view.up)

        tails, dirs, cols, pts3 = [], [], [], []
        for prim in scene.primitives:
            if isinstance(prim, P.Arrow3D):
                tails.append(prim.tail)
                dirs.append(prim.head - prim.tail)
                cols.append(rgb01(prim.color))
                pts3.extend([prim.tail, prim.head])
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
            elif isinstance(prim, P.Mesh3D):
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection

                polys = [prim.verts[f] for f in prim.faces]
                if prim.colors is not None:
                    fc = [
                        (*rgb01(int(c)), prim.alpha)
                        if np.isscalar(c)
                        else (*np.clip(c, 0, 1)[:3], prim.alpha)
                        for c in prim.colors
                    ]
                else:
                    fc = (*rgb01(prim.facecolor), prim.alpha)
                ec = rgb01(prim.edgecolor) if prim.edgecolor is not None else "none"
                ax.add_collection3d(
                    Poly3DCollection(
                        polys, facecolors=fc, edgecolors=ec, linewidths=prim.edgewidth
                    )
                )
                pts3.extend(prim.verts)
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

        if pts3 and any(isinstance(p, P.Mesh3D) for p in scene.primitives):
            # mplot3d does not autoscale to Poly3DCollection — frame from the mesh vertices
            arr = np.asarray(pts3, float)
            lo, hi = arr.min(0), arr.max(0)
            c = (lo + hi) / 2
            r = max((hi - lo).max() / 2, 1e-6) * 1.15
            ax.set_xlim(c[0] - r, c[0] + r)
            ax.set_ylim(c[1] - r, c[1] + r)
            ax.set_zlim(c[2] - r, c[2] + r)
            ax.set_box_aspect((1, 1, 1))

        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_pane_color((0, 0, 0, 0))
        ax.grid(False)
        return self._emit(fig, save, format)

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
        elif isinstance(prim, P.Polygon):
            from matplotlib.patches import Polygon as _MplPolygon

            ax.add_patch(
                _MplPolygon(
                    prim.pts,
                    closed=True,
                    facecolor=rgb01(prim.facecolor),
                    alpha=prim.alpha,
                    edgecolor=(
                        rgb01(prim.edgecolor) if prim.edgecolor is not None else "none"
                    ),
                    linewidth=prim.edgewidth,
                    zorder=2.5,
                )
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
