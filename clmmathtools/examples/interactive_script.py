"""Interactive clmmathtools from a plain Python script.

Run it with a display attached:

    uv run python clmmathtools/examples/interactive_script.py            # analytic landscape
    uv run python clmmathtools/examples/interactive_script.py field      # 3D vector field
    uv run python clmmathtools/examples/interactive_script.py ppiped     # parallelepiped

Because we are **not** in a Jupyter notebook, ``display(interactive=True)`` routes to a native,
orbitable VTK window (drag to rotate, scroll to zoom). The call **blocks** until you close the window,
then the script continues. On a headless machine (no display) this will fail — that's expected;
interactive rendering needs a screen. For a notebook, use ``examples/06_interactive.ipynb`` instead.
"""

from __future__ import annotations

import sys

import numpy as np

import clmmathtools.viz as mv


def landscape():
    f = lambda z: (z**2 - 1) / (z**2 + 1)  # zeros at ±1, poles at ±i
    return mv.Space3D(bounds=2.5).landscape(f, res=140, zmax=3)


def field():
    def swirl(P):
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        return np.column_stack([-y, x, 0.3 * z])

    return mv.Space3D(bounds=3).field(swirl, n=7)


def ppiped():
    return mv.Space3D(bounds=3).parallelepiped(
        [0, 0, 0], [2, 0.3, 0], [0.4, 1.8, 0.2], [0.3, 0.5, 1.6], facecolor=mv.ORANGE
    )


SCENES = {"landscape": landscape, "field": field, "ppiped": ppiped}


def main(argv):
    name = argv[1] if len(argv) > 1 else "landscape"
    if name not in SCENES:
        sys.exit(f"unknown scene {name!r}; choose one of {sorted(SCENES)}")
    print(
        f"clmmathtools — opening an interactive window for '{name}'. Drag to orbit; close it to exit."
    )
    print("in a notebook?", mv.in_notebook(), "(False → native VTK window)")
    SCENES[name]().display(interactive=True)  # blocks until the window is closed
    print("window closed — done.")


if __name__ == "__main__":
    main(sys.argv)
