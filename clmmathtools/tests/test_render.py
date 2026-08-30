"""End-to-end render smoke tests: each backend writes a non-blank PNG headless.

Parametrized over both backends via the ``backend`` fixture (see conftest.py).
"""

import numpy as np
from PIL import Image

from clmmathtools.viz import BLUE, GREY, ORANGE, Plane


def _content_pixels(path):
    arr = np.asarray(Image.open(path).convert("RGB"))
    return int((arr > 20).sum())


def test_render_primitives(tmp_path, backend, circle):
    out = tmp_path / f"prims_{backend}.png"
    p = Plane(extent=3, backend=backend)
    p.basis().curve(circle, BLUE).points([[2, 1]], ORANGE).text([-2.5, 2.5], "hi")
    p.save(str(out))
    assert out.exists()
    assert _content_pixels(out) > 1000  # something was actually drawn


def test_render_apply_matrix(tmp_path, backend):
    out = tmp_path / f"mat_{backend}.png"
    Plane(extent=3, grid=False, backend=backend).apply_matrix([[1, 1], [0, 1]]).save(
        str(out)
    )
    assert _content_pixels(out) > 1000


def test_render_apply_complex(tmp_path, backend):
    out = tmp_path / f"cplx_{backend}.png"
    Plane(extent=2, grid=False, backend=backend).apply_complex(
        lambda z: z**2, color=GREY, step=0.5
    ).save(str(out))
    assert _content_pixels(out) > 500


def test_render_polygon(tmp_path, backend):
    out = tmp_path / f"poly_{backend}.png"
    Plane(extent=2, backend=backend).parallelogram(
        [0, 0], [1.5, 0.3], [0.4, 1.4], edgecolor=BLUE
    ).save(str(out))
    assert _content_pixels(out) > 800


def test_save_returns_path(tmp_path):
    out = tmp_path / "r.png"
    assert Plane(extent=1).save(str(out)) == str(out)
