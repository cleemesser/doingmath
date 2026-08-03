"""Color conversions: 0xRRGGBB ints ↔ '#rrggbb' strings ↔ (r,g,b) floats."""

import pytest

from mathviz import palette


def test_hexstr_from_int():
    assert palette.hexstr(0x4FC3F7) == "#4fc3f7"
    assert palette.hexstr(0x000000) == "#000000"
    assert palette.hexstr(0xFFFFFF) == "#ffffff"


def test_hexstr_passthrough_string():
    assert palette.hexstr("#abc123") == "#abc123"


@pytest.mark.parametrize(
    "color, expected",
    [
        (0xFF0000, (1.0, 0.0, 0.0)),
        (0x00FF00, (0.0, 1.0, 0.0)),
        (0x0000FF, (0.0, 0.0, 1.0)),
        (0x000000, (0.0, 0.0, 0.0)),
        ("#ffffff", (1.0, 1.0, 1.0)),
    ],
)
def test_rgb01(color, expected):
    assert palette.rgb01(color) == pytest.approx(expected)


def test_rgb01_accepts_int_and_string_equivalently():
    assert palette.rgb01(0x4FC3F7) == pytest.approx(palette.rgb01("#4fc3f7"))


def test_named_colors_exist():
    for name in ("BLUE", "ORANGE", "GREEN", "RED", "PURPLE", "GREY", "BG"):
        assert isinstance(getattr(palette, name), int)
