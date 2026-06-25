project('doingmath', version: '0.1.0')

# `sync` — regenerate the paired .ipynb notebooks from their jupytext .py sources
# (the .py files are the source of truth). Run with:
#     uv run meson setup build          # once, creates the ./build dir
#     uv run meson compile -C build sync
# MESON_SOURCE_ROOT is set by meson for run_target commands, so we cd back to the
# project root (meson runs targets from the build directory) before globbing.
run_target(
  'sync',
  command: [
    'sh', '-c',
    'cd "$MESON_SOURCE_ROOT" && uv sync && uv run jupytext --sync GeometricLinearAlgebra/*.py',
  ],
)
