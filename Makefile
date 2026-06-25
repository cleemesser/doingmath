.PHONY: sync

# Regenerate the paired .ipynb notebooks from their jupytext .py sources
# (the .py files are the source of truth).
sync:
	uv sync && uv run jupytext --sync GeometricLinearAlgebra/*.py
