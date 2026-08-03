# doingmath
fun things to do in code to illustrate math

- experiment with using sympy for calculation


### visualization options
I have started a small library to help vizualize concepts in math called mathviz
It combines different things, using numpy, sympy and matplotlib or vedo for graphics.
Basic stuff is working with it, but will want to iterate on it for a while.



thinking about this:
- matplotlib - ok, common, not great 3D
- pyvista
- vedo - like pyvista
- k3d - js only output, is a backend for vedo
- plotly?, bokeh?, other pyviz options, visual phyisics?
- manim (community) clearly can do a lot for visualization.

### brainstorming regarding calculation engines/libraries
- numpy - standard for mulitarrays, includes linalg, can be supplmented with scipy
- sympy - sympbolic standard
- ?many sage toolkits - there is a project that breaks these out
- clifford - geometric algebra
- torch and jax
- lean4 integration ?
- what about standard diff geometry and such libraries, sympletic libraries?

### editing/publication/interaction
I'm using jupytext so that python code can be my source of truth but I can interact with it in a notebook environment
The basics are to use:
- python code in py:percent format
- jupyter notebooks


#### Future directions
- quarto (can use jupyter as one of its formats too)
- observable notebook
- marimo notebooks: favor matplotlib, altar and plotly for visualization
