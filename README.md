# doingmath
a hodge podge of fun things to do in code to illustrate math

Directions for exploration
- experiment with using sympy for calculation
- tools for teaching - see what is available and develop my own library clmmathtools
- tools for exploration of pure mathematics and mathematical applications

### visualization options
I have started a small library to help vizualize concepts in math called clmmathtools
It combines different things, using numpy, sympy and matplotlib or vedo for graphics.
Basic stuff is working with it, but will want to iterate on it for a while.

#### visualizatio brainstorming
- matplotlib - ok, common, not great 3D
- pyvista
- vedo - like pyvista
- k3d - js only output, is a backend for vedo
- plotly?, bokeh?, other pyviz options, visual phyisics?
- manim (community) clearly can do a lot for visualization.
- whole pydata ecosystem: seaborn, vega, plotnine, graphviz, ggplot2, holviews,
  panel
- many interactive widget options

### brainstorming regarding calculation engines/libraries
- numpy - standard for mulitarrays, includes linalg, can be supplmented with scipy
- sympy - sympbolic standard
- ?many sage toolkits - [passage math is a project that breaks these out](https://github.com/passagemath/passagemath)
- clifford - geometric algebra
- torch and jax
- lean4 integration ?
- what about standard diff geometry and such libraries, sympletic libraries?
- [passagemath](https://github.com/passagemath/passagemath) which has all of
  sage as pip installable modules
- [think DSP book and code](https://github.com/AllenDowney/ThinkDSP) a nice
  exmaple of an interactive coding approach to signal processing.
- [cadabra](https://cadabra.science/) a computer algebra system optimized for
  field theory input is tex-inspired, programmable in python, C++
#### visualization libraries for illustrating math concepts
- [sympy-plot-backends](https://sympy-plot-backends.readthedocs.io)
  - I especially like the [complex
    analysis](https://sympy-plot-backends.readthedocs.io/en/latest/modules/graphics/complex_analysis.html)
    Wegert style domain_coloring functions
  - see the design of the [graphics module](https://sympy-plot-backends.readthedocs.io/en/latest/modules/graphics/index.html#graphics)
- dtumathtools
- [wigglystuff awesomeness](https://koaning.github.io/wigglystuff/) including:
  - TangleLatex widget
  - manim widget
  - matrix widget

### editing/publication/interaction
I'm using jupytext so that python code can be my source of truth but I can interact with it in a notebook environment
The basics are to use:
- python code in py:percent format
- jupyter notebooks


### example courses and demos
- [Learn Multibody Mechanics by JK
  Moore](https://moorepants.github.io/learn-multibody-dynamics/introduction.html)
- [DTU math1a demos](https://math1a.compute.dtu.dk/SymPy/sympydemos.html) from
     the [dtu math1a course](https://math1a.compute.dtu.dk) :
     (google-this sympy demo site:https://math1a.compute.dtu.dk/SymPy/  )
#### Future directions
- [quarto](https://quarto.org/) (can use jupyter as one of its formats too)
  - observable notebook
- marimo notebooks: reactive design, not so nice to write with but not terrible
  with vscode: reactive approach limits size/scope of notebook but makes web
  publication easier: matplotlib, altar and plotly for visualization
- wigglystuff tangleLatex widget is pretty great for integrated math widgets

- [jupyter_myst](https://github.com/jupyter-book/jupyterlab-myst) plugin might
  make to make integrating results into text better
  - there is MyST Markdown extension and Jupyter extension for VS Code

#### Want to use latex command definition in a notebook?
- it is possible to define new commands but..
- problem, each cell of a jupyter notebook has a fresh KaTex context

marimo has a solution

create a file macros.tex
```
\gdef\E{\mathbb{E}}
\gdef\P{\mathbb{P}}
\gdef\grad{\nabla}
```
\gdef is a standard TeX command for global definition (cf., \def, \edef, \xdef)
```python
import marimo as mo

mo.latex(filename="macros.tex")
```
