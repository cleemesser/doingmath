# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="SazxT1wreLFe"
# # Finding Primes with the Sieve of Eratosthenes
# pronunciation: "air-uh-TAWS-thuh-neeze"

# %% [markdown] id="UlxmlaOaeLF5"
# Basic idea is that prime numbers a a precious resource in number land (there are only an infinite number of them!). They are sort of the building blocks of all the integers.

# %% colab={"base_uri": "https://localhost:8080/"} id="cEenhDWZeLGH" outputId="69a1f516-486b-495f-efb6-b4e9685ab795"
N = 20
candidates = {ii: "unknown" for ii in range(1, N + 1)}
candidates
# use 'unknown' to say we don't know yet if an integer is prime
# use True if it is a prime
# use False if it is not a prime

# %% colab={"base_uri": "https://localhost:8080/"} id="ZMYkUPqFeLGN" outputId="504017a9-21bc-48ac-fa81-34d400a3308a"
for candidate_number in candidates:
    if candidates[candidate_number] != "unknown":
        print(
            f"""We already think {candidate_number} is {"prime" if candidates[candidate_number] else "not-prime"}"""
        )
        continue
    ans = input(f"is {candidate_number} prime? [y/n/unknown]")
    if ans == "y":
        candidates[candidate_number] = True
        for ii in range(candidate_number + 1, N + 1):
            if ii % candidate_number == 0 and candidate_number != 1:
                candidates[ii] = False
    elif ans == "n":
        candidates[candidate_number] = False
    else:
        candidates[candidate_number] = "unknown"
        print(
            f"you entered that you did not know if {candidate_number} was prime. Will have to go back to that one."
        )

# %% colab={"base_uri": "https://localhost:8080/"} id="Uoy_4HaXlV-x" outputId="a5ea50fc-93aa-4539-ec28-7b30c66caf69"
2 % 2

# %% colab={"base_uri": "https://localhost:8080/"} id="k-__97bgeLGX" outputId="5b598488-9177-4cc6-9d9a-4fe058707215"
print(f"primes up to {N} are:")
# print(candidates)
[ii for ii in candidates if candidates[ii] == True]

# %% id="wz-npRglnfbA"
import numba


# %% id="tVKE8G3mnXEj"
def is_prime(n: int) -> bool:
    """Primality test using 6k+-1 optimization."""
    if n <= 3:
        return n > 1
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i**2 <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


# %% colab={"base_uri": "https://localhost:8080/"} id="g6iLNnCdnLmZ" outputId="184be81c-03d8-439a-e543-12a4913314d3"
@numba.jit
def is_prime1(n: int) -> bool:
    """Primality test using 6k+-1 optimization."""
    if n <= 3:
        return n > 1
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i**2 <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


# compile
is_prime1(10001)

# %% id="wgB4vnhspBlX"
big = 10293912834928234283

# %% colab={"base_uri": "https://localhost:8080/"} id="sXlS0Lbwnv5_" outputId="042d0e93-58d8-41da-d45f-c7ad1b135a2f"
# %timeit  for ii in range(1000):  is_prime(ii)

# %% colab={"base_uri": "https://localhost:8080/"} id="gDdpcA07oPTo" outputId="937f14c9-6256-4c8e-faf5-91a89cc1ef29"
# %timeit for ii in range(1000): is_prime1(ii)

# %% colab={"base_uri": "https://localhost:8080/"} id="dd05pCwuoYnA" outputId="a2e73898-a3eb-4c8b-d0b6-af0244502f4d"
# %timeit is_prime(big)

# %% colab={"base_uri": "https://localhost:8080/", "height": 387} id="cG8_OLz2o-aI" outputId="35725ca7-dbd9-4600-f598-c33a792f8b4a"
# %timeit for ii in range(big, big+10): is_prime1(ii)

# %% colab={"base_uri": "https://localhost:8080/"} id="R0nX63aRoWAZ" outputId="e03a3a10-71c8-407f-8c8b-5cf6b130a5c1"
is_prime1.inspect_llvm()

# %% id="ZF_DUv3sQ1gA"
import sympy

# %% id="jlgLVTCrREl-"
sympy

# %% id="Uqzsduj_n85g"
import mpmath

# %% colab={"base_uri": "https://localhost:8080/"} id="7NBtnXXkR4Ge" outputId="6718e904-8a02-44d1-c5ae-b697feffca00"
from sympy import init_printing, init_session

init_printing()
init_session()

# %% id="_zqG0MK2XGGO"
from sympy import sieve

# %% colab={"base_uri": "https://localhost:8080/"} id="wCTlP2znXxC-" outputId="8334a297-ffed-46a5-b2b2-3fa9ccceec67"
1000 in sieve

# %% colab={"base_uri": "https://localhost:8080/"} id="pN7_UVCNX0YG" outputId="ef2f73ea-d4bb-4be3-ba56-94d5122d983a"
sieve._list

# %% colab={"base_uri": "https://localhost:8080/"} id="T2lTPm0VX2L2" outputId="8055ae29-dc12-4ada-9e13-b3bbc049ad1f"
1001 in sieve

# %% colab={"base_uri": "https://localhost:8080/"} id="1gOqvOfKX6ql" outputId="6226d9dd-40c4-441e-a40d-3e6691d8b311"
sieve._list

# %% colab={"base_uri": "https://localhost:8080/"} id="8AXrArJTX75W" outputId="a5b63407-dfcd-4f91-f765-9ffdc1749c3b"
print([ii for ii in sieve.primerange(1, 21)])

# %% id="CptJQf9pYOgW"
from sympy import prime as nthprime

# %% colab={"base_uri": "https://localhost:8080/", "height": 36} id="0j_7A8Y8Yez2" outputId="c6a65061-73f1-4623-b6d2-4bf7c18d07d8"
nthprime(10)

# %% colab={"base_uri": "https://localhost:8080/", "height": 36} id="7FZLf9VHYh89" outputId="2c1c23f2-2b62-4ee8-b43e-62eb0a84222b"
nthprime(100)

# %% colab={"base_uri": "https://localhost:8080/", "height": 36} id="UVkh1fZRYjow" outputId="3c2fa3bb-c93c-413c-a079-00377f97737f"
nthprime(1000)

# %% colab={"base_uri": "https://localhost:8080/", "height": 36} id="2mZDGaIvYlLd" outputId="0a4ba2b4-a316-4d73-bdcd-e9f3051ca12d"
nthprime(10000)

# %% colab={"base_uri": "https://localhost:8080/", "height": 118} id="vouGWS-HYoXO" outputId="f8e21f0d-ef99-4cb6-f6b4-d9a720783f57"
import altair as alt
from vega_datasets import data

source = data.cars()

# Configure common options
base = (
    alt.Chart(source)
    .transform_aggregate(num_cars="count()", groupby=["Origin", "Cylinders"])
    .encode(
        alt.X("Cylinders:O", scale=alt.Scale(paddingInner=0)),
        alt.Y("Origin:O", scale=alt.Scale(paddingInner=0)),
    )
)

# Configure heatmap
heatmap = base.mark_rect().encode(
    color=alt.Color(
        "num_cars:Q",
        scale=alt.Scale(scheme="viridis"),
        legend=alt.Legend(direction="horizontal"),
    )
)

# Configure text
text = base.mark_text(baseline="middle").encode(
    text="num_cars:Q",
    color=alt.condition(
        alt.datum.num_cars > 100, alt.value("black"), alt.value("white")
    ),
)

# Draw the chart
heatmap + text

# %% id="xMXyNBUXdgL7"
import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_notebook

# Call once to configure Bokeh to display plots inline in the notebook.
output_notebook()

# %% colab={"base_uri": "https://localhost:8080/", "height": 486} id="9hYNeogsdYcp" outputId="aed6809f-dd67-4a6d-b481-e6fb22638daa"
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.sampledata.periodic_table import elements
from bokeh.transform import dodge, factor_cmap

output_file("periodic.html")

periods = ["I", "II", "III", "IV", "V", "VI", "VII"]
groups = [str(x) for x in range(1, 19)]

df = elements.copy()
df["atomic mass"] = df["atomic mass"].astype(str)
df["group"] = df["group"].astype(str)
df["period"] = [periods[x - 1] for x in df.period]
df = df[df.group != "-"]
df = df[df.symbol != "Lr"]
df = df[df.symbol != "Lu"]

cmap = {
    "alkali metal": "#a6cee3",
    "alkaline earth metal": "#1f78b4",
    "metal": "#d93b43",
    "halogen": "#999d9a",
    "metalloid": "#e08d49",
    "noble gas": "#eaeaea",
    "nonmetal": "#f1d4Af",
    "transition metal": "#599d7A",
}

TOOLTIPS = [
    ("Name", "@name"),
    ("Atomic number", "@{atomic number}"),
    ("Atomic mass", "@{atomic mass}"),
    ("Type", "@metal"),
    ("CPK color", "$color[hex, swatch]:CPK"),
    ("Electronic configuration", "@{electronic configuration}"),
]

p = figure(
    title="Periodic Table (omitting LA and AC Series)",
    plot_width=1000,
    plot_height=450,
    x_range=groups,
    y_range=list(reversed(periods)),
    tools="hover",
    toolbar_location=None,
    tooltips=TOOLTIPS,
)

r = p.rect(
    "group",
    "period",
    0.95,
    0.95,
    source=df,
    fill_alpha=0.6,
    legend_field="metal",
    color=factor_cmap("metal", palette=list(cmap.values()), factors=list(cmap.keys())),
)

text_props = {"source": df, "text_align": "left", "text_baseline": "middle"}

x = dodge("group", -0.4, range=p.x_range)

p.text(x=x, y="period", text="symbol", text_font_style="bold", **text_props)

p.text(
    x=x,
    y=dodge("period", 0.3, range=p.y_range),
    text="atomic number",
    text_font_size="11px",
    **text_props,
)

p.text(
    x=x,
    y=dodge("period", -0.35, range=p.y_range),
    text="name",
    text_font_size="7px",
    **text_props,
)

p.text(
    x=x,
    y=dodge("period", -0.2, range=p.y_range),
    text="atomic mass",
    text_font_size="7px",
    **text_props,
)

p.text(
    x=["3", "3"],
    y=["VI", "VII"],
    text=["LA", "AC"],
    text_align="center",
    text_baseline="middle",
)

p.outline_line_color = None
p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_standoff = 0
p.legend.orientation = "horizontal"
p.legend.location = "top_center"
p.hover.renderers = [r]  # only hover element boxes

show(p)

# %% id="RLNeubQgdaE2"
