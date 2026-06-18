# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3.8.5 64-bit ('vizdev')
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import sympy

from bokeh.plotting import figure
from bokeh.io import output_notebook, push_notebook, output_file, show
from bokeh.transform import dodge, factor_cmap

# Call once to configure Bokeh to display plots inline in the notebook.
output_notebook()

# %%
# create the initial structure for defining things with a dictionary
# the data source will ultimately be a pandas.DataFrame

N = 100  # integers from 1 upto and including N
# use 'unknown' to say we don't know yet if an integer is prime
# use True if it is a prime
# use False if it is not a prime
candidates = {ii: "unknown" for ii in range(1, N + 1)}
candidates[1] = False
f"{len(candidates)=}"

# %%
# start making the dataframe
df = pd.DataFrame(candidates.items())
df.columns = ["integer", "is_prime"]
df.set_index("integer")
df.head()

# %%
# Now decorate the dataframe and make it so that it can use strings for most things which is what bokeh needs
# columns = ["I", "II", "III", "IV", "V", "VI", "VII"]
# columns = list(range(1,11))
# rows = [str(x) for x in range(1, 19)]
rows = [str(ii) for ii in range(0, 11)]
columns = [str(ii) for ii in range(10)]


df = pd.DataFrame(candidates.items(), columns=["integer", "is_prime"])
df.set_index("integer")

df["row"] = df.integer.apply(lambda x: (x - 1) // 10)  # not sure if these make sense
df["column"] = df.integer.apply(lambda x: (x - 1) % 10)
df["row"] = df["row"].astype(str)
df["column"] = df["column"].astype(str)
df["number"] = df["integer"].astype(str)

# %%
print(f"{set(df['row'])=}")
print(f"{set(df['column'])=}")
df.is_prime = df.is_prime.astype(str)
# df.is_prime.to_list()

# %%
# df["atomic mass"] = df["atomic mass"].astype(str)
# df["row"] = df["row"].astype(str)
# df["column"] = [columns[x-1] for x in df.column]
# df = df[df.row != "-"]
# df = df[df.symbol != "Lr"]
# df = df[df.symbol != "Lu"]

cmap = {
    str(True): "#FFFFFF",  # "#a6cee3",
    str(False): "#0000b4",
    # "metal"                : "#d93b43",
    "unknown": "#999d9a",
    # "metalloid"            : "#e08d49",
    # "noble gas"            : "#eaeaea",
    # "nonmetal"             : "#f1d4Af",
    # "transition metal"     : "#599d7A",
}
cmap

# %%
TOOLTIPS = [
    ("number", "@number"),
    ("is_prime", "@is_prime"),
    # # factors?
    #   ("Atomic number", "@{atomic number}"),
    #   ("Atomic mass", "@{atomic mass}"),
    #  ("Type", "@metal"),
    #   ("CPK color", "$color[hex, swatch]:CPK"),
    #  ("Electronic configuration", "@{electronic configuration}"),
]

# %%
print(list(cmap.values()))
list(cmap.keys())

# %%
p = figure(
    title="Primes",
    width=1000,
    height=450,
    x_range=columns,
    y_range=list(reversed(rows)),
    tools="hover",
    toolbar_location=None,
    tooltips=TOOLTIPS,
)

# %%
r = p.rect(
    "column",
    "row",
    0.95,
    0.95,
    source=df,
    fill_alpha=0.6,
    color=factor_cmap(
        "is_prime", palette=list(cmap.values()), factors=list(cmap.keys())
    ),
)
# legend_field="is_prime",


# color=factor_cmap('unknown', palette=list(cmap.values()), factors=list(cmap.keys())))

# %%
# r.data_source.data # this works

# %%
text_props = {"source": df, "text_align": "left", "text_baseline": "middle"}
x = dodge("column", -0.4, range=p.x_range)
y = dodge("row", -0.0, range=p.y_range)

p.text(x=x, y=y, text="number", text_font_style="bold", **text_props)


p.outline_line_color = None
p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_standoff = 0
handle = show(p, notebook_handle=True)


def update_is_prime(int_num: int, value: str) -> None:
    df.at[int_num - 1, "is_prime"] = str(value)
    r.data_source.data.update(dict(is_prime=df.is_prime.to_numpy()))
    push_notebook(handle=handle)


# %%
update_is_prime(2, True)

# %%

# %%

# %%

# %%
df.at[3, "is_prime"] = str(True)


# %%
df.at[3, "integer"]

# %%

# %%
# del df.at[100,'is_prime']
df.at[99, "is_prime"] = str(False)

# %%
r.data_source.data.update(dict(is_prime=df.is_prime.to_numpy()))
push_notebook(handle=handle)

# %%
df.at[10, "is_prime"] = str(False)
push_notebook(handle=handle)


# %%
handle

# %%
p.x_range.to_json_string(0)

# %%

# %%
help(factor_cmap)

# %%
factor_cmap("is_prime", palette=list(cmap.values()), factors=list(cmap.keys()))

# %%
help(p.rect)

# %%

# %%

# %%

# %%
