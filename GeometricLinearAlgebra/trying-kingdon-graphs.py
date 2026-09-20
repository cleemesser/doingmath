# %%
my_camera = [1, 0, 0, 1, 0, 0, 4.0]
alg2.graph(
    # 0x00FF00,
    # lambda: [100*v_mv, u_mv, w_mv], "Vectors u,v,w in 2D space",
    *[0x0000FF, alg2.vector([1, 0]), "e1"],
    grid=True,
    labels=True,
    camera=my_camera,
    lineWidth=3,
    # pga=True,  # did not work in vscode
    # conformal=True, # nope in vscode
    # gl=True,
    width="600px",
    height="400px",
    # style=dict(width="600px", height="400px"),
    scale=10.0,
)
# %%
# this gets me the 3D grid and points working in VS Code
alg301 = Algebra(3, 0, 1)
W, H = "1200px", "800px"
obj = alg301.graph(
    *[
        0x000000,
        alg301.vector([1, 0, 0, 0]).dual(),
        "O",
        0x990000,
        alg301.vector([1, 1, 0, 0]).dual(),
        "X",
        0x009900,
        alg301.vector([1, 0, 1, 0]).dual(),
        "Y",
        0x000099,
        alg301.vector([1, 0, 0, 1]).dual(),
        "Z",
    ],
    grid=True,
    labels=True,
    lineWidth=2,
    style=dict(width=W, height=H),
    scale=1.0,
)
obj
