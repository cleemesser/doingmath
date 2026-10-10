# %% [markdown]
# # The donut, part 2: gradient-based samplers
#
# `mcmc_donut_jax.py` sampled the donut with a hand-written random-walk
# Metropolis sampler. Getting the angle right (r_hat ≤ 1.01 for x and y) took
# about 100,000 steps per chain. This notebook asks how many draws three NUTS
# implementations need for the same result:
#
# | sampler                  | what it is                                          |
# |--------------------------|-----------------------------------------------------|
# | naive JAX Metropolis     | `make_metropolis_step` from part 1, step size 0.2   |
# | BlackJAX NUTS            | NUTS built from BlackJAX parts, in JAX              |
# | NumPyro NUTS             | NumPyro's NUTS, given the log density directly      |
# | nutpie (PyMC)            | a Rust NUTS sampler; PyMC model compiled with numba |
#
# NUTS (the No-U-Turn Sampler) is Hamiltonian Monte Carlo that picks its own
# trajectory length. Each draw follows the gradient of the log density for many
# small "leapfrog" steps, so one draw costs many gradient evaluations. A fair
# comparison therefore counts evaluations, not draws: Metropolis costs one
# density evaluation per draw, NUTS one gradient evaluation per leapfrog step.
# A gradient costs a small multiple of a density evaluation, so the counts
# below slightly favor NUTS.
#
# "Getting the distribution right" is checked two ways:
#
# 1. **Mixing:** r_hat ≤ 1.01 for x, y, and r, from ArviZ.
# 2. **Exact values:** the donut's radius distribution is known, so E[r] and
#    sd(r) from the draws are compared with values from numerical integration.
#    The angle is uniform, so the mean direction (the mean of the unit vectors
#    x/r, y/r) must be close to zero.

# %%
import os

# PyTensor's C compiler fails on this Mac (see pymc_numpyro_example.py); nutpie
# uses numba instead. Must be set before pymc is imported.
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

import sys
import time

import arviz as az
import blackjax
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import nutpie
import pymc as pm
import pytensor.tensor as pt
from numpyro.infer import MCMC, NUTS

from mcmc_donut_jax import State, donut_log_density, make_metropolis_step

RADIUS, SIGMA2 = 2.6, 0.033  # the defaults of donut_log_density
NUM_CHAINS = 16
NUTS_WARMUP = 500

# Every sampler except nutpie starts from the same 16 random points. NUTS cannot
# start exactly at the origin: the gradient of ||x|| is undefined there.
START_KEY, MH_KEY, BJ_KEY, NP_KEY = jr.split(jr.key(0), 4)
STARTS = jr.uniform(START_KEY, (NUM_CHAINS, 2), minval=-3.0, maxval=3.0)

# %% [markdown]
# ## Exact reference values
#
# In polar coordinates the density is p(r, θ) ∝ r · exp(−(r − R)² / 2σ²). The
# factor r is the area of a thin ring at radius r. So the radius is not exactly
# Gaussian: its mean is slightly larger than R, about R + σ²/R = 2.6127.

# %%


def exact_radius_moments(radius=RADIUS, sigma2=SIGMA2):
    r = np.linspace(1e-6, radius + 12 * np.sqrt(sigma2), 200_001)
    weight = r * np.exp(-((r - radius) ** 2) / (2 * sigma2))
    weight /= np.trapezoid(weight, r)
    mean = np.trapezoid(r * weight, r)
    sd = np.sqrt(np.trapezoid((r - mean) ** 2 * weight, r))
    return mean, sd


EXACT_R_MEAN, EXACT_R_SD = exact_radius_moments()
print(f"exact E[r] = {EXACT_R_MEAN:.4f}, sd(r) = {EXACT_R_SD:.4f}")

# %% [markdown]
# ## The four samplers
#
# Each function returns a dict with:
#
# - `positions`: shape `(draws, chains, 2)`, the layout used in part 1,
# - `evals_per_draw`: evaluations spent on each draw, averaged over chains
#   (used by the visualizer to compare samplers at equal cost),
# - `warmup_evals` and `evals`: totals for warmup and for the kept draws,
# - `seconds`: wall time, including JIT or numba compilation.

# %%


def run_metropolis(num_steps=100_000, burn_in=1_000, step_size=0.2):
    t = time.time()
    step = make_metropolis_step(donut_log_density, step_size)
    vinit = State(STARTS, jax.vmap(donut_log_density)(STARTS))
    keys = jr.split(MH_KEY, (num_steps, NUM_CHAINS))
    _, states = jax.lax.scan(jax.vmap(step), vinit, keys)
    positions = np.asarray(states.position, dtype=np.float64)[burn_in:]
    return {
        "name": "naive JAX Metropolis",
        "positions": positions,
        "evals_per_draw": np.ones(len(positions)),
        "warmup_evals": burn_in * NUM_CHAINS,
        "evals": positions.shape[0] * NUM_CHAINS,
        "seconds": time.time() - t,
    }


def run_blackjax(draws=500, warmup=NUTS_WARMUP):
    def one_chain(key, x0):
        warmup_key, draw_key = jr.split(key)
        adapt = blackjax.window_adaptation(blackjax.nuts, donut_log_density)
        (state, params), adapt_info = adapt.run(warmup_key, x0, num_steps=warmup)
        nuts_step = blackjax.nuts(donut_log_density, **params).step

        def one_step(state, k):
            state, info = nuts_step(k, state)
            return state, (state.position, info.num_integration_steps)

        _, (positions, n_steps) = jax.lax.scan(
            one_step, state, jr.split(draw_key, draws)
        )
        return positions, n_steps, adapt_info.info.num_integration_steps

    t = time.time()
    positions, n_steps, warm_steps = jax.jit(jax.vmap(one_chain))(
        jr.split(BJ_KEY, NUM_CHAINS), STARTS
    )
    positions = np.asarray(positions, dtype=np.float64).transpose(1, 0, 2)
    n_steps = np.asarray(n_steps)  # (chains, draws)
    return {
        "name": "BlackJAX NUTS",
        "positions": positions,
        "evals_per_draw": n_steps.mean(axis=0),
        "warmup_evals": int(np.sum(warm_steps)),
        "evals": int(n_steps.sum()),
        "seconds": time.time() - t,
    }


def run_numpyro(draws=500, warmup=NUTS_WARMUP):
    t = time.time()
    mcmc = MCMC(
        NUTS(potential_fn=lambda x: -donut_log_density(x)),
        num_warmup=warmup,
        num_samples=draws,
        num_chains=NUM_CHAINS,
        chain_method="vectorized",
        progress_bar=False,
    )
    # Warmup as a separate call so that its leapfrog steps can be counted too.
    mcmc.warmup(
        NP_KEY, init_params=STARTS, collect_warmup=True, extra_fields=("num_steps",)
    )
    warm_steps = np.asarray(mcmc.get_extra_fields()["num_steps"])
    mcmc.run(mcmc.post_warmup_state.rng_key, extra_fields=("num_steps",))
    positions = np.asarray(mcmc.get_samples(group_by_chain=True), dtype=np.float64)
    n_steps = np.asarray(mcmc.get_extra_fields(group_by_chain=True)["num_steps"])
    return {
        "name": "NumPyro NUTS",
        "positions": positions.transpose(1, 0, 2),
        "evals_per_draw": n_steps.mean(axis=0),
        "warmup_evals": int(warm_steps.sum()),
        "evals": int(n_steps.sum()),
        "seconds": time.time() - t,
    }


def run_nutpie(draws=500, warmup=NUTS_WARMUP):
    # nutpie samples a PyMC model. pm.Flat gives an improper uniform prior on
    # the plane, and pm.Potential adds the donut log density to the model.
    with pm.Model() as donut_model:
        xy = pm.Flat("xy", shape=2)
        r = pt.sqrt(pt.sum(xy**2))
        pm.Potential("donut", -((r - RADIUS) ** 2) / (2 * SIGMA2))

    t = time.time()
    compiled = nutpie.compile_pymc_model(donut_model, backend="numba")
    # nutpie picks its own jittered starting points; it has no per-chain init.
    trace = nutpie.sample(
        compiled,
        draws=draws,
        tune=warmup,
        chains=NUM_CHAINS,
        seed=0,
        progress_bar=False,
    )
    positions = trace.posterior["xy"].values.transpose(1, 0, 2)  # (draw, chain, 2)
    n_steps = trace.sample_stats["n_steps"].values  # (chain, draw)
    return {
        "name": "nutpie (PyMC)",
        "positions": positions.astype(np.float64),
        "evals_per_draw": n_steps.mean(axis=0),
        "warmup_evals": int(trace.warmup_sample_stats["n_steps"].values.sum()),
        "evals": int(n_steps.sum()),
        "seconds": time.time() - t,
    }


# %% [markdown]
# ## Scoring a run
#
# `score` computes the ArviZ diagnostics for x, y, and r, the error of E[r] and
# sd(r) against the exact values, and the length of the mean direction. A run
# passes when r_hat ≤ 1.01 for all three quantities, E[r] is within 0.01 of the
# exact value, and the mean direction is shorter than 0.05.

# %%


def score(result):
    positions = result["positions"].transpose(1, 0, 2)  # (chain, draw, 2) for ArviZ
    radius = np.linalg.norm(positions, axis=-1)
    summary = az.summary(
        az.from_dict(
            {
                "posterior": {
                    "x": positions[..., 0],
                    "y": positions[..., 1],
                    "r": radius,
                }
            }
        )
    )
    unit = positions / radius[..., None]
    mean_direction = float(np.linalg.norm(unit.reshape(-1, 2).mean(axis=0)))
    ess_xy = float(min(summary.loc["x", "ess_bulk"], summary.loc["y", "ess_bulk"]))
    total_evals = result["warmup_evals"] + result["evals"]
    r_mean_error = abs(radius.mean() - EXACT_R_MEAN)
    rhat_max = float(summary["r_hat"].max())
    return {
        "draws": positions.shape[1],
        "rhat_max": rhat_max,
        "ess_xy": ess_xy,
        "ess_per_1k_evals": 1000 * ess_xy / total_evals,
        "total_evals": total_evals,
        "r_mean": float(radius.mean()),
        "r_sd": float(radius.std()),
        "mean_direction": mean_direction,
        "passes": rhat_max <= 1.01 and r_mean_error < 0.01 and mean_direction < 0.05,
    }


def print_scores(rows):
    print(
        f"{'sampler':<22}{'draws':>8}{'evals':>11}{'r_hat':>7}{'ESS x,y':>9}"
        f"{'ESS/1k':>8}{'E[r]':>8}{'sd(r)':>8}{'|dir|':>7}{'time':>7}  ok"
    )
    for result, s in rows:
        print(
            f"{result['name']:<22}{s['draws']:>8,}{s['total_evals']:>11,}"
            f"{s['rhat_max']:>7.3f}{s['ess_xy']:>9,.0f}{s['ess_per_1k_evals']:>8.2f}"
            f"{s['r_mean']:>8.4f}{s['r_sd']:>8.4f}{s['mean_direction']:>7.3f}"
            f"{result['seconds']:>6.1f}s  {'yes' if s['passes'] else 'NO'}"
        )
    print(
        f"{'exact':<22}{'':>8}{'':>11}{'':>7}{'':>9}{'':>8}"
        f"{EXACT_R_MEAN:>8.4f}{EXACT_R_SD:>8.4f}{0:>7.3f}"
    )


# %% [markdown]
# ## Visualizer
#
# Four panels, one per sampler, animated together for `duration` seconds. Every
# draw stays on screen as a dot, and the current position of each chain is a
# larger dot with a tail. A NUTS draw can land anywhere on the ring, so its
# tail is only the last move, a chord across the ring. That chord is the point:
# successive NUTS draws are far apart, so they are nearly independent.
# Metropolis moves are short, so its tail covers more draws.
#
# Two ways to pace the panels:
#
# - `pace="draws"`: every panel shows its whole run in `duration` seconds,
#   however many draws it has.
# - `pace="cost"`: time is proportional to evaluations spent. A sampler that
#   needs fewer evaluations finishes early and stops, so the panels compare
#   what each sampler achieves for the same amount of work. This counts only
#   the kept draws, not warmup. With the settings below, the NUTS panels
#   finish at about 10% to 15% of the time.
#
# Keys: space pauses and resumes the animation, Enter restarts it from the
# first draw, and Escape (or closing the window) quits.
#
# pygame's font modules do not load in this environment (pygame 2.6.1 on
# Python 3.14), so the labels are drawn once with matplotlib and copied in as
# images.

# %%


def text_surface(lines, fontsize=11, color="#c8c8d2"):
    """Render text lines to a pygame surface with matplotlib (no pygame fonts)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pygame

    fig = plt.figure(figsize=(4.4, 0.18 * len(lines) + 0.06), dpi=100)
    fig.patch.set_alpha(0.0)
    for i, line in enumerate(lines):
        fig.text(
            0.01,
            1 - (i + 0.8) / len(lines),
            line,
            fontsize=fontsize,
            color=color,
            family="monospace",
            va="baseline",
        )
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    h, w, _ = rgba.shape
    return pygame.image.frombuffer(rgba.tobytes(), (w, h), "RGBA")


def pygame_viz_samplers(results, scores, duration=30.0, pace="draws"):
    import pygame

    pygame.init()
    PANEL = 450
    WIDTH, HEIGHT = 2 * PANEL, 2 * PANEL
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"MCMC donut: four samplers (pace = {pace})")
    clock = pygame.time.Clock()

    # Colors (as in mcmc_donut_jax.py)
    BACKGROUND = (22, 22, 29)
    AXIS_COLOR = (100, 100, 110)
    GRID_COLOR = (40, 40, 50)
    BAR_COLOR = (0, 255, 200)
    SCALE = 56  # pixels per unit, so radius + band fits below the label

    def chain_colors(n):
        colors = []
        for c in range(n):
            color = pygame.Color(0)
            color.hsva = (360 * c / n, 65, 100, 100)
            colors.append(color)
        return colors

    panels = []
    max_cost = max(r["evals_per_draw"].sum() for r in results)
    for k, (result, s) in enumerate(zip(results, scores)):
        origin = ((k % 2) * PANEL, (k // 2) * PANEL)
        cx, cy = PANEL // 2, PANEL // 2 + 28
        pos = result["positions"]
        pixels = np.empty(pos.shape, dtype=int)
        pixels[..., 0] = cx + (pos[..., 0] * SCALE).astype(int)
        pixels[..., 1] = cy - (pos[..., 1] * SCALE).astype(int)

        background = pygame.Surface((PANEL, PANEL))
        background.fill(BACKGROUND)
        for v in range(-4, 5):
            pygame.draw.line(
                background, GRID_COLOR, (cx + v * SCALE, 0), (cx + v * SCALE, PANEL)
            )
            pygame.draw.line(
                background, GRID_COLOR, (0, cy - v * SCALE), (PANEL, cy - v * SCALE)
            )
        pygame.draw.line(background, AXIS_COLOR, (0, cy), (PANEL, cy), 1)
        pygame.draw.line(background, AXIS_COLOR, (cx, 0), (cx, PANEL), 1)
        sigma = SIGMA2**0.5
        for r in (RADIUS - 2 * sigma, RADIUS + 2 * sigma):
            pygame.draw.circle(background, AXIS_COLOR, (cx, cy), r * SCALE, 1)
        pygame.draw.rect(background, AXIS_COLOR, background.get_rect(), 1)
        label = text_surface(
            [
                result["name"],
                f"{s['draws']:,} draws/chain  {s['total_evals']:,} evals",
                f"ESS(x,y) {s['ess_xy']:,.0f}   r_hat {s['rhat_max']:.3f}",
            ]
        )
        background.blit(label, (6, 4))

        panels.append(
            {
                "origin": origin,
                "pixels": pixels,
                "background": background,
                "dots": pygame.Surface((PANEL, PANEL), pygame.SRCALPHA),
                "drawn": -1,
                "cum_cost": np.cumsum(result["evals_per_draw"]),
                "colors": chain_colors(pos.shape[1]),
                # Tail length in draws: a NUTS draw can jump across the ring, so
                # its tail is the last move only; Metropolis gets a longer tail.
                "tail": max(1, len(pos) // 2_000),
            }
        )

    # Keys: space pauses and resumes, Enter restarts, Escape quits.
    # elapsed_ms advances only while not paused, so a pause freezes the pacing.
    caption = pygame.display.get_caption()[0]
    elapsed_ms = 0
    paused = False
    running = True
    while running:
        dt = clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    elapsed_ms = 0
                    paused = False
                    for p in panels:
                        p["dots"].fill((0, 0, 0, 0))
                        p["drawn"] = -1
                pygame.display.set_caption(caption + ("  [paused]" if paused else ""))
        if not paused:
            elapsed_ms += dt

        frac = min(1.0, elapsed_ms / 1000 / duration)
        for p in panels:
            num_draws = len(p["pixels"])
            if pace == "cost":
                target = int(np.searchsorted(p["cum_cost"], frac * max_cost))
                target = min(target, num_draws - 1)
            else:
                target = int(frac * (num_draws - 1))

            # New draws become permanent dots (dimmed chain color)
            for i in range(p["drawn"] + 1, target + 1):
                for c, color in enumerate(p["colors"]):
                    dim = pygame.Color(color.r // 2, color.g // 2, color.b // 2)
                    p["dots"].set_at(tuple(p["pixels"][i, c]), dim)
            p["drawn"] = max(p["drawn"], target)

            surface = p["background"].copy()
            surface.blit(p["dots"], (0, 0))
            if p["drawn"] >= 0:
                lo = max(0, p["drawn"] - p["tail"])
                for c, color in enumerate(p["colors"]):
                    pts = p["pixels"][lo : p["drawn"] + 1, c]
                    if len(pts) > 1:
                        pygame.draw.lines(surface, color, False, pts, 1)
                    pygame.draw.circle(surface, color, pts[-1], 3)
            # progress bar: fraction of this sampler's draws shown
            done = (p["drawn"] + 1) / num_draws
            pygame.draw.rect(surface, BAR_COLOR, (0, PANEL - 4, int(done * PANEL), 4))
            screen.blit(surface, p["origin"])

        pygame.display.flip()

    pygame.quit()


# %% [markdown]
# ## Results
#
# Sweep with 16 chains, NUTS warmup 500 iterations ("evals" includes warmup):
#
# ```
# sampler                  draws      evals  r_hat  ESS x,y  ESS/1k    E[r]   sd(r)  |dir|  ok
# naive JAX Metropolis     9,000    160,000  1.101      108    0.68  2.6121  0.1818  0.110  NO
# naive JAX Metropolis    29,000    480,000  1.033      503    1.05  2.6129  0.1821  0.038  NO
# naive JAX Metropolis    99,000  1,600,000  1.008    1,731    1.08  2.6129  0.1812  0.033  yes
# BlackJAX NUTS              100    337,567  1.031      506    1.50  2.6104  0.1856  0.060  NO
# BlackJAX NUTS              250    416,257  1.013    1,182    2.84  2.6123  0.1831  0.034  NO
# BlackJAX NUTS              500    540,437  1.005    2,442    4.52  2.6129  0.1836  0.016  yes
# BlackJAX NUTS            1,000    791,391  1.003    4,590    5.80  2.6135  0.1822  0.005  yes
# NumPyro NUTS               100    335,322  1.021      546    1.63  2.6162  0.1748  0.049  NO
# NumPyro NUTS               250    413,790  1.016    1,230    2.97  2.6128  0.1784  0.020  NO
# NumPyro NUTS               500    534,872  1.011    2,372    4.43  2.6096  0.1828  0.015  NO
# NumPyro NUTS             1,000    801,952  1.003    5,031    6.27  2.6114  0.1808  0.008  yes
# nutpie (PyMC)              100    203,561  1.027      491    2.41  2.6119  0.1862  0.032  NO
# nutpie (PyMC)              250    251,915  1.010    1,082    4.30  2.6116  0.1872  0.037  yes
# nutpie (PyMC)              500    333,789  1.005    2,234    6.69  2.6071  0.1847  0.007  yes
# nutpie (PyMC)            1,000    501,049  1.003    4,455    8.89  2.6106  0.1850  0.004  yes
# exact                                                              2.6127  0.1812  0.000
# ```
#
# **Fewer draws, yes.** All three NUTS samplers reach r_hat ≤ 1.01 with 250 to
# 1,000 draws per chain, against 99,000 for Metropolis: 100 to 400 times fewer
# draws. Each NUTS draw is worth far more, because it travels a long way
# around the ring.
#
# **Less work, by a smaller factor.** Each NUTS draw costs 20 to 30 leapfrog
# steps, plus the warmup. Counting evaluations, NUTS reaches r_hat ≤ 1.01
# with 250,000 to 800,000 evaluations against 1.6 million, and gets 4 to 8
# times more ESS per evaluation. A gradient costs a few density
# evaluations, so the real compute advantage is smaller still.
#
# **Warmup dominates short runs.** At 100 draws, more than 80% of the NUTS
# evaluations are warmup. Below about 250 draws, adding draws is cheap compared
# with the warmup already paid.
#
# **nutpie is the most efficient here.** It needs about two-thirds of the
# leapfrog steps per draw of BlackJAX and NumPyro (about 21 against 31), and
# its warmup costs about 40% less. It gets the
# highest ESS per evaluation at every draw count and passes at 250 draws.
# BlackJAX and NumPyro run the same algorithm and perform alike.
#
# **The pass/fail line is noisy.** NumPyro at 500 draws missed r_hat ≤ 1.01 by
# 0.001, while BlackJAX passed. With a different seed the order can flip.
# Likewise nutpie's sd(r) of about 0.185 is noise, not bias: across four seeds
# at 1,000 draws it ranged from 0.1801 to 0.1850, with a Monte Carlo standard
# error of about 0.0014, and at 5,000 draws all four seeds gave 0.1810 to
# 0.1817. The sweep repeats the same seed, so its nutpie rows share that noise.
#
# **Metropolis is still better across the ring.** r is the easy direction, and
# small random steps of about the ring's width explore it well: Metropolis
# gives the most accurate sd(r) per evaluation. NUTS spends its long
# trajectories going around the ring.

# %% [markdown]
# ## Run the comparison
#
# `python bayes/mcmc_donut_part2.py` prints a sweep over draw counts and the
# final comparison. Add `--viz` to animate the comparison, and `--cost` to pace
# the animation by evaluations instead of draws.

# %%
if __name__ == "__main__":
    # How few draws does each sampler need? (NUTS warmup fixed at 500)
    sweep = []
    for steps in (10_000, 30_000, 100_000):
        result = run_metropolis(num_steps=steps)
        sweep.append((result, score(result)))
    for runner in (run_blackjax, run_numpyro, run_nutpie):
        for draws in (100, 250, 500, 1_000):
            result = runner(draws=draws)
            sweep.append((result, score(result)))
    print("\nSweep over draws per chain (16 chains):\n")
    print_scores(sweep)

    # The comparison shown in the visualizer
    results = [
        run_metropolis(num_steps=100_000),
        run_blackjax(draws=500),
        run_nutpie(draws=500),
        run_numpyro(draws=500),
    ]
    scores = [score(r) for r in results]
    print("\nComparison for the visualizer:\n")
    print_scores(list(zip(results, scores)))

    if "--viz" in sys.argv:
        pace = "cost" if "--cost" in sys.argv else "draws"
        pygame_viz_samplers(results, scores, duration=30.0, pace=pace)
