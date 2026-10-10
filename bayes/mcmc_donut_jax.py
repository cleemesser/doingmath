# %%
import sys
from typing import NamedTuple

import arviz as az
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

# %% [markdown]
# $$
# log[P(x')] - log[P(x)] > log U \implies P(x')/P(x) > U
# $$

# %%


def make_metropolis_step(log_density, step_size=0.01):
    def metropolis_hastings(state, key):
        key1, key2 = jr.split(key)
        new_position = state.position + step_size * jr.normal(key1, shape=(2,))
        new_log_density = log_density(new_position)
        accepted = new_log_density - state.log_density > jnp.log(jr.uniform(key2))

        state = State(
            position=jnp.where(accepted, new_position, state.position),
            log_density=jnp.where(accepted, new_log_density, state.log_density),
        )

        return state, state  # not a mistake, setting up for jax scan operation

    return metropolis_hastings


class State(NamedTuple):
    position: jax.Array  # shape (2,)
    log_density: jax.Array  # scalar, cached so each step evaluates log_density once


def donut_log_density(x, radius=2.6, sigma2=0.033):
    r"""Unnormalized log density of a ring ("donut") in the plane.

    x is a point of shape (2,). The density should peak on the circle
    ||x|| = radius and fall off with distance from it at a width set by sigma.
    sigma2 := variance of the gaussian $\sigma^2$
    """

    r = jnp.linalg.norm(x)
    # TODO: return the log density at x (a scalar)
    # return log(exp(-1/2 * delta /sigma2))
    # where delta is distance from radius
    return -((r - radius) ** 2 / (2 * sigma2))


def run_chain(key, log_density, initial_position, num_steps=10_000, step_size=0.01):
    step = jax.jit(make_metropolis_step(log_density, step_size))
    initial_state = State(initial_position, log_density(initial_position))
    keys = jr.split(key, num_steps)
    # draws
    _, states = jax.lax.scan(
        step,
        initial_state,
        xs=keys,  # jr.split(jr.key(0), 128_000)
    )
    return states


# %% [markdown]
# ## Running many chains at once with vmap
#
# Rule: write the code for one chain, then `vmap` it. `jax.vmap(f)` maps over
# axis 0 of every leaf of every argument, so every input to the batched step
# needs a leading chain axis. Nothing inside `make_metropolis_step` changes.
#
# 1. **Batch the starting state.** `State` is a `NamedTuple`, which JAX treats as
#    a pytree (a container whose leaves are arrays), so each field needs the
#    chain axis:
#
#    | field         | one chain | C chains |
#    |---------------|-----------|----------|
#    | `position`    | `(2,)`    | `(C, 2)` |
#    | `log_density` | `()`      | `(C,)`   |
#
#    Repeat the start point with `jnp.broadcast_to(initial_position, (C, 2))`,
#    and compute the log densities with `jax.vmap(log_density)`. The density is
#    written for one point: `jnp.linalg.norm(x)` with no `axis` would return one
#    number for the whole `(C, 2)` array.
#
# 2. **Make a grid of keys.** `jr.split(key, (num_steps, C))` gives an
#    independent key for every step of every chain. `lax.scan` takes one row per
#    step, and `vmap` gives one key from that row to each chain. Build the keys
#    from the `key` argument, never from a fixed `jr.key(0)`, so that the seed
#    controls the run.
#
# 3. **Scan the batched step:** `lax.scan(jax.vmap(step), vinit, xs=keys)`. All
#    C chains advance together, and each step is one vectorized operation.
#
# 4. **Read the output with axes counted from the end.** Positions have shape
#    `(num_steps, C, 2)` instead of `(num_steps, 2)`. Use `axis=-1` and
#    `positions[..., i]` for the x/y axis, so that the same statistics code works
#    for both shapes (see `print_chain_stats`). `axis=1` or `positions[:, 1]`
#    would pick the chain axis.
#
# Pooled statistics can mislead: chains that start at the origin walk out in
# different directions, so together they cover the whole ring even when each
# chain covers only a small arc. Check per-chain values as well.

# %%


def run_chain_vmap(
    key, log_density, initial_position, num_steps=10_000, step_size=0.01, num_chains=128
):
    """
    exactly the same but vmap the function
    """
    step = jax.jit(make_metropolis_step(log_density, step_size))
    # initial_state = State(initial_position, log_density(initial_position))
    initial_positions = jnp.broadcast_to(initial_position, (num_chains, 2))

    vinit = State(initial_positions, jax.vmap(log_density)(initial_positions))

    # jr.split wth a tuple returns a grid of independent keys with shape (num_steps, num_chains)
    keys = jr.split(key, (num_steps, num_chains))
    # draws
    _, states = jax.lax.scan(
        jax.vmap(step),
        vinit,  # init=vinit in presentation
        xs=keys,
    )
    return states


def pygame_viz_donut(positions, radius, sigma2):
    import pygame

    # Initialize Pygame
    pygame.init()
    WIDTH, HEIGHT = 600, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Mathematical Visualization: MCMC donut distribution")
    clock = pygame.time.Clock()

    # Colors
    BACKGROUND = (22, 22, 29)
    AXIS_COLOR = (100, 100, 110)
    GRID_COLOR = (40, 40, 50)
    PLOT_COLOR = (0, 255, 200)

    # Math Grid Config
    SCALE = 50  # Pixels per 1 mathematical unit
    CENTER_X = WIDTH // 2
    CENTER_Y = HEIGHT // 2

    def to_screen(x, y):
        """Converts Cartesian math coordinates to Pygame screen space."""
        return (CENTER_X + int(x * SCALE), CENTER_Y - int(y * SCALE))

    # Main Loop
    time_offset = 0.0
    point_mark = 0

    POINTS_PER_STEP = len(positions) // (100)
    point_max = len(positions) / POINTS_PER_STEP
    running = True

    while running:
        clock.tick(60)  # Caps framework at 60 FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear screen
        screen.fill(BACKGROUND)

        # 1. Draw Grid Lines
        for x_val in range(-int(CENTER_X / SCALE) - 1, int(CENTER_X / SCALE) + 1):
            pt1 = to_screen(x_val, -HEIGHT)
            pt2 = to_screen(x_val, HEIGHT)
            pygame.draw.line(screen, GRID_COLOR, (pt1[0], 0), (pt1[0], HEIGHT), 1)

        for y_val in range(-int(CENTER_Y / SCALE) - 1, int(CENTER_Y / SCALE) + 1):
            pt1 = to_screen(-WIDTH, y_val)
            pt2 = to_screen(WIDTH, y_val)
            pygame.draw.line(screen, GRID_COLOR, (0, pt1[1]), (WIDTH, pt1[1]), 1)

        # 2. Draw Main Cartesian Axes
        pygame.draw.line(
            screen, AXIS_COLOR, (0, CENTER_Y), (WIDTH, CENTER_Y), 2
        )  # X-axis
        pygame.draw.line(
            screen, AXIS_COLOR, (CENTER_X, 0), (CENTER_X, HEIGHT), 2
        )  # Y-axis

        # 2.5 draw circle
        pygame.draw.circle(screen, GRID_COLOR, (CENTER_X, CENTER_Y), radius * SCALE)
        # 3. Calculate and Plot the points

        points = [
            to_screen(xx.item(), yy.item())
            for (xx, yy) in positions[0 : point_mark * POINTS_PER_STEP]
        ]
        # Loop across the pixel width of the screen to sample math points
        # for pixel_x in range(WIDTH):
        #     # Convert screen X back to math x
        #     math_x = (pixel_x - CENTER_X) / SCALE

        #     # Calculate Math Y
        #     math_y = math.sin(math_x - time_offset) * 2.0  # Amplitude of 2 units

        #     # Convert Math Y to Screen Y
        #     _, pixel_y = to_screen(math_x, math_y)
        #     points.append((pixel_x, pixel_y))

        # Draw continuous curve lines between calculated points
        if len(points) > 1:
            pygame.draw.lines(screen, PLOT_COLOR, False, points, 3)

        # Animate time forward
        time_offset += 0.05
        point_mark += 1
        if point_mark >= point_max:
            point_mark = 0
            running = False
        pygame.display.flip()

    pygame.quit()


def pygame_viz_chains(positions, radius, sigma2, duration=30.0):
    """Animate many chains growing in parallel; the whole run takes `duration` seconds.

    positions has shape (num_steps, num_chains, 2), the output of run_chain_vmap.
    The step shown is tied to the clock, not the frame count, so a slow frame
    skips ahead instead of stretching the run. After the last step the final
    picture stays up until the window is closed or Esc is pressed.
    Space pauses and resumes the animation, and Enter restarts it from step 1.
    """
    import numpy as np
    import pygame

    positions = np.asarray(positions)  # one transfer off the device, not one per point
    num_steps, num_chains, _ = positions.shape

    pygame.init()
    WIDTH, HEIGHT = 700, 700
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"MCMC donut: {num_chains} chains in parallel")
    clock = pygame.time.Clock()
    # pygame.font is missing from some builds (e.g. pygame 2.6.1 on Python 3.14);
    # without it, the step counter goes in the window title instead.
    try:
        font = pygame.font.SysFont(None, 24)
    except ImportError, NotImplementedError:
        font = None

    # Colors
    BACKGROUND = (22, 22, 29)
    AXIS_COLOR = (100, 100, 110)
    GRID_COLOR = (40, 40, 50)
    TEXT_COLOR = (200, 200, 210)

    # Math Grid Config
    SCALE = 100  # Pixels per 1 mathematical unit, so the ring (r = 2.6) fits
    CENTER_X = WIDTH // 2
    CENTER_Y = HEIGHT // 2

    # Screen coordinates for every step of every chain, computed once: (steps, chains, 2)
    pixels = np.empty(positions.shape, dtype=int)
    pixels[..., 0] = CENTER_X + (positions[..., 0] * SCALE).astype(int)
    pixels[..., 1] = CENTER_Y - (positions[..., 1] * SCALE).astype(int)

    # One hue per chain, spread around the color wheel
    colors = []
    for c in range(num_chains):
        color = pygame.Color(0)
        color.hsva = (360 * c / num_chains, 65, 100, 100)
        colors.append(color)

    # Static background: grid, axes, and the ring with a +/- 2 sigma band
    background = pygame.Surface((WIDTH, HEIGHT))
    background.fill(BACKGROUND)
    for x_val in range(-int(CENTER_X / SCALE) - 1, int(CENTER_X / SCALE) + 2):
        x_px = CENTER_X + x_val * SCALE
        pygame.draw.line(background, GRID_COLOR, (x_px, 0), (x_px, HEIGHT), 1)
    for y_val in range(-int(CENTER_Y / SCALE) - 1, int(CENTER_Y / SCALE) + 2):
        y_px = CENTER_Y - y_val * SCALE
        pygame.draw.line(background, GRID_COLOR, (0, y_px), (WIDTH, y_px), 1)
    pygame.draw.line(background, AXIS_COLOR, (0, CENTER_Y), (WIDTH, CENTER_Y), 2)
    pygame.draw.line(background, AXIS_COLOR, (CENTER_X, 0), (CENTER_X, HEIGHT), 2)
    sigma = sigma2**0.5
    for r in (radius - 2 * sigma, radius, radius + 2 * sigma):
        pygame.draw.circle(background, AXIS_COLOR, (CENTER_X, CENTER_Y), r * SCALE, 1)

    # Trails are drawn incrementally onto their own surface: each frame adds only the
    # new segments, instead of redrawing up to num_steps * num_chains points.
    trails = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    drawn = 0  # index of the last step already on the trails surface
    shown_in_title = None

    # Keys: space pauses and resumes, Enter restarts, Escape quits.
    # elapsed_ms advances only while not paused, so a pause freezes the pacing.
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
                    trails.fill((0, 0, 0, 0))
                    drawn = 0
        if not paused:
            elapsed_ms += dt

        # Which step should be showing now, from elapsed time
        elapsed = elapsed_ms / 1000
        target = min(num_steps - 1, int(elapsed / duration * (num_steps - 1)))

        if target > drawn:
            for c in range(num_chains):
                pygame.draw.lines(
                    trails, colors[c], False, pixels[drawn : target + 1, c], 1
                )
            drawn = target

        screen.blit(background, (0, 0))
        screen.blit(trails, (0, 0))
        for c in range(num_chains):
            pygame.draw.circle(screen, colors[c], pixels[drawn, c], 3)

        label = (
            f"step {drawn + 1:,} / {num_steps:,}    t = {min(elapsed, duration):4.1f} s"
            + ("    [paused]" if paused else "")
        )
        if font is not None:
            screen.blit(font.render(label, True, TEXT_COLOR), (10, 10))
        elif (drawn, paused) != shown_in_title:
            pygame.display.set_caption(f"MCMC donut: {num_chains} chains    {label}")
            shown_in_title = (drawn, paused)

        pygame.display.flip()

    pygame.quit()


def print_chain_stats(positions, burn_in=0):
    """Summarize samples of shape (num_steps, 2) or (num_steps, num_chains, 2).

    Every axis is counted from the end (x/y is always last), so the same code
    works with or without a chain axis. burn_in drops that many initial steps.

    ESS stands for effective sample size. It estimates how many independent draws your
    correlated MCMC draws are worth. Each step of a chain starts from the previous one, so
    neighboring draws are similar. 1,000 correlated draws can carry as much information as
    only 50 independent ones. A low ESS compared with the number of draws means the chain
    mixes slowly.
    """
    positions = positions[burn_in:]
    moved = jnp.any(positions[1:] != positions[:-1], axis=-1)
    radii = jnp.linalg.norm(positions, axis=-1)
    angles = jnp.arctan2(positions[..., 1], positions[..., 0])

    print(f"acceptance rate: {moved.mean():.3f}")
    print(f"radius mean {radii.mean():.3f}, std {radii.std():.3f}")
    print(f"angle range covered: {angles.min():.2f} to {angles.max():.2f} rad")
    if positions.ndim == 3:
        per_chain = moved.mean(axis=0)
        print(
            f"per-chain acceptance: min {per_chain.min():.3f}, max {per_chain.max():.3f}"
        )


# %% [markdown]
# ## Chain diagnostics with ArviZ
#
# `print_chain_stats` pools all chains, so it cannot tell whether each chain
# explored the whole ring. ArviZ's diagnostics compare the chains with each other:
#
# - `r_hat` compares the variance between chains with the variance within each
#   chain. Near 1.00 means the chains agree; above about 1.01 means they have not
#   mixed. ArviZ 1.3 returns `nan` for a single chain, so `chain_diagnostics`
#   splits a lone chain into its first and second halves and treats them as two
#   chains. That still detects a chain whose two halves explore different arcs.
# - `ess_bulk` and `ess_tail` are effective sample sizes: how many independent
#   draws the correlated chains are worth, for the center and for the tails.
#
# ArviZ wants arrays laid out as `(chain, draw)`, but `lax.scan` returns
# `(draw, chain, 2)`, so the positions are transposed first.
#
# Which quantities to check matters for this target:
#
# - `r` (the radius) measures movement *across* the ring, the easy direction.
# - `x` and `y` measure movement *around* the ring, the hard direction. Chains
#   that sit on different arcs disagree on x and y, so `r_hat` rises.
# - Do not use the raw angle from `arctan2`. It jumps from π to −π, so a chain
#   that circles past that point looks as if it crossed the plane, and its
#   `r_hat` and ESS are meaningless. x and y carry the angle without the jump.
#
# Results with 128 chains from the origin and 1,000 burn-in steps:
#
# | step_size | r_hat x, y | ess_bulk x | r_hat r | sd of r |
# |-----------|------------|------------|---------|---------|
# | 0.01      | 3.8, 3.6   | 137        | 1.14    | 0.24    |
# | 0.2       | 1.10, 1.10 | 884        | 1.00    | 0.1813  |
#
# At 0.2 the radius is fully mixed: its sd matches the exact √0.033 = 0.1817.
# But each chain still does not go around the ring often enough for all chains
# to agree on x and y. Starting the chains far apart in angle is what makes
# `r_hat` able to detect that.

# %%


def chain_diagnostics(positions, burn_in=0):
    """ArviZ summary of x, y, and radius for positions of shape (num_steps, 2)
    or (num_steps, num_chains, 2). Returns the summary DataFrame."""
    # float64: in float32, ArviZ can return nan on very long runs (about 10M draws)
    positions = np.asarray(positions, dtype=np.float64)[burn_in:]
    if positions.ndim == 2:  # one chain: use its two halves as two chains
        half = len(positions) // 2
        positions = np.stack([positions[:half], positions[half : 2 * half]], axis=1)
        positions = positions.transpose(1, 0, 2)  # (chain, draw, 2), as ArviZ expects
        idata = az.from_dict(
            {
                "posterior": {
                    "x": positions[..., 0],
                    "y": positions[..., 1],
                    "r": np.linalg.norm(positions, axis=-1),
                }
            }
        )
        summary = az.summary(idata)
        print(summary[["mean", "sd", "ess_bulk", "ess_tail", "r_hat"]])
    return summary


if __name__ == "__main__":
    # start = jnp.array([2.6, 0.0])  # on the ring, so the chain needs no burn-in
    start = jnp.array([0.0, 0.0])  # at the origin, so the chain needs burn-in
    burn_in = 1_000

    # first run the simple chain: positions has shape (num_steps, 2)
    states = run_chain(jr.key(0), donut_log_density, start)
    positions = states.position
    print(f"single chain, positions {positions.shape}, burn-in {burn_in}")
    print_chain_stats(positions, burn_in)
    chain_diagnostics(positions, burn_in)

    # now run the vmap parallel version: positions has shape (num_steps, num_chains, 2)
    # default step_size=0.01 will take very small steps try different ones 0.2 is a good
    # one from a coverage point of view
    # can go to 128 chains, but easier to see at first with a few
    vstates = run_chain_vmap(
        jr.key(0), donut_log_density, start, step_size=0.5, num_chains=128
    )
    vpositions = vstates.position
    print(
        f"\n{vpositions.shape[1]} chains (vmap), positions {vpositions.shape}, burn-in {burn_in}"
    )
    print_chain_stats(vpositions, burn_in)
    chain_diagnostics(vpositions, burn_in)

    # optional viz: run with --viz to watch the vmap chains grow (30 s)
    # pygame_viz_donut(positions, 2.6, 0.33)
    if "--viz" in sys.argv:
        print()
        print("starting visualization")
        pygame_viz_chains(vpositions, 2.6, 0.033, duration=30.0)


# %% [markdown]
# ### Evaluating the quality of a sampling run
#
# ESS stands for effective sample size. It estimates how many independent
# draws the correlated MCMC draws are worth. Each step of a chain starts from
# the previous one, so neighboring draws are similar. 1,000 correlated draws
# can carry as much information as only 50 independent ones. A low ESS
# compared with the number of draws means the chain mixes slowly.
#
# R̂ (R-hat) is the potential scale reduction factor, introduced by Gelman and
# Rubin in 1992. The R stands for "reduction". It estimates how much the spread
# of the current draws would shrink if the chains ran forever. The hat marks it
# as an estimate, the usual statistics notation. A value of 1 means further
# sampling would not narrow anything, because the chains already agree.
#
# The definition compares the variance between chains with the variance within
# chains. Take m chains of n draws each, where θᵢⱼ is draw i of chain j:
#
# ```
# chain means       θ̄ⱼ  = (1/n) Σᵢ θᵢⱼ
# overall mean      θ̄   = (1/m) Σⱼ θ̄ⱼ
#
# between-chain     B   = n/(m−1) · Σⱼ (θ̄ⱼ − θ̄)²
# within-chain      W   = (1/m) Σⱼ sⱼ²,   sⱼ² = 1/(n−1) · Σᵢ (θᵢⱼ − θ̄ⱼ)²
#
# pooled estimate   var⁺ = (n−1)/n · W  +  B/n
#
#                   R̂   = √(var⁺ / W)
# ```
#
# W comes only from the spread inside each chain. If the chains have not yet
# explored the whole distribution, each chain sees only part of it, so W
# underestimates the true variance. var⁺ adds the disagreement between chain
# means, so it overestimates the true variance while the chains are still
# spread out. As the chains mix, both approach the true variance and their
# ratio approaches 1. The square root makes R̂ a ratio of standard deviations,
# hence "scale". At 10,000 steps with step_size=0.01, the donut chains sat on
# different arcs. Their means of x differed a lot, so B was large compared with
# W, which gave R̂ = 3.8.
#
# ArviZ does not compute the 1992 version. It uses the improved version from
# Vehtari, Gelman, Simpson, Carpenter, and Bürkner (2021), which makes two
# changes. First, it splits each chain in half and treats the halves as
# separate chains (split-R̂), which also catches a chain that is still
# drifting. Second, it replaces the draws with their ranks, mapped to normal
# quantiles, before applying the formula. That makes it work for heavy-tailed
# distributions, where raw variances are unreliable. It also computes the same
# statistic on |θ − median| (the "folded" draws), which catches chains with the
# same center but different spread, and it reports the larger of the two
# values. The same paper lowered the recommended threshold from the older 1.1
# to 1.01.
#
# NB: R̂ can only detect disagreement among the chains that were run. If all
# the chains are stuck in the same region, B is small and R̂ looks fine. That
# is why the formula comes paired with the advice to start chains far apart.
# Running 128 chains that spread out from the origin makes a strong test.

# %% [markdown]
# ### R̂ for a single chain
#
# ArviZ 1.3 does not split a lone chain in half on its own: it returns `nan`
# for `r_hat` when there is only one chain. `chain_diagnostics` therefore splits
# a single chain into its first and second halves and treats them as two
# chains. This is the same idea as ArviZ's split-R̂, which it applies to each
# chain when there are several.
#
# With step_size=0.01 for the single chain, and step_size=0.2 with 2 chains
# for the vmap run, the output was:
#
# ```
# single chain (halves as chains)       2 chains, step 0.2
#    ess_bulk  r_hat                       ess_bulk  r_hat
# x         4   1.36                    x        38   1.02
# y         3   1.45                    y         5   1.27
# r         9   1.19                    r      2514   1.00
# ```
#
# The single chain fails everywhere. Its 9,000 draws of x are worth about 4
# independent samples, and its two halves disagree. The two-chain run mixes
# the radius fully, with sd 0.181, matching the exact 0.1817. But y has r_hat
# 1.27 and an ESS of 5, so the two chains still disagree about where they are
# on the ring. x happens to look fine at 1.02 here. With only two chains, how
# the chains happen to sit on the ring decides which of x and y looks good.
# That is one reason to run more chains, spread out at the start, when the
# angle is the hard direction.
#
# Splitting a chain catches a chain that is still drifting, because then its
# first and second halves look different. It cannot catch a chain that is
# stuck in one region the whole time: both halves look the same, and r_hat
# reports good mixing. Only chains started in different places can detect
# that, which is why the 128 chains from the origin are a stronger test than
# one long chain.

# %% [markdown]
# ### 128 chains at step_size=0.2
#
# Running `run_chain_vmap` with step_size=0.2 and num_chains=128, and dropping
# 1,000 burn-in steps, gave:
#
# ```
# positions (10000, 128, 2) burn-in 1000
# acceptance rate: 0.680
# radius mean 2.613, std 0.181
# angle range covered: -3.14 to 3.14 rad
# per-chain acceptance: min 0.670, max 0.693
#
#      mean      sd ess_bulk ess_tail r_hat
# x      -0    1.85      884     2778  1.10
# y     0.1    1.85      897     2482  1.10
# r  2.6127  0.1813   156369   209020  1.00
# ```
#
# The radius is fully mixed. Its r_hat is 1.00, and its standard deviation of
# 0.1813 matches the exact √0.033 = 0.1817. The acceptance rate of 0.68 is
# fairly even across chains, from 0.67 to 0.69.
#
# The angle is not mixed. x and y have r_hat 1.10, well above 1.01. The ESS for
# x is 884 out of about 1.15 million draws (9,000 × 128). That is about 7
# independent draws per chain. The pooled means of x and y near 0 look right,
# but only because 128 chains started in every direction cancel each other
# out. That is the pooled-statistics trap.
#
# A rough estimate explains why. The ring's circumference is 2π × 2.6 ≈ 16.3.
# A random walk with steps of about 0.2 needs about (16.3 / 0.2)² ≈ 6,600 moves
# to spread over a distance that long. With 68% of proposals accepted, that is
# about 10,000 steps. So in 9,000 draws, each chain goes about once around the
# ring. For r_hat to approach 1.01, each chain needs many circuits. Larger
# steps do not help much, because a step much larger than the ring's width
# (σ ≈ 0.18) usually lands off the ring and is rejected.
#
# Note: this is the main weakness of random-walk Metropolis. The distance it
# covers grows only with the square root of the number of steps, so traveling
# twice as far takes four times as many steps. NUTS uses the gradient to move
# along the ring in long, nearly straight paths, so it covers the angle much
# faster for the same number of density evaluations (see
# `mcmc_donut_part2.py`).

# %% [markdown]
# ### The simple fix for this toy problem: get more samples
#
# About 10 times longer is enough. Longer runs with 128 chains at
# step_size=0.2 gave:
#
# ```
#  num_steps    time    x: ess_bulk  r_hat    y: ess_bulk  r_hat
#     10,000    <1 s          884    1.10           897    1.10
#     30,000     1 s        3,156    1.03         3,647    1.03
#    100,000     2 s       14,095    1.01        14,699    1.01
# ```
#
# At 100,000 steps, x and y reach the usual 1.01 threshold, with about 14,000
# effective draws each. All 128 chains take 2 seconds, because vmap runs them
# as one vectorized operation. The radius stays fully mixed throughout, with
# sd 0.1812 against the exact 0.1817. ESS grows faster than the step count,
# because each extra lap around the ring counts for more once the chains have
# spread out.
#
# So these are the pathways to a solution:
#
# - Use `num_steps=100_000` when trustworthy r_hat values are needed, aiming
#   for about 1.01. The positions take about 100 MB, which is fine.
# - Keep fewer steps for the animation. `pygame_viz_chains` computes screen
#   coordinates for every step, and 30 seconds cannot show 100,000 steps
#   anyway. Either keep 10,000 steps for viewing, or pass only every 10th
#   step, as in `vpositions[::10]`.
# - For a real problem, change the sampler instead of running longer. Running
#   longer works here because 2 seconds is nothing. When each density
#   evaluation is expensive, the better fix is a gradient-based sampler such
#   as NUTS, which moves along the ring in long paths instead of diffusing.

# %% [markdown]
# ### float32 and long runs
#
# The 100,000-step run first returned `nan` for y's r_hat and ESS, along with
# numerical warnings from ArviZ. JAX returns float32, and with 12.7 million
# values, ArviZ's calculations lost too much precision. `chain_diagnostics`
# now converts the positions to float64 first.
#
# The problem was invisible at 10,000 steps and appeared only at scale, and
# only for y, not x. That pattern is typical of precision loss: whether it
# appears depends on the exact values. JAX uses float32 by default, which is
# fast and usually enough for sampling. But statistics summed over millions of
# values need float64. Another option is
# `jax.config.update("jax_enable_x64", True)` at the top of the script, which
# makes the whole run use float64. It doubles memory use and slows things down
# on GPUs.
