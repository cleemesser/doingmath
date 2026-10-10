# %%
import sys
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr

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
    shown_in_title = -1

    start_ms = pygame.time.get_ticks()
    running = True
    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Which step should be showing now, from elapsed time
        elapsed = (pygame.time.get_ticks() - start_ms) / 1000
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
        )
        if font is not None:
            screen.blit(font.render(label, True, TEXT_COLOR), (10, 10))
        elif drawn != shown_in_title:
            pygame.display.set_caption(f"MCMC donut: {num_chains} chains    {label}")
            shown_in_title = drawn

        pygame.display.flip()

    pygame.quit()


def print_chain_stats(positions, burn_in=0):
    """Summarize samples of shape (num_steps, 2) or (num_steps, num_chains, 2).

    Every axis is counted from the end (x/y is always last), so the same code
    works with or without a chain axis. burn_in drops that many initial steps.
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


if __name__ == "__main__":
    # start = jnp.array([2.6, 0.0])  # on the ring, so the chain needs no burn-in
    start = jnp.array([0.0, 0.0])  # at the origin, so the chain needs burn-in
    burn_in = 1_000

    # first run the simple chain: positions has shape (num_steps, 2)
    states = run_chain(jr.key(0), donut_log_density, start)
    positions = states.position
    print(f"single chain, positions {positions.shape}, burn-in {burn_in}")
    print_chain_stats(positions, burn_in)

    # now run the vmap parallel version: positions has shape (num_steps, num_chains, 2)
    # default step_size=0.01 will take very small steps try different ones 0.2 is a good
    # one from a coverage point of view
    vstates = run_chain_vmap(jr.key(0), donut_log_density, start, step_size=0.04)
    vpositions = vstates.position
    print(
        f"\n{vpositions.shape[1]} chains (vmap), positions {vpositions.shape}, burn-in {burn_in}"
    )
    print_chain_stats(vpositions, burn_in)

    # optional viz: run with --viz to watch the vmap chains grow (30 s)
    # pygame_viz_donut(positions, 2.6, 0.33)
    if "--viz" in sys.argv:
        print()
        print("starting visualization")
        pygame_viz_chains(vpositions, 2.6, 0.033, duration=30.0)
