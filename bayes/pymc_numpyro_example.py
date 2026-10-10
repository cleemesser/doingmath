# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: doingmath (3.14.3)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Bayesian MCMC with PyMC, NumPyro, and BlackJAX
#
# This notebook fits one small model three ways, to show how the pieces of the
# JAX-based Bayesian stack fit together. It follows the steps in `workflow.qmd`:
# build a model, check the prior, fit, validate the computation, and use the
# result.
#
# The pieces, from the top down:
#
# | library   | job                                                            |
# |-----------|----------------------------------------------------------------|
# | PyMC      | You write the model. PyMC builds its log density as a graph.   |
# | PyTensor  | PyMC's graph library. It can compile the graph to C or to JAX. |
# | JAX       | Runs the compiled log density, with autodiff, `jit`, `vmap`.   |
# | NumPyro   | A JAX library with a fast NUTS sampler. PyMC can call it.      |
# | BlackJAX  | A JAX library of samplers, used as building blocks.            |
# | ArviZ     | Stores the draws and computes diagnostics and summaries.       |
#
# NUTS (the No-U-Turn Sampler) is a version of Hamiltonian Monte Carlo that
# picks its own trajectory length. It uses the gradient of the log density,
# which is why the log density must go through JAX (or PyTensor's C backend).
#
# The model comes from Colin's talk (`pymc_bayeux_example.py`). That file uses
# Bayeux, which wraps all of these samplers behind one interface. Bayeux 0.1.15
# cannot be imported with JAX 0.11: it always loads TensorFlow Probability and
# Oryx, and both use JAX internals that newer JAX releases removed. So this
# notebook does by hand what Bayeux automated.

# %% [markdown]
# ## Setup
#
# PyTensor's C compiler fails on this Mac. PyTensor 3.3.3 passes `-ld64` to clang
# on macOS 15 and later (a workaround for the Xcode 15 linker), and newer clang
# reads it as "link a library named d64":
# `ld: library 'd64' not found`. An empty `cxx` turns off the C compiler. The JAX
# samplers do not need it, because they compile the model through JAX.
# `.envrc` sets the same flag; it is set here too so that `uv run` works without
# direnv. It must be set before PyMC (and so PyTensor) is imported.

# %%
import os

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

import arviz as az
import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
from pymc.sampling.jax import get_jaxified_logp

# %% [markdown]
# ## 1. The model
#
# We observe 10 successes in 100 trials and want the success probability `p`.
# Instead of fixing a prior on `p`, the model puts a Beta(a, b) prior on it and
# lets the data inform `a` and `b` too, through weak HalfNormal(10) priors. A
# prior on the parameters of a prior is called a hyperprior.
#
# $$
# a, b \sim \text{HalfNormal}(10), \qquad
# p \mid a, b \sim \text{Beta}(a, b), \qquad
# y \mid p \sim \text{Binomial}(100, p), \quad y = 10
# $$
#
# Inside the `with` block, each `pm.*` call adds a random variable to the model.
# `observed=` turns a variable into the likelihood.

# %%
with pm.Model() as model:
    a = pm.HalfNormal("a", 10)
    b = pm.HalfNormal("b", 10)
    p = pm.Beta("p", a, b)
    observed = pm.Binomial("obs", p=p, n=100, observed=10)

# %% [markdown]
# ## 2. Prior predictive check
#
# Before fitting, simulate data from the prior alone and ask whether it is
# plausible. If the prior predicts only values that cannot happen, the model
# needs changing before any data is used. Here the question is what range of
# success counts (out of 100) the priors allow.

# %%
with model:
    prior = pm.sample_prior_predictive(draws=2_000, random_seed=0)

prior_counts = prior.prior_predictive["obs"].values.ravel()
prior_p = prior.prior["p"].values.ravel()
print(
    f"prior p: mean {prior_p.mean():.2f}, 5-95% {np.percentile(prior_p, [5, 95]).round(2)}"
)
print(
    f"prior predictive successes: 5-95% {np.percentile(prior_counts, [5, 95])}, "
    f"P(y <= 10) = {(prior_counts <= 10).mean():.2f}"
)

# %% [markdown]
# The prior on `p` is wide and centered near 0.5, so it allows almost any
# success count. The observed 10 of 100 is well inside that range. The prior
# does not rule out the data, and it is not so tight that it would dominate it.

# %% [markdown]
# ## 3. Fit with NumPyro through PyMC
#
# `nuts_sampler="numpyro"` makes PyMC compile the log density to JAX and hand it
# to NumPyro's NUTS. It takes the place of `pm_bx.mcmc.numpyro_nuts(...)` in the
# Bayeux version. The defaults are 4 chains, 1,000 warmup ("tune") steps, and
# 1,000 draws per chain. PyMC also jitters each chain's starting point.
#
# The result is an ArviZ `InferenceData` object: draws stored with named
# dimensions `(chain, draw)`, plus groups such as `sample_stats`.

# %%
with model:
    idata = pm.sample(nuts_sampler="numpyro", random_seed=0, progressbar=False)

print(az.summary(idata))

# %% [markdown]
# ### Reading the summary
#
# - `mean`, `sd`, and `eti89_*` describe the posterior: its mean, standard
#   deviation, and the 89% equal-tailed interval.
# - `r_hat` compares the chains with each other. Near 1.00 means they agree.
#   Values above about 1.01 mean the chains have not mixed, and the numbers above
#   it in that row cannot be trusted.
# - `ess_bulk` and `ess_tail` are effective sample sizes: how many independent
#   draws the correlated chain is worth, for the center and for the tails.
# - `mcse_*` is the Monte Carlo standard error, the error in `mean` and `sd` that
#   comes from using a finite number of draws.
#
# `p` comes out near 0.11, close to the raw rate 10/100, pulled slightly toward
# the prior.

# %% [markdown]
# ## 4. Fit with BlackJAX directly
#
# The simple route, `pm.sample(nuts_sampler="blackjax")`, fails with PyMC 6.3.2
# and BlackJAX 1.7.1:
# `TypeError: ... kernel() got an unexpected keyword argument 'progress_bar'`.
# PyMC passes `progress_bar` to `blackjax.window_adaptation` and removes it from
# its keyword arguments only afterward (`pymc/sampling/jax.py`, lines 253 and
# 277). BlackJAX forwards unknown keywords to the NUTS kernel, which rejects it.
#
# So here PyMC only compiles the log density, and BlackJAX does the rest. This
# shows the steps that `pm.sample` (and Bayeux) normally hide.
#
# ### 4a. The log density in unconstrained space
#
# NUTS moves freely over all real numbers, but `a` and `b` must be positive and
# `p` must be in (0, 1). PyMC therefore samples transformed variables that can
# take any real value: `a_log__ = log a`, `b_log__ = log b`, and
# `p_logodds__ = logit p`. The compiled log density is for those variables, and
# it includes the Jacobian term (the change-of-variables correction) for each
# transform.
#
# Watch the sign. Despite its name, `get_jaxified_logp(model)` with the default
# `negative_logp=True` returns +log p, and `negative_logp=False` returns −log p
# (PyMC 6.3.2). The check below compares it with PyMC's own `compile_logp`.
# Passing `False` because of the name makes the sampler climb toward
# low-probability regions.

# %%
logdensity_from_list = get_jaxified_logp(model)
value_names = [v.name for v in model.value_vars]
print("sampled variables:", value_names)


def logdensity(position):
    """position is a dict {value name: scalar}, the pytree BlackJAX moves around."""
    return logdensity_from_list([position[name] for name in value_names])


initial_point = model.initial_point()
reference = model.compile_logp()(initial_point)
ours = logdensity({name: jnp.asarray(v) for name, v in initial_point.items()})
print(
    f"log density at the initial point: ours {float(ours):.6f}, PyMC {float(reference):.6f}"
)
assert np.allclose(ours, reference), "log density sign or ordering is wrong"

# %% [markdown]
# ### 4b. Starting points with jitter
#
# Each chain starts at PyMC's default initial point plus uniform noise in
# (−1, 1) on every unconstrained variable. This copies PyMC's default
# `init="jitter+adapt_diag"`.
#
# Jitter matters for diagnostics. `r_hat` detects chains that disagree. If every
# chain starts at the same point, the chains can agree only because they have
# not yet moved far from that point, and `r_hat` reports 1.00 anyway. Chains
# that start apart and still agree are stronger evidence that each one found
# the whole posterior.

# %%
num_chains = 4
init_key, sample_key = jax.random.split(jax.random.key(0))

start = {name: jnp.asarray(v) for name, v in initial_point.items()}
jitter_keys = dict(zip(value_names, jax.random.split(init_key, len(value_names))))
initial_positions = {
    name: start[name]
    + jax.random.uniform(jitter_keys[name], (num_chains,), minval=-1.0, maxval=1.0)
    for name in value_names
}
print({name: np.round(np.asarray(v), 2) for name, v in initial_positions.items()})

# %% [markdown]
# ### 4c. Warmup and sampling for one chain
#
# BlackJAX gives the parts, and you put them together:
#
# 1. `blackjax.window_adaptation` runs the warmup. It tunes the NUTS step size
#    and a diagonal mass matrix (an estimate of each variable's scale), the same
#    job as PyMC's `tune=` steps. Warmup draws are thrown away.
# 2. `blackjax.nuts(logdensity, **parameters).step` is the tuned NUTS step.
# 3. `jax.lax.scan` applies that step `num_draws` times and collects positions.
#
# As in `mcmc_donut_jax.py`, the code is written for one chain and then batched
# with `jax.vmap`. Here `vmap` maps over both the keys and the starting
# positions, so every chain gets its own start and its own random numbers.

# %%


def run_one_chain(key, initial_position, num_warmup=1_000, num_draws=1_000):
    warmup_key, draw_key = jax.random.split(key)
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    (state, parameters), _ = warmup.run(
        warmup_key, initial_position, num_steps=num_warmup
    )
    step = blackjax.nuts(logdensity, **parameters).step

    def one_step(state, key):
        state, info = step(key, state)
        return state, (state.position, info.is_divergent)

    _, (positions, divergent) = jax.lax.scan(
        one_step, state, jax.random.split(draw_key, num_draws)
    )
    return positions, divergent  # arrays of shape (num_draws,)


chain_keys = jax.random.split(sample_key, num_chains)
unconstrained, divergent = jax.jit(jax.vmap(run_one_chain))(
    chain_keys, initial_positions
)
print(f"divergent transitions: {int(divergent.sum())}")

# %% [markdown]
# A divergent transition means the simulated trajectory became unstable, usually
# in a region of high curvature. Divergences mean the sampler may be missing part
# of the posterior. NumPyro and PyMC count them for you; here the count comes
# from the `info` that each NUTS step returns.
#
# ### 4d. Back to the model's parameters
#
# The draws are in unconstrained space. Undo each transform (`exp` for the log
# transforms, the sigmoid for log-odds) and give ArviZ arrays in its
# `(chain, draw)` layout. `pm.sample` does this step automatically for any model;
# here it is written out for this model's three variables.

# %%
idata_blackjax = az.from_dict(
    {
        "posterior": {
            "a": np.exp(unconstrained["a_log__"]),
            "b": np.exp(unconstrained["b_log__"]),
            "p": np.asarray(jax.nn.sigmoid(unconstrained["p_logodds__"])),
        }
    }
)
print(az.summary(idata_blackjax))

# %% [markdown]
# ## 5. Validate the computation
#
# Two independent checks that the samplers found the right posterior.
#
# ### 5a. Two samplers, one answer
#
# NumPyro and BlackJAX share only the compiled log density. If both give the
# same posterior to within Monte Carlo error, a bug in either sampler is
# unlikely. A wrong log density would fool both, which is why 4a checks it
# against PyMC separately.
#
# This check caught a real bug while this notebook was being put together. With
# `negative_logp=False`, BlackJAX reported `p` near 0.996, `r_hat` of 3.5, and
# effective sample sizes of 4. Nothing crashed, and each number looked plausible
# alone. The diagnostics and the disagreement with NumPyro exposed it.

# %%
for name in ["a", "b", "p"]:
    numpyro_mean = float(idata.posterior[name].mean())
    blackjax_mean = float(idata_blackjax.posterior[name].mean())
    print(f"{name}: NumPyro {numpyro_mean:.3f}   BlackJAX {blackjax_mean:.3f}")

# %% [markdown]
# ### 5b. An exact result to compare with
#
# The Beta prior is conjugate to the Binomial likelihood: given `a` and `b`,
# the posterior of `p` is again a Beta distribution, with the data added to the
# prior counts:
#
# $$
# p \mid a, b, y \sim \text{Beta}(a + y,\; b + n - y),
# \qquad
# E[p \mid a, b, y] = \frac{a + y}{a + b + n}
# $$
#
# Averaging that exact conditional mean over the sampled `(a, b)` gives a second
# estimate of the posterior mean of `p`. This is called Rao-Blackwellization,
# and it has less noise than averaging the sampled `p` directly. The two
# estimates must agree to within Monte Carlo error.

# %%
y, n = 10, 100
for label, data in [("NumPyro", idata), ("BlackJAX", idata_blackjax)]:
    a_draws = data.posterior["a"].values
    b_draws = data.posterior["b"].values
    p_direct = data.posterior["p"].values.mean()
    p_rao_blackwell = ((a_draws + y) / (a_draws + b_draws + n)).mean()
    print(
        f"{label}: E[p] from draws {p_direct:.4f}, from conjugacy {p_rao_blackwell:.4f}"
    )
    assert np.allclose(p_direct, p_rao_blackwell, atol=0.003)

# %% [markdown]
# ## What to take from this
#
# - **The model and the sampler are separate.** PyMC turns the model into one
#   log density function over unconstrained variables. Any JAX sampler can use
#   it. That is why one model ran under NumPyro and BlackJAX with the same
#   result.
# - **Wrappers that use internals break.** Bayeux broke because TensorFlow
#   Probability and Oryx used private parts of JAX. `pm.sample(...,
#   nuts_sampler="blackjax")` broke because two packages disagreed on a keyword.
#   The path that kept working used only public APIs: PyMC's compiled log
#   density and BlackJAX's documented sampler functions.
# - **Write one chain, then `vmap`.** The same pattern as the donut sampler runs
#   any number of chains, each with its own key and starting point.
# - **Validate the computation every time.** Check `r_hat`, effective sample
#   sizes, and divergences; start chains apart; and compare with a second
#   sampler or an exact result when one is available.

# %% [markdown]
#
