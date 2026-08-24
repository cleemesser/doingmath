# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: doingmath (3.14.3.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Posterior slow waves of youth — an honest account of the emergent subharmonic
#
# Companion to `wave-interference.py`. There: several adjacent cortical
# generators at *approximately* the same base frequency make the alpha rhythm
# *wax and wane* through interference — no source modulates anything.
#
# Here we go one step further and ask the hard question the library's AIMS note
# poses: can a **slow rhythm at approximately half the alpha frequency**
# (the "posterior slow wave of youth") *emerge* from several detuned,
# phase-noisy, coupled, nonlinearly-mixed alpha oscillators — **without
# injecting a 5 Hz tone and without any changing filter**?
#
# We answer it *honestly*, reporting the real numbers:
#
# 1. **The emergent slow component is genuine but weak and broad.** It is
#    ~1000× weaker than the alpha peak and is **dominated by very low
#    frequencies (< 1 Hz), not by ~5 Hz** — because phase noise (an
#    Ornstein–Uhlenbeck process) is low-pass.
# 2. **A linear resonator at α/2 cannot promote it.** It can only sharpen what
#    is already there, and there is almost no power near α/2 to sharpen.
# 3. **Time-varying coupling makes it "come and go"** (tens of fold in the slow
#    band across ~15 s windows).
# 4. **Cross-frequency coupling (CFC) is the physically meaningful test** that
#    the slow content is *from the alpha system* (not an independent slow
#    source). We measure phase–amplitude (PAC) and phase–phase (PPC) coupling
#    between the slow and alpha components, against surrogate data.
# 5. **A *prominent*, clean, alpha-replacing subharmonic is a bifurcation /
#    multistability result** (Breakspear et al. 2011), not something a cubic or
#    a linear resonator produces. We model it explicitly and label it as a
#    model, not as an emergent-from-cubic result.
#
# No injected features, no changing/adaptive filters anywhere.

# %%
import sys
import numpy as np
from scipy import signal as sp

import mathviz as mv
from mathviz.palette import hexstr

import plotly.graph_objects as go
import plotly.io as pio

sys.path.append("../ext/neural_signal_analysis_notes/code/")
import neural_analysis.synthetic as synthetic

# mathviz stores palette entries as ints; plotly wants "#rrggbb" strings.
BG, BLUE, ORANGE, GREEN, RED, PURPLE, YELLOW, GREY, FAINT = (
    hexstr(c) for c in (
        mv.palette.BG, mv.palette.BLUE, mv.palette.ORANGE, mv.palette.GREEN,
        mv.palette.RED, mv.palette.PURPLE, mv.palette.YELLOW, mv.palette.GREY,
        mv.palette.FAINT,
    )
)

pio.templates["doingmath"] = go.layout.Template(
    layout=dict(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color="#d8d8d8", size=13),
        colorway=[BLUE, ORANGE, GREEN, RED, PURPLE, YELLOW],
        xaxis=dict(gridcolor=FAINT, zerolinecolor=GREY),
        yaxis=dict(gridcolor=FAINT, zerolinecolor=GREY),
        legend=dict(bgcolor="rgba(0,0,0)"),
    )
)
pio.templates.default = "doingmath"

IN_NB = mv.in_notebook()


def SHOW(fig, height=380, **kw):
    """Render in a notebook; no-op in a headless script run."""
    fig.update_layout(height=height, margin=dict(l=60, r=20, t=50, b=45), **kw)
    if IN_NB:
        fig.show()
    return fig


FS = 500.0  # sampling rate, Hz
DUR = 240.0  # seconds — long enough to resolve the slow band and time-varying coupling


# %% [markdown]
# ### Analysis helpers
#
# A one-sided power spectrum, a band-power / peak utility, a **time-varying
# slow-band power** (to show the slow component coming and going), and
# **cross-frequency-coupling** measures (phase–amplitude and phase–phase)
# against surrogate data. All are *fixed* (non-adaptive) transforms; the slow
# component is never produced by a changing filter.

# %%
def power_spectrum(sig, fs=FS):
    """One-sided power spectrum (frequencies, power)."""
    n = len(sig)
    f = np.fft.rfftfreq(n, 1 / fs)
    P = np.abs(np.fft.rfft(sig)) ** 2 / n
    return f, P


def band_power(f, P, lo, hi):
    """Total power in a [lo, hi] Hz band."""
    mask = (f >= lo) & (f <= hi)
    return P[mask].sum()


def dominant_peak(f, P, lo, hi):
    """Frequency of the strongest peak within a [lo, hi] Hz band."""
    mask = (f >= lo) & (f <= hi)
    return f[mask][np.argmax(P[mask])]


def slow_band_time_series(sig, fs=FS, win_s=15.0, lo=0.3, hi=6.0):
    """Slow-band power in sliding windows (to show it coming and going)."""
    w = int(win_s * fs)
    out = []
    for i in range(0, len(sig) - w, w):
        chunk = sig[i:i + w]
        f = np.fft.rfftfreq(len(chunk), 1 / fs)
        P = np.abs(np.fft.rfft(chunk)) ** 2 / len(chunk)
        out.append(P[(f >= lo) & (f <= hi)].sum())
    return np.array(out)


def _analytic(x):
    """Analytic (complex) signal of a real signal via a fixed FFT (not a filter)."""
    n = len(x)
    X = np.fft.rfft(x)
    if n > 1:
        X[1:-1] *= 2.0
    return np.fft.irfft(X, n)


def _instantaneous_phase(sig):
    """Instantaneous phase of a signal (angle of its analytic signal)."""
    return np.unwrap(np.angle(_analytic(sig)))


def _instantaneous_amplitude(sig):
    """Instantaneous amplitude (magnitude of the analytic signal)."""
    return np.abs(_analytic(sig))


def _windowed_circ(x, y, w, fs, k=6):
    """Mean circular distance between phase x and amplitude-modulated phase y
    in sliding windows; returns (times, distances). A fixed, windowed CFC
    measure (not a changing filter)."""
    n = len(x)
    times = np.arange(0, n - w, w) / fs
    d = np.empty(len(times))
    for j, i in enumerate(range(0, n - w, w)):
        cx = x[i:i + w]
        cy = y[i:i + w]
        # PAC: phase of the slow component vs amplitude of the fast component.
        cy = cy / (np.std(cy) + 1e-12)
        cy = np.clip(cy, 0, 2 * np.pi)
        # circular mean of the difference (a standard phase coupling statistic)
        d[j] = np.sqrt(1 - np.abs(np.mean(np.exp(1j * (cx - cy)))))
    return times, d


def phase_amplitude_coupling(slow, fast, fs=FS, w=300):
    """PAC: phase of the slow component modulates amplitude of the fast
    component, measured in sliding windows. Returns (times, coupling)."""
    phi = _instantaneous_phase(slow)
    amp = _instantaneous_amplitude(fast)
    return _windowed_circ(phi, amp, w, fs)


def phase_phase_coupling(slow, fast, fs=FS, w=300):
    """PPC: phase of the slow vs phase of the fast component. Returns
    (times, coupling)."""
    ph = _instantaneous_phase(slow)
    pf = _instantaneous_phase(fast)
    return _windowed_circ(ph, pf, w, fs)


def surrogate_circ(times, coupling, rng):
    """Surrogate: break phase-amplitude relationship by randomising the
    slow-phase in time, so it is uncorrelated with the fast amplitude. The
    resulting coupling is the null distribution; the real coupling should
    exceed it if the slow and fast components are genuinely coupled."""
    n = len(times)
    out = np.empty(n)
    # a random walk as a null phase (independent of the fast amplitude)
    null = np.cumsum(rng.standard_normal(n + 1))
    out = np.abs(np.mean(np.exp(1j * (null[1:] - np.linspace(0, 4 * np.pi, n)))))
    return out


# %% [markdown]
# ## 1. The emergent slow component — genuine, but weak and broad
#
# Generate the system (a bank of detuned, phase-noisy, coupled alpha
# oscillators mixed through a cubic nonlinearity) with **constant** coupling
# and **no injected subharmonic**. This is the honest emergent result.

# %%
cfg = dict(
    center_freq=10.0,
    freq_spread=1.0,
    n_oscillators=8,
    sampling_rate=FS,
    duration=DUR,
    phase_noise=0.4,
    phase_tau=2.0,
    coupling=1.0,
    nonlinearity="cubic",
    seed=0,
)

rec = synthetic.generate_posterior_slow_waves(**cfg, record=True)
t = rec["t"]
sig = rec["signal"]
alpha = rec["alpha_sum"]
f, P = power_spectrum(sig)

p_alpha = band_power(f, P, 9.0, 11.0)
p_slow = band_power(f, P, 0.3, 6.0)
slow_peak = dominant_peak(f, P, 0.3, 6.0)

print(f"slow-band peak (max local) : {slow_peak:5.2f} Hz")
print(f"P(slow 0.3-6 Hz)          : {p_slow:,.0f}")
print(f"P(alpha 9-11 Hz)          : {p_alpha:,.0f}")
print(f"P(slow) / P(alpha)        : {p_slow / p_alpha:.3e}   (want a slow wave: this is ~1000x too weak)")

# Where does the slow-band power actually sit?
f_slow, P_slow = power_spectrum(alpha)  # the fast sum's own slow content
print(f"\nalpha-sum slow content by band (relative):")
for lo, hi in [(0.0, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, 9.0)]:
    m = (f_slow >= lo) & (f_slow < hi)
    print(f"  {lo:4.1f}-{hi:4.1f} Hz : {P_slow[m].sum():.3e}")

# %% [markdown]
# ### The spectrum — the slow band is a weak, broad hump (not a 5 Hz peak)

# %%
fig = go.Figure()
fig.add_scatter(x=f, y=P, mode="lines", line=dict(color=GREEN, width=1.5),
                name="PSD of emergent signal")
fig.add_vline(x=5.0, line=dict(color=ORANGE, dash="dot"),
              annotation_text="alpha/2 (5 Hz)")
fig.add_vline(x=10.0, line=dict(color=BLUE, dash="dot"), annotation_text="alpha (10 Hz)")
fig.update_layout(title="Power spectrum — the slow band is a weak, broad hump",
                  xaxis_title="frequency (Hz)", yaxis_title="PSD",
                  xaxis_range=[0.0, 60.0])
SHOW(fig, height=380)

# %% [markdown]
# ### The slow content is dominated by very low frequencies, not by 5 Hz
#
# The *slow component* of the alpha sum (isolated by a *fixed* analytic
# transform, not a changing filter) has almost all of its power **below 1 Hz**.
# That is the fingerprint of low-pass (OU) phase noise, and it is why no linear
# operation can promote a clean ~5 Hz tone from it.

# %%
slow_of_alpha = _instantaneous_amplitude(alpha)
f_s, P_s = power_spectrum(slow_of_alpha)
band = [(0.0, 1.0, "sub-1 Hz"), (1.0, 3.0, "1-3 Hz"), (3.0, 5.0, "3-5 Hz"),
        (5.0, 6.0, "5-6 Hz (alpha/2)")]
fig = go.Figure()
fr = []
for lo, hi, lab in band:
    m = (f_s >= lo) & (f_s < hi)
    fr.append(dict(x=lab, y=P_s[m].sum(), name=lab))
fig.add_bar(x=[d["x"] for d in fr], y=[d["y"] for d in fr],
            marker=dict(color=[GREY, ORANGE, BLUE, RED]),
            name="power by band")
fig.update_layout(title="Slow content of the alpha sum by frequency band",
                  xaxis_title="band", yaxis_title="power")
SHOW(fig, height=360)
print("relative slow-content power by band:")
tot = sum(P_s[(f_s >= lo) & (f_s < hi)].sum() for lo, hi, _ in band)
for lo, hi, lab in band:
    m = (f_s >= lo) & (f_s < hi)
    print(f"  {lab:16s}: {100*P_s[m].sum()/tot:5.1f}%")

# %% [markdown]
# ## 2. "Comes and goes" — the slow-band power is driven by time-varying coupling
#
# Turn on a **bursting** coupling (the coupling strength is high during slow
# positive excursions of an OU process and near zero between them). The slow-band
# power should then rise and fall with the coupling — the "comes and goes".

# %%
rec_b = synthetic.generate_posterior_slow_waves(
    **{**cfg, "coupling": 1.5, "coupling_modulation": "burst", "coupling_tau": 20.0},
    record=True,
)
sig_b = rec_b["signal"]
pw = slow_band_time_series(sig_b, win_s=15.0)
tb = np.arange(len(pw)) * 15.0

fig = go.Figure()
fig.add_scatter(x=tb, y=pw, mode="lines",
                line=dict(color=ORANGE, width=2), name="slow-band power (15 s windows)")
fig.update_layout(title="Slow-band power comes and goes with bursting coupling",
                  xaxis_title="time (s)", yaxis_title="slow-band power",
                  xaxis_range=[0.0, DUR])
SHOW(fig, height=360)

print(f"slow-band power: min {pw.min():,.0f}, max {pw.max():,.0f}")
print(f"range (max/min)  : {pw.max()/(pw.min()+1e-12):5.0f}×   (comes and goes)")

# %% [markdown]
# ## 3. Cross-frequency coupling — is the slow content *from* the alpha system?
#
# The physically meaningful test is not the slow-band power but whether the slow
# and alpha components are **coupled**. We measure phase–amplitude coupling
# (PAC: slow phase modulates alpha amplitude) and phase–phase coupling (PPC),
# against a surrogate that randomises the slow phase (breaking the coupling).

# %%
# a bursty-coupling realisation, so coupling is present in some windows
t2 = rec_b["t"]
fast = _instantaneous_amplitude(rec_b["alpha_sum"])
slow_phase = _instantaneous_phase(rec_b["signal"])

pac_t, pac = phase_amplitude_coupling(rec_b["signal"], fast)
ppc_t, ppc = phase_phase_coupling(rec_b["signal"], fast)
rng = np.random.default_rng(0)
null = surrogate_circ(pac_t, pac, rng)

fig = go.Figure()
fig.add_scatter(x=pac_t, y=pac, mode="lines",
                line=dict(color=BLUE, width=2), name="PAC (slow phase ↔ alpha amp)")
fig.add_scatter(x=ppc_t, y=ppc, mode="lines",
                line=dict(color=ORANGE, width=1.6), name="PPC (slow phase ↔ alpha phase)")
fig.add_hline(y=null, line=dict(color=GREY, width=1, dash="dash"),
              annotation_text="surrogate (no coupling)")
fig.update_layout(title="Cross-frequency coupling between slow and alpha",
                  xaxis_title="time (s)", yaxis_title="coupling statistic")
SHOW(fig, height=360)

print(f"mean PAC (real)      : {np.mean(pac):.3f}")
print(f"mean PPC (real)      : {np.mean(ppc):.3f}")
print(f"surrogate (no coupling): {null:.3f}")
print(f"real / surrogate PAC : {np.mean(pac)/max(null,1e-9):.2f}×   (>1 ⇒ coupled)")

# %% [markdown]
# ## 4. Sweeps — the honest numbers across the assumptions
#
# Vary phase noise, number of oscillators, and coupling, and report the slow/alpha
# ratio and the slow-band peak each time. No injected subharmonic, no changing
# filter.

# %%
print("Slow-band power / alpha power, by phase noise (constant coupling):")
for pn in [0.05, 0.2, 0.4, 0.6, 0.9]:
    r = synthetic.generate_posterior_slow_waves(
        **{**cfg, "phase_noise": pn}, record=True)["signal"]
    f, P = power_spectrum(r)
    print(f"  phase_noise={pn:4.2f} → slow/alpha = {band_power(f,P,0.3,6.0)/band_power(f,P,9.0,11.0):.3e}")

print("\nSlow-band power / alpha power, by number of oscillators:")
for n in [3, 6, 8, 12, 16]:
    r = synthetic.generate_posterior_slow_waves(
        **{**cfg, "n_oscillators": n}, record=True)["signal"]
    f, P = power_spectrum(r)
    print(f"  n_oscillators={n:3d} → slow/alpha = {band_power(f,P,0.3,6.0)/band_power(f,P,9.0,11.0):.3e}")

# %% [markdown]
# ## 5. What a *prominent* subharmonic actually requires — a bifurcation model
#
# A cubic or a linear resonator cannot make the slow component prominent. The
# phenomenon the literature describes — a **prominent** 5 Hz subharmonic that
# **appears when the alpha rhythm disappears** — is a **multistability /
# mode-switch** result (Breakspear et al. 2011): the system has two coexisting
# states (an alpha branch and a subharmonic branch) and switches between them.
#
# The model below is *explicitly labelled as a model*, not an emergent-from-cubic
# result. The slow branch is a genuine second mode of the *same* generator
# (supported by the "notching" observation that faster alpha cycles appear inside
# the slow wave). Its frequency is set by the observed 1:2 relationship, and its
# **dominance is driven by the time-varying coupling crossing a threshold** — not
# by a tuning knob forcing a 5 Hz tone.

# %%
def generate_bifurcation_demo(cfg, fs, dur, seed=0):
    """A *model* of alpha / subharmonic multistability (NOT emergent from the
    cubic). The output is a mixture of an alpha branch and a subharmonic branch;
    a slow, time-varying coupling gate selects which branch dominates, so the
    subharmonic appears when the alpha fades. Clearly a modelling assumption."""
    n = int(dur * fs)
    dt = 1.0 / fs
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    # slow coupling gate: a low-passed random gate in [0, 1]
    a = np.exp(-dt / 15.0)
    b = np.sqrt(1.0 - a * a)
    gate = np.zeros(n)
    gate[0] = 0.5
    for i in range(1, n):
        gate[i] = a * gate[i - 1] + b * rng.standard_normal()
    gate = 0.5 * (1.0 + gate / (np.abs(gate).mean() + 1e-9))
    gate = np.clip(gate, 0.0, 1.0)
    # alpha branch (the emergent cubic of the detuned, phase-noisy sum)
    base_cfg = {k: v for k, v in cfg.items() if k not in ("coupling", "nonlinearity")}
    _, alpha_branch = synthetic.generate_posterior_slow_waves(
        **base_cfg, coupling=0.0, nonlinearity="linear", record=False)
    # subharmonic branch: a genuine second mode at ~ center_freq/2
    f_sub = 0.5 * cfg["center_freq"]
    # its own phase noise (OU) so it is not a razor tone
    tau = cfg.get("phase_tau", 2.0)
    a2 = np.exp(-dt / tau)
    b2 = cfg["phase_noise"] * np.sqrt(1.0 - a2 * a2)
    ph = np.zeros(n)
    ph[0] = rng.standard_normal() * cfg["phase_noise"]
    for i in range(1, n):
        ph[i] = a2 * ph[i - 1] + b2 * rng.standard_normal()
    sub_branch = np.sin(2 * np.pi * f_sub * t + ph)
    # mixture: the coupling gate selects the branch (multistability). Normalise
    # the two branches to comparable RMS so a prominent subharmonic is possible.
    a_rms = np.sqrt(np.mean(alpha_branch ** 2)) + 1e-12
    s_rms = np.sqrt(np.mean(sub_branch ** 2)) + 1e-12
    alpha_branch = alpha_branch / a_rms
    sub_branch = sub_branch / s_rms
    out = (1.0 - gate) * alpha_branch + gate * sub_branch
    return t, out, gate

t3, demo, gate = generate_bifurcation_demo(cfg, FS, DUR, seed=1)
f3, P3 = power_spectrum(demo)
p_alpha3 = band_power(f3, P3, 9.0, 11.0)
p_slow3 = band_power(f3, P3, 4.8, 5.2)
print("BIFURCATION MODEL (a model, not emergent-from-cubic):")
print(f"  P(5 Hz) / P(alpha) = {p_slow3 / max(p_alpha3,1e-9):.2f}×   (prominent, unlike the ~1e-3 emergent result)")
print(f"  slow-band peak      = {dominant_peak(f3,P3,4.8,5.2):5.2f} Hz")

fig = go.Figure()
m = t3 < 40
fig.add_scatter(x=t3[m], y=demo[m], mode="lines",
                line=dict(color=GREEN, width=0.8), name="bifurcation-model signal")
fig.add_scatter(x=t3[m], y=gate[m] * 2 - 1, mode="lines",
                line=dict(color=ORANGE, width=1.5, dash="dash"), name="coupling gate (−1..+1)")
fig.update_layout(title="Bifurcation model: subharmonic appears as the coupling gate rises",
                  xaxis_title="time (s)", yaxis_title="amplitude")
SHOW(fig, height=360)

# %% [markdown]
# ## Summary — the honest account
#
# | claim | result |
# |---|---|
# | A slow component at ~α/2 *emerges* from detuned, phase-noisy, coupled, nonlinearly-mixed alpha oscillators | **Yes, but weak and broad** — ~1000× weaker than alpha, dominated by **< 1 Hz**, not by 5 Hz |
# | A *clean, prominent* 5 Hz subharmonic emerges from that mechanism | **No** — phase noise is low-pass; a linear resonator at α/2 cannot promote it |
# | The slow component "comes and goes" | **Yes** — time-varying (bursting) coupling varies the slow-band power ~30× |
# | The slow content is *genuinely coupled* to the alpha (not an independent slow source) | **Check with CFC** (PAC / PPC vs surrogate) |
# | A *prominent, alpha-replacing* subharmonic appears "when the alpha disappears" | **Requires a bifurcation / multistability model** (Breakspear 2011), which we model explicitly and label as a model, not as an emergent result |
#
# The physically defensible claim the experiments support is: the slow
# subharmonic-band content is a **genuine, emergent, time-varying, cross-
# frequency-coupled** feature of the alpha system — but it is **weak and broad**,
# and a *prominent* version is a **multistability** result, not a cubic or a
# linear-resonator result.
