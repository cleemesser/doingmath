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
# # Wave interference — and why alpha rhythms wax and wane
#
# We demonstrate constructive and destructive interference, then apply it to EEG.
#
# In the brain we can model an EEG signal as a fundamental frequency plus white and $1/f$ noise.
# We start with one channel. But when several adjacent regions oscillate at *approximately* the
# same base frequency — say 10.0 Hz, 10.05 Hz, 10.1 Hz — their sum does something the individual
# generators never do: it **waxes and wanes**, even though no single generator changes amplitude.
#
# That is the punchline of this notebook. Apparent amplitude modulation of the alpha rhythm does
# not require any source to modulate its amplitude. Slight detuning is enough.

# %%
import sys
import numpy as np
import sympy as sp
import scipy as sci
from scipy import signal

sp.init_printing()

import mathviz as mv
from mathviz import palette as pal
from mathviz.palette import hexstr

sys.path.append("../ext/neural_signal_analysis_notes/code/")
import neural_analysis.synthetic

# mathviz stores palette entries as ints (0x4FC3F7) because vedo/k3d want them that way;
# plotly wants "#4fc3f7" strings, so run them through the shipped converter.
BG, BLUE, ORANGE, GREEN, RED, PURPLE, YELLOW, GREY, FAINT = (
    hexstr(c) for c in (pal.BG, pal.BLUE, pal.ORANGE, pal.GREEN,
                        pal.RED, pal.PURPLE, pal.YELLOW, pal.GREY, pal.FAINT)
)

import plotly.graph_objects as go
import plotly.io as pio

# %% [markdown]
# ### Plot setup
#
# A dark plotly template built from the `mathviz` palette, so these figures sit next to the
# matplotlib figures in the rest of the repo without a style clash.
#
# `SHOW` guards `fig.show()`: inside Jupyter it renders, but under
# `uv run python wave-interference.py` it stays silent instead of spawning a browser tab.

# %%
pio.templates["doingmath"] = go.layout.Template(
    layout=dict(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color="#d8d8d8", size=13),
        colorway=[BLUE, ORANGE, GREEN, RED, PURPLE, YELLOW],
        xaxis=dict(gridcolor=FAINT, zerolinecolor=GREY),
        yaxis=dict(gridcolor=FAINT, zerolinecolor=GREY),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
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


FS = 250.0  # EEG sampling rate, Hz

# %% [markdown]
# ## 1. Same frequency, different phase — the phasor picture
#
# Two waves at the *same* frequency $f$ but different amplitude and phase,
#
# $$x_1(t) = A_1\cos(2\pi f t + \varphi_1), \qquad x_2(t) = A_2\cos(2\pi f t + \varphi_2)$$
#
# always sum to a wave at that same frequency. The sum's amplitude and phase come from adding
# two **phasors** — complex numbers $A_k e^{i\varphi_k}$ — because
# $A\cos(\omega t + \varphi) = \operatorname{Re}\!\left[A e^{i\varphi} e^{i\omega t}\right]$
# and $\operatorname{Re}$ is linear. So the whole problem collapses to one complex addition:
#
# $$A e^{i\varphi} = A_1 e^{i\varphi_1} + A_2 e^{i\varphi_2}$$
#
# **Constructive** interference is $\varphi_2-\varphi_1 = 0$ (phasors parallel, $A = A_1 + A_2$);
# **destructive** is $\varphi_2-\varphi_1 = \pi$ (antiparallel, $A = \lvert A_1 - A_2 \rvert$).
# Everything between interpolates by the law of cosines.

# %%
def phasor_sum(amps, phases):
    """Amplitude and phase of the sum of same-frequency sinusoids. Returns (A, phi)."""
    z = np.sum(np.asarray(amps) * np.exp(1j * np.asarray(phases)))
    return np.abs(z), np.angle(z)


f0 = 10.0
t = np.arange(0, 1.0, 1 / FS)

A1, A2 = 1.0, 0.7
for name, dphi in [("constructive", 0.0), ("quadrature", np.pi / 2), ("destructive", np.pi)]:
    x1 = A1 * np.cos(2 * np.pi * f0 * t + 0.0)
    x2 = A2 * np.cos(2 * np.pi * f0 * t + dphi)
    A, phi = phasor_sum([A1, A2], [0.0, dphi])
    predicted = A * np.cos(2 * np.pi * f0 * t + phi)
    # law of cosines: A^2 = A1^2 + A2^2 + 2 A1 A2 cos(dphi)
    law = np.sqrt(A1**2 + A2**2 + 2 * A1 * A2 * np.cos(dphi))
    assert np.allclose(x1 + x2, predicted), name
    assert np.allclose(A, law), name
    print(f"{name:>13s}  Δφ={dphi:5.2f} rad   A = {A:.4f}  (law of cosines {law:.4f})  ✓")

# %% [markdown]
# Both assertions hold for every phase offset: the phasor sum *is* the waveform sum, and its
# magnitude *is* the law of cosines. Now sweep the phase offset continuously and watch the
# amplitude trace out that cosine law.

# %%
dphis = np.linspace(0, 2 * np.pi, 400)
amps = np.array([phasor_sum([A1, A2], [0.0, d])[0] for d in dphis])

fig = go.Figure()
fig.add_scatter(x=dphis, y=amps, mode="lines", line=dict(color=BLUE, width=2.5), name="|A₁+A₂e^{iΔφ}|")
fig.add_hline(y=A1 + A2, line=dict(color=GREEN, dash="dot"), annotation_text="A₁+A₂ (constructive)")
fig.add_hline(y=abs(A1 - A2), line=dict(color=RED, dash="dot"), annotation_text="|A₁−A₂| (destructive)")
fig.update_layout(
    title="Resultant amplitude vs phase offset (same frequency)",
    xaxis_title="Δφ (rad)", yaxis_title="resultant amplitude",
)
SHOW(fig)

# %% [markdown]
# ## 2. Slightly different frequencies — beats
#
# Once $f_1 \neq f_2$ the phase difference is no longer constant: it *drifts* at rate
# $2\pi(f_2-f_1)$. So the pair sweeps continuously through the constructive/destructive cycle of
# §1. Let `sympy` do the product-to-sum identity rather than trusting memory:

# %%
t_s, f1_s, f2_s = sp.symbols("t f_1 f_2", real=True)
expr = sp.cos(2 * sp.pi * f1_s * t_s) + sp.cos(2 * sp.pi * f2_s * t_s)
beat_form = sp.simplify(sp.fu(expr))
beat_form

# %% [markdown]
# which is
#
# $$\cos(2\pi f_1 t) + \cos(2\pi f_2 t)
#   = \underbrace{2\cos\!\left(2\pi \tfrac{f_1-f_2}{2} t\right)}_{\text{envelope}}
#     \;\underbrace{\cos\!\left(2\pi \tfrac{f_1+f_2}{2} t\right)}_{\text{carrier}}$$
#
# A fast **carrier** at the mean frequency, multiplied by a slow **envelope** at half the
# difference frequency. Because the ear/eye perceives $\lvert \text{envelope} \rvert$, the
# *audible* beat rate is $\lvert f_1 - f_2 \rvert$ — twice the envelope frequency, since
# $\lvert\cos\rvert$ has half the period of $\cos$.

# %%
target = 2 * sp.cos(sp.pi * (f1_s - f2_s) * t_s) * sp.cos(sp.pi * (f1_s + f2_s) * t_s)
assert sp.simplify(sp.expand_trig(expr - target)) == 0
print("product-to-sum identity verified symbolically ✓")

# %%
dur = 20.0
t = np.arange(0, dur, 1 / FS)
fa, fb = 10.0, 10.5  # 0.5 Hz apart -> beat period 2 s
x = np.cos(2 * np.pi * fa * t) + np.cos(2 * np.pi * fb * t)
env = 2 * np.abs(np.cos(np.pi * (fa - fb) * t))

# numeric check against the closed form
carrier = 2 * np.cos(np.pi * (fa - fb) * t) * np.cos(np.pi * (fa + fb) * t)
assert np.allclose(x, carrier)

m = t < 6
fig = go.Figure()
fig.add_scatter(x=t[m], y=x[m], mode="lines", line=dict(color=BLUE, width=1.2), name="sum")
fig.add_scatter(x=t[m], y=env[m], mode="lines", line=dict(color=ORANGE, width=2, dash="dash"), name="envelope")
fig.add_scatter(x=t[m], y=-env[m], mode="lines", line=dict(color=ORANGE, width=2, dash="dash"), showlegend=False)
fig.update_layout(
    title=f"Beats: {fa} Hz + {fb} Hz → envelope repeats every {1/abs(fa-fb):.1f} s",
    xaxis_title="time (s)", yaxis_title="amplitude",
)
SHOW(fig)

# %% [markdown]
# ## 3. Three adjacent regions: 10.0, 10.05, 10.1 Hz
#
# Now the case from the opening. Three cortical patches, equally detuned by
# $\delta = 0.05$ Hz about $f_0 = 10.05$ Hz. Write $\Delta = 2\pi\delta$; for equal amplitudes
# the outer pair beats against itself and the centre rides along:
#
# $$\cos((\omega_0-\Delta)t) + \cos(\omega_0 t) + \cos((\omega_0+\Delta)t)
#   = \cos(\omega_0 t)\,\bigl[\,1 + 2\cos(\Delta t)\,\bigr]$$
#
# The bracket swings between $+3$ and $-1$, so the envelope $\lvert 1 + 2\cos\Delta t\rvert$ runs
# from **3 down to 0** — full constructive interference, then complete cancellation — with period
# $1/\delta = 20$ s. Three perfectly steady generators produce twenty-second waxing and waning.

# %%
delta = 0.05
f_c = 10.05
freqs = np.array([f_c - delta, f_c, f_c + delta])

dur = 60.0
t = np.arange(0, dur, 1 / FS)
comps = np.cos(2 * np.pi * freqs[:, None] * t[None, :])
x3 = comps.sum(axis=0)

closed = np.cos(2 * np.pi * f_c * t) * (1 + 2 * np.cos(2 * np.pi * delta * t))
assert np.allclose(x3, closed)
print(f"closed form verified ✓   envelope period = {1/delta:.0f} s, peak amplitude = {x3.max():.3f}")

# %% [markdown]
# Each generator is a flat, unwavering 10-ish Hz sinusoid. Their sum is not.

# %%
fig = go.Figure()
for f, c, col in zip(freqs, comps, [GREY, FAINT, GREY]):
    fig.add_scatter(x=t, y=c, mode="lines", line=dict(color=col, width=0.7),
                    name=f"{f:.2f} Hz", opacity=0.5)
fig.add_scatter(x=t, y=x3, mode="lines", line=dict(color=BLUE, width=1.3), name="sum of 3")
fig.add_scatter(x=t, y=np.abs(1 + 2 * np.cos(2 * np.pi * delta * t)), mode="lines",
                line=dict(color=ORANGE, width=2.5, dash="dash"), name="envelope |1+2cos Δt|")
fig.update_layout(
    title="Three steady generators 0.05 Hz apart — the sum waxes and wanes over 20 s",
    xaxis_title="time (s)", yaxis_title="amplitude",
)
SHOW(fig, height=420)

# %% [markdown]
# ### The resolution trap
#
# Could you just look at the spectrum and see three peaks? Only with enough data. Frequency
# resolution of a Welch estimate is $\Delta f \approx f_s/N_{\text{perseg}}$, so telling 10.00 from
# 10.05 Hz needs a window of at least $1/0.05 = 20$ s — and to *separate* them comfortably,
# several times that.
#
# So there is a regime, familiar from real recordings, where the beating is obvious in the time
# domain while the spectrum still shows a single alpha bump. This is the time–frequency
# uncertainty principle wearing a clinical hat: **the envelope modulation and the peak splitting
# carry the same information**, and short epochs let you see only the first.

# %%
fig = go.Figure()
for win_s, col in [(4.0, RED), (20.0, YELLOW), (60.0, GREEN)]:
    nps = int(win_s * FS)
    if nps > len(x3):
        continue
    fr, pxx = signal.welch(x3, fs=FS, nperseg=nps)
    band = (fr > 9.7) & (fr < 10.4)
    fig.add_scatter(x=fr[band], y=pxx[band], mode="lines+markers",
                    line=dict(color=col, width=2), marker=dict(size=4),
                    name=f"{win_s:.0f} s window (Δf={FS/nps:.3f} Hz)")
for f in freqs:
    fig.add_vline(x=f, line=dict(color=FAINT, dash="dot", width=1))
fig.update_layout(
    title="Same signal, three window lengths — the peaks only split when the window is long enough",
    xaxis_title="frequency (Hz)", yaxis_title="PSD",
)
SHOW(fig)

# %% [markdown]
# ## 4. Making it look like EEG — white and $1/f$ noise
#
# Real cortical recordings sit on a broadband **aperiodic** background whose power falls off as
# $P(f) \propto 1/f^{\beta}$, with $\beta \approx 1$–2 in scalp EEG, plus a roughly flat white
# component from instrumentation and muscle. The alpha rhythm is the narrow peak riding on top.

# %%
rng = np.random.default_rng(20260823)


def white_noise(n, sigma=1.0, rng=rng):
    """Flat-spectrum Gaussian noise, β = 0."""
    return sigma * rng.standard_normal(n)


def one_over_f_noise(n, beta=1.0, fs=FS, rng=rng):
    """Aperiodic background with PSD ∝ 1/f**beta, unit standard deviation.

    FFT spectral shaping: draw white noise (flat expected spectrum), scale each
    positive-frequency bin, and transform back.

    The exponent is **beta/2**, not beta: the PSD is |X(f)|**2, so shaping the
    *amplitude* spectrum by f**(-beta/2) delivers power ∝ f**(-beta). Using
    f**(-beta) here is the classic factor-of-two error that leaves every fitted
    aperiodic slope twice as steep as intended.

    The DC bin is zeroed rather than scaled -- f**(-beta) diverges at f=0, and a
    constant offset is not part of the noise. Normalisation is done in the time
    domain at the end, so "unit sigma" means unit sample standard deviation.
    """
    f = np.fft.rfftfreq(n, d=1 / fs)
    X = np.fft.rfft(rng.standard_normal(n))
    scale = np.zeros_like(f)
    scale[1:] = f[1:] ** (-beta / 2)
    x = np.fft.irfft(X * scale, n=n)
    return x / x.std()


# %% [markdown]
# Verify it: fit a straight line to the Welch PSD on log-log axes and check the slope comes back
# as $-\beta$. This is the assertion that would have caught an exponent of $\beta$ instead of
# $\beta/2$ — the fitted slope would have been twice as steep.

# %%
def fitted_slope(x, fs=FS, band=(1.0, 50.0)):
    """Least-squares slope of log10(PSD) vs log10(f) over `band` -- i.e. -beta."""
    fr, pxx = signal.welch(x, fs=fs, nperseg=int(20 * fs))
    m = (fr >= band[0]) & (fr <= band[1])
    return np.polyfit(np.log10(fr[m]), np.log10(pxx[m]), 1)[0]


for beta in (0.0, 1.0, 1.4, 2.0):
    est = -fitted_slope(one_over_f_noise(int(120 * FS), beta=beta))
    assert abs(est - beta) < 0.08, (beta, est)
    print(f"requested β = {beta:4.1f}   recovered from PSD fit = {est:5.2f}  ✓")


# %%
dur = 60.0
t = np.arange(0, dur, 1 / FS)
n = len(t)

alpha = 12.0 * np.cos(2 * np.pi * freqs[:, None] * t[None, :]).sum(axis=0)  # µV-ish
eeg = alpha + 8.0 * one_over_f_noise(n, beta=1.4) + 2.0 * white_noise(n)

# %% [markdown]
# ### The raw channel, on its own
#
# First, what the recording actually looks like — one trace, nothing overlaid, no envelope, no
# decomposition. This is all a reader of the EEG gets to see. The full 60 s is shown so the slow
# waxing and waning survives contact with the noise and is visible unaided.

# %%
fig = go.Figure()
fig.add_scatter(x=t, y=eeg, mode="lines", line=dict(color=BLUE, width=0.8), showlegend=False)
fig.update_layout(
    title="Simulated EEG channel — raw",
    xaxis_title="time (s)", yaxis_title="µV",
)
SHOW(fig, height=300)

# %% [markdown]
# Now a single **10 s page** — the standard review epoch in clinical EEG, inherited from 30 mm/s
# paper speed. At this scale individual alpha cycles resolve instead of smearing into a solid band,
# and you would expect to count roughly 100 of them across the page at 10 Hz:

# %%
PAGE_S = 10.0  # clinical EEG review convention: 10 s/page (30 mm/s paper speed)
m = t < PAGE_S
fig = go.Figure()
fig.add_scatter(x=t[m], y=eeg[m], mode="lines", line=dict(color=BLUE, width=1.1), showlegend=False)
fig.update_layout(
    title=f"Simulated EEG channel — raw, {PAGE_S:.0f} s page",
    xaxis_title="time (s)", yaxis_title="µV",
)
SHOW(fig, height=300)

# %% [markdown]
# ### And now with the generators put back in
#
# The same page, with the noiseless alpha sum drawn over it. Everything that looks like amplitude
# modulation in the trace above is the §3 interference pattern — no source changed strength.

# %%
fig = go.Figure()
fig.add_scatter(x=t[m], y=eeg[m], mode="lines", line=dict(color=BLUE, width=1), name="raw channel")
fig.add_scatter(x=t[m], y=alpha[m], mode="lines", line=dict(color=ORANGE, width=1.6),
                name="alpha component", opacity=0.8)
fig.update_layout(title="Raw channel with its alpha component", xaxis_title="time (s)", yaxis_title="µV")
SHOW(fig)

# %%
fr, pxx = signal.welch(eeg, fs=FS, nperseg=int(20 * FS))
band = (fr > 0.5) & (fr < 60)
fig = go.Figure()
fig.add_scatter(x=fr[band], y=pxx[band], mode="lines", line=dict(color=BLUE, width=1.8), name="PSD")
fig.update_layout(
    title="Welch PSD — alpha peak on the aperiodic background",
    xaxis_title="frequency (Hz)", yaxis_title="PSD (µV²/Hz)",
    xaxis_type="log", yaxis_type="log",
)
SHOW(fig)

# %% [markdown]
# ## 5. Why phase alignment matters more than count
#
# The scalp electrode sums a great many cortical sources. Interference decides how that sum scales
# with the number of sources $N$ — and the answer depends entirely on their phases:
#
# - **Phase-locked** (all $\varphi_k$ equal): phasors add head-to-tail in a straight line,
#   amplitude $\propto N$.
# - **Independent phases** (uniform on $[0,2\pi)$): the phasor sum is a 2-D random walk of $N$
#   unit steps, so the expected amplitude $\propto \sqrt{N}$.
#
# That gap is the whole reason evoked potentials need averaging and why synchronisation — not
# firing rate — dominates scalp amplitude. Doubling the number of *incoherent* sources buys 41%;
# phase-locking the ones you already have buys everything.

# %%
Ns = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
trials = 2000
rand_amp, lock_amp = [], []
for N in Ns:
    ph = rng.uniform(0, 2 * np.pi, size=(trials, N))
    rand_amp.append(np.abs(np.exp(1j * ph).sum(axis=1)).mean())
    lock_amp.append(float(N))

fig = go.Figure()
fig.add_scatter(x=Ns, y=lock_amp, mode="lines+markers", line=dict(color=ORANGE, width=2), name="phase-locked ∝ N")
fig.add_scatter(x=Ns, y=rand_amp, mode="lines+markers", line=dict(color=BLUE, width=2), name="random phase (measured)")
fig.add_scatter(x=Ns, y=np.sqrt(np.pi / 4) * np.sqrt(Ns), mode="lines",
                line=dict(color=GREEN, width=2, dash="dash"), name="½√π · √N (Rayleigh mean)")
fig.update_layout(
    title="Resultant amplitude of N sources: coherent vs incoherent summation",
    xaxis_title="N sources", yaxis_title="mean resultant amplitude",
    xaxis_type="log", yaxis_type="log",
)
SHOW(fig)

# %% [markdown]
# The measured random-phase curve tracks $\tfrac{1}{2}\sqrt{\pi N}$ — the mean of a Rayleigh
# distribution, which is exactly what the magnitude of a 2-D Gaussian random walk follows. The
# interference story of §1 and the statistics of large sums are the same fact seen at two scales.

# %% [markdown]
# One caveat the numbers make visible: the Rayleigh mean is an *asymptotic* result, reached as the
# central limit theorem takes hold. At $N=1$ there is nothing random about the magnitude — a single
# unit phasor has resultant exactly 1, not $\tfrac12\sqrt\pi \approx 0.886$. So we assert the fit
# only from $N \ge 2$, and print the $N=1$ discrepancy rather than sweeping it under an `atol`.

# %%
ratio = np.array(rand_amp) / (np.sqrt(np.pi / 4) * np.sqrt(Ns))
print(f"N=1 (deterministic, pre-CLT): measured/Rayleigh = {ratio[0]:.4f}  ~ 1/(½√π) = {1/np.sqrt(np.pi/4):.4f}")
assert np.allclose(ratio[Ns >= 2], 1.0, atol=0.05), ratio
print(f"N≥2 random-phase growth matches ½√(πN) within 5% ✓   (max dev {np.abs(ratio[Ns>=2]-1).max():.3f})")

# %%
freqs=[14.0, 14.1,13.95, 14.07]
tarr, sig = neural_analysis.synthetic.generate_multi_component(
    freqs=freqs, 
    amplitudes=[40.0,40.0,40.0,40.0],
    sampling_rate=500,
    duration=10.0,
    noise_level=4.0,
    )

# %%
sig.shape

# %%
fig = go.Figure()
fig.add_scatter(x=tarr, y=sig, mode="lines",
                line=dict(color=ORANGE, width=2),
                name="4 sig phase-locked ∝ N"
    )
#fig.add_scatter(x=Ns, y=lock_amp, mode="lines+markers", line=dict(color=ORANGE, width=2), name="phase-locked ∝ N")
#fig.add_scatter(x=Ns, y=rand_amp, mode="lines+markers", line=dict(color=BLUE, width=2), name="random phase (measured)")
#fig.add_scatter(x=Ns, y=np.sqrt(np.pi / 4) * np.sqrt(Ns), mode="lines",
#                line=dict(color=GREEN, width=2, dash="dash"), name="½√π · √N (Rayleigh mean)")
fig.update_layout(
    title=f"Resultant amplitude of {len(freqs)} sources",
    xaxis_title="N sources", yaxis_title="mean resultant amplitude",
    # xaxis_type="log", yaxis_type="log",
)
# SHOW(fig)

# %%

# %%
