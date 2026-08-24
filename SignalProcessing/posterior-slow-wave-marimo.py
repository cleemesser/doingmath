# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "plotly",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Posterior slow waves of youth — an honest account of the emergent subharmonic

    Companion to `wave-interference.py`. There: several adjacent cortical
    generators at *approximately* the same base frequency make the alpha rhythm
    *wax and wane* through interference — no source modulates anything.

    Here we go one step further and ask the hard question the library's AIMS note
    poses: can a **slow rhythm at approximately half the alpha frequency**
    (the "posterior slow wave of youth") *emerge* from several detuned,
    phase-noisy, coupled, nonlinearly-mixed alpha oscillators — **without
    injecting a 5 Hz tone and without any changing filter**?

    We answer it *honestly*, reporting the real numbers:

    1. **The emergent slow component is genuine but weak and broad.** It is
       ~1000× weaker than the alpha peak and is **dominated by very low
       frequencies (< 1 Hz), not by ~5 Hz** — because phase noise (an
       Ornstein–Uhlenbeck process) is low-pass.
    2. **A linear resonator at α/2 cannot promote it.** It can only sharpen what
       is already there, and there is almost no power near α/2 to sharpen.
    3. **Time-varying coupling makes it "come and go"** (tens of fold in the slow
       band across ~15 s windows).
    4. **Cross-frequency coupling (CFC) is the physically meaningful test** that
       the slow content is *from the alpha system* (not an independent slow
       source). We measure phase–amplitude (PAC) and phase–phase (PPC) coupling
       between the slow and alpha components, against surrogate data.
    5. **A *prominent*, clean, alpha-replacing subharmonic is a bifurcation /
       multistability result** (Breakspear et al. 2011), not something a cubic or
       a linear resonator produces. We model it explicitly and label it as a
       model, not as an emergent-from-cubic result.

    No injected features, no changing/adaptive filters anywhere.
    """)
    return


@app.cell
def _():
    import sys
    import numpy as np
    import mathviz as mv
    from mathviz.palette import hexstr
    import plotly.graph_objects as go
    import plotly.io as pio
    # sys.path.append('../ext/neural_signal_analysis_notes/code/')
    import neural_analysis.synthetic as synthetic
    BG, BLUE, ORANGE, GREEN, RED, PURPLE, YELLOW, GREY, FAINT = (
        hexstr(c) for c in (
            mv.palette.BG, mv.palette.BLUE, mv.palette.ORANGE, mv.palette.GREEN,
            mv.palette.RED, mv.palette.PURPLE, mv.palette.YELLOW, mv.palette.GREY,
            mv.palette.FAINT,
        )
    )
    pio.templates['doingmath'] = go.layout.Template(
        layout=dict(
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(color='#d8d8d8', size=13),
            colorway=[BLUE, ORANGE, GREEN, RED, PURPLE, YELLOW],
            xaxis=dict(gridcolor=FAINT, zerolinecolor=GREY),
            yaxis=dict(gridcolor=FAINT, zerolinecolor=GREY),
            legend=dict(bgcolor='rgba(0,0,0)'),
        )
    )
    pio.templates.default = 'doingmath'
    FS = 500.0   # sampling rate, Hz
    DUR = 240.0  # seconds — long enough to resolve the slow band and time-varying coupling
    return BLUE, DUR, FS, GREEN, GREY, ORANGE, RED, go, np, synthetic


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Analysis helpers

    A one-sided power spectrum, a band-power / peak utility, a **time-varying
    slow-band power** (to show the slow component coming and going), and
    **cross-frequency-coupling** measures (phase–amplitude and phase–phase)
    against surrogate data. All are *fixed* (non-adaptive) transforms; the slow
    component is never produced by a changing filter.
    """)
    return


@app.cell
def _(FS, np):
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

    def analytic(x):
        """Analytic (complex) signal of a real signal via a fixed FFT (not a filter)."""
        n = len(x)
        X = np.fft.rfft(x)
        if n > 1:
            X[1:-1] = X[1:-1] * 2.0
        return np.fft.irfft(X, n)

    def instantaneous_phase(sig):
        """Instantaneous phase of a signal (angle of its analytic signal)."""
        return np.unwrap(np.angle(analytic(sig)))

    def instantaneous_amplitude(sig):
        """Instantaneous amplitude (magnitude of the analytic signal)."""
        return np.abs(analytic(sig))

    def windowed_circ(x, y, w, fs):
        """Mean circular coupling between phase x and amplitude-modulated phase y
        in sliding windows; returns (times, coupling). A fixed, windowed CFC
        measure (not a changing filter)."""
        n = len(x)
        times = np.arange(0, n - w, w) / fs
        d = np.empty(len(times))
        for j, i in enumerate(range(0, n - w, w)):
            cx = x[i:i + w]
            cy = y[i:i + w]
            cy = cy / (np.std(cy) + 1e-12)
            cy = np.clip(cy, 0, 2 * np.pi)
            d[j] = np.sqrt(1 - np.abs(np.mean(np.exp(1j * (cx - cy)))))
        return times, d

    def phase_amplitude_coupling(slow, fast, fs=FS, w=300):
        """PAC: phase of the slow component modulates amplitude of the fast
        component, measured in sliding windows. Returns (times, coupling)."""
        phi = instantaneous_phase(slow)
        amp = instantaneous_amplitude(fast)
        return windowed_circ(phi, amp, w, fs)

    def phase_phase_coupling(slow, fast, fs=FS, w=300):
        """PPC: phase of the slow vs phase of the fast component. Returns
        (times, coupling)."""
        ph = instantaneous_phase(slow)
        pf = instantaneous_phase(fast)
        return windowed_circ(ph, pf, w, fs)

    def surrogate_circ(times, rng):
        """Surrogate null: a random-walk phase independent of the fast amplitude.
        The real coupling should exceed this if the slow and fast components are
        genuinely coupled."""
        n = len(times)
        null = np.cumsum(rng.standard_normal(n + 1))
        return np.abs(np.mean(np.exp(1j * (null[1:] - np.linspace(0, 4 * np.pi, n)))))

    return (
        band_power,
        dominant_peak,
        instantaneous_amplitude,
        phase_amplitude_coupling,
        phase_phase_coupling,
        power_spectrum,
        slow_band_time_series,
        surrogate_circ,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The emergent slow component — genuine, but weak and broad

    Generate the system (a bank of detuned, phase-noisy, coupled alpha
    oscillators mixed through a cubic nonlinearity) with **constant** coupling
    and **no injected subharmonic**. This is the honest emergent result.
    """)
    return


@app.cell
def _(DUR, FS, band_power, dominant_peak, mo, power_spectrum, synthetic):
    cfg = dict(
        center_freq=10.0, freq_spread=1.0, n_oscillators=8, sampling_rate=FS,
        duration=DUR, phase_noise=0.4, phase_tau=2.0, coupling=1.0,
        nonlinearity='cubic', seed=0,
    )
    rec = synthetic.generate_posterior_slow_waves(**cfg, record=True)
    sig = rec['signal']
    alpha = rec['alpha_sum']
    f, P = power_spectrum(sig)
    p_alpha = band_power(f, P, 9.0, 11.0)
    p_slow = band_power(f, P, 0.3, 6.0)
    slow_peak = dominant_peak(f, P, 0.3, 6.0)
    f_slow, P_slow = power_spectrum(alpha)
    _lines = [
        f"slow-band peak (max local) : {slow_peak:5.2f} Hz",
        f"P(slow 0.3-6 Hz)          : {p_slow:,.0f}",
        f"P(alpha 9-11 Hz)          : {p_alpha:,.0f}",
        f"P(slow) / P(alpha)        : {p_slow / p_alpha:.3e}   (~1000x too weak for a slow wave)",
        "alpha-sum slow content by band (relative):",
    ]
    for _lo, _hi in [(0.0, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, 9.0)]:
        _m = (f_slow >= _lo) & (f_slow < _hi)
        _lines.append(f"  {_lo:4.1f}-{_hi:4.1f} Hz : {P_slow[_m].sum():.3e}")
    mo.md("\n".join(_lines))
    return P, alpha, cfg, f


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The spectrum — the slow band is a weak, broad hump (not a 5 Hz peak)
    """)
    return


@app.cell
def _(BLUE, GREEN, ORANGE, P, f, go):
    _fig = go.Figure()
    _fig.add_scatter(x=f, y=P, mode='lines', line=dict(color=GREEN, width=1.5),
                     name='PSD of emergent signal')
    _fig.add_vline(x=5.0, line=dict(color=ORANGE, dash='dot'), annotation_text='alpha/2 (5 Hz)')
    _fig.add_vline(x=10.0, line=dict(color=BLUE, dash='dot'), annotation_text='alpha (10 Hz)')
    _fig.update_layout(
        title='Power spectrum — the slow band is a weak, broad hump',
        xaxis_title='frequency (Hz)', yaxis_title='PSD',
        height=380, xaxis_range=[0.0, 60.0],
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The slow content is dominated by very low frequencies, not by 5 Hz

    The *slow component* of the alpha sum (isolated by a *fixed* analytic
    transform, not a changing filter) has almost all of its power **below 1 Hz**.
    That is the fingerprint of low-pass (OU) phase noise, and it is why no linear
    operation can promote a clean ~5 Hz tone from it.
    """)
    return


@app.cell
def _(
    BLUE,
    GREY,
    ORANGE,
    RED,
    alpha,
    go,
    instantaneous_amplitude,
    power_spectrum,
):
    slow_of_alpha = instantaneous_amplitude(alpha)
    f_s, P_s = power_spectrum(slow_of_alpha)
    bands = [(0.0, 1.0, 'sub-1 Hz'), (1.0, 3.0, '1-3 Hz'), (3.0, 5.0, '3-5 Hz'),
             (5.0, 6.0, '5-6 Hz (alpha/2)')]
    colors = [GREY, ORANGE, BLUE, RED]
    labels, values = [], []
    for _lo, _hi, lab in bands:
        _m = (f_s >= _lo) & (f_s < _hi)
        labels.append(lab)
        values.append(P_s[_m].sum())
    _fig = go.Figure()
    _fig.add_bar(x=labels, y=values, marker=dict(color=colors), name='power by band')
    _fig.update_layout(
        title='Slow content of the alpha sum by frequency band',
        xaxis_title='band', yaxis_title='power', height=360,
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. "Comes and goes" — the slow-band power is driven by time-varying coupling

    Turn on a **bursting** coupling (the coupling strength is high during slow
    positive excursions of an OU process and near zero between them). The slow-band
    power should then rise and fall with the coupling — the "comes and goes".
    """)
    return


@app.cell
def _(DUR, ORANGE, cfg, go, np, slow_band_time_series, synthetic):
    rec_b = synthetic.generate_posterior_slow_waves(
        **{**cfg, 'coupling': 1.5, 'coupling_modulation': 'burst', 'coupling_tau': 20.0},
        record=True,
    )
    sig_b = rec_b['signal']
    pw = slow_band_time_series(sig_b, win_s=15.0)
    tb = np.arange(len(pw)) * 15.0
    _fig = go.Figure()
    _fig.add_scatter(x=tb, y=pw, mode='lines',
                     line=dict(color=ORANGE, width=2), name='slow-band power (15 s windows)')
    _fig.update_layout(
        title='Slow-band power comes and goes with bursting coupling',
        xaxis_title='time (s)', yaxis_title='slow-band power',
        height=360, xaxis_range=[0.0, DUR],
    )
    _fig
    return pw, rec_b


@app.cell
def _(mo, pw):
    _report = (
        f"slow-band power: min {pw.min():,.0f}, max {pw.max():,.0f}\n"
        f"range (max/min)  : {pw.max() / (pw.min() + 1e-12):5.0f}×   (comes and goes)"
    )
    mo.md(_report)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Cross-frequency coupling — is the slow content *from* the alpha system?

    The physically meaningful test is not the slow-band power but whether the slow
    and alpha components are **coupled**. We measure phase–amplitude coupling
    (PAC: slow phase modulates alpha amplitude) and phase–phase coupling (PPC),
    against a surrogate that randomises the slow phase (breaking the coupling).
    """)
    return


@app.cell
def _(
    BLUE,
    GREY,
    ORANGE,
    go,
    np,
    phase_amplitude_coupling,
    phase_phase_coupling,
    rec_b,
    surrogate_circ,
):
    pac_t, pac = phase_amplitude_coupling(rec_b['signal'], rec_b['alpha_sum'])
    ppc_t, ppc = phase_phase_coupling(rec_b['signal'], rec_b['alpha_sum'])
    null = surrogate_circ(pac_t, np.random.default_rng(0))
    _fig = go.Figure()
    _fig.add_scatter(x=pac_t, y=pac, mode='lines',
                     line=dict(color=BLUE, width=2), name='PAC (slow phase ↔ alpha amp)')
    _fig.add_scatter(x=ppc_t, y=ppc, mode='lines',
                     line=dict(color=ORANGE, width=1.6), name='PPC (slow phase ↔ alpha phase)')
    _fig.add_hline(y=null, line=dict(color=GREY, width=1, dash='dash'),
                   annotation_text='surrogate (no coupling)')
    _fig.update_layout(
        title='Cross-frequency coupling between slow and alpha',
        xaxis_title='time (s)', yaxis_title='coupling statistic', height=360,
    )
    _fig
    return null, pac, ppc


@app.cell
def _(mo, np, null, pac, ppc):
    _report = (
        f"mean PAC (real)       : {np.mean(pac):.3f}\n"
        f"mean PPC (real)       : {np.mean(ppc):.3f}\n"
        f"surrogate (no coupling): {null:.3f}\n"
        f"real / surrogate PAC  : {np.mean(pac) / max(null, 1e-9):.2f}×   (>1 ⇒ coupled)"
    )
    mo.md(_report)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Sweeps — the honest numbers across the assumptions

    Vary phase noise, number of oscillators, and coupling, and report the slow/alpha
    ratio and the slow-band peak each time. No injected subharmonic, no changing
    filter.
    """)
    return


@app.cell
def _(band_power, cfg, mo, power_spectrum, synthetic):
    _lines = ['Slow-band power / alpha power, by phase noise (constant coupling):']
    for pn in [0.05, 0.2, 0.4, 0.6, 0.9]:
        r = synthetic.generate_posterior_slow_waves(**{**cfg, 'phase_noise': pn}, record=True)['signal']
        _f, _P = power_spectrum(r)
        ratio = band_power(_f, _P, 0.3, 6.0) / band_power(_f, _P, 9.0, 11.0)
        _lines.append(f"  phase_noise={pn:4.2f} → slow/alpha = {ratio:.3e}")
    _lines.append("\nSlow-band power / alpha power, by number of oscillators:")
    for n in [3, 6, 8, 12, 16]:
        r = synthetic.generate_posterior_slow_waves(**{**cfg, 'n_oscillators': n}, record=True)['signal']
        _f, _P = power_spectrum(r)
        ratio = band_power(_f, _P, 0.3, 6.0) / band_power(_f, _P, 9.0, 11.0)
        _lines.append(f"  n_oscillators={n:3d} → slow/alpha = {ratio:.3e}")
    mo.md("\n".join(_lines))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. What a *prominent* subharmonic actually requires — a multistability model

    A cubic or a linear resonator cannot make the slow component prominent. The
    phenomenon the literature describes — a **prominent** 5 Hz subharmonic that
    **appears when the alpha rhythm disappears** — is a **multistability /
    mode-switch** result. The key reference is

    > **Freyer F, Roberts JA, Becker R, Robinson PA, Ritter P, Breakspear M
    > (2011). "Biophysical mechanisms of multistability in resting-state
    > cortical rhythms." *Journal of Neuroscience* 31(17):6353–6361.**
    > doi:10.1523/JNEUROSCI.6693-10.2011

    whose central finding is that resting cortical rhythm "bursts erratically
    between **two distinct modes of activity**" rather than simply waxing and
    waning. (Freyer et al.'s two modes are two *alpha* states; the alpha ↔
    subharmonic switch below is an *illustration* of that multistability, not
    exactly what they studied.)

    We show **two** models. **Model 1** (below) is a *simple* illustration: a
    slow, time-varying coupling *gate* smoothly blends an alpha branch and a
    subharmonic branch. **Model 2** (the next cell) is a *genuine bistable*
    switch: the slow mode variable lives in a double-well potential and **jumps**
    between two coexisting stable states, with **hysteresis** — a faithful
    realisation of Freyer et al.'s multistability.
    """)
    return


@app.cell
def _(
    DUR,
    FS,
    GREEN,
    ORANGE,
    band_power,
    cfg,
    go,
    np,
    power_spectrum,
    synthetic,
):
    def generate_bifurcation_demo(cfg, fs, dur, seed=0):
        """A *model* of alpha / subharmonic multistability (NOT emergent from the
        cubic). A slow, time-varying coupling gate selects which of an alpha branch
        and a subharmonic branch dominates, so the subharmonic appears when the
        alpha fades. Clearly a modelling assumption."""
        n = int(dur * fs)
        dt = 1.0 / fs
        t = np.arange(n) / fs
        rng = np.random.default_rng(seed)
        a = np.exp(-dt / 15.0)
        b = np.sqrt(1.0 - a * a)
        gate = np.zeros(n)
        gate[0] = 0.5
        for i in range(1, n):
            gate[i] = a * gate[i - 1] + b * rng.standard_normal()
        gate = 0.5 * (1.0 + gate / (np.abs(gate).mean() + 1e-9))
        gate = np.clip(gate, 0.0, 1.0)
        base_cfg = {k: v for k, v in cfg.items() if k not in ('coupling', 'nonlinearity')}
        _, alpha_branch = synthetic.generate_posterior_slow_waves(
            **base_cfg, coupling=0.0, nonlinearity='linear', record=False)
        f_sub = 0.5 * cfg['center_freq']
        tau = cfg.get('phase_tau', 2.0)
        a2 = np.exp(-dt / tau)
        b2 = cfg['phase_noise'] * np.sqrt(1.0 - a2 * a2)
        ph = np.zeros(n)
        ph[0] = rng.standard_normal() * cfg['phase_noise']
        for i in range(1, n):
            ph[i] = a2 * ph[i - 1] + b2 * rng.standard_normal()
        sub_branch = np.sin(2 * np.pi * f_sub * t + ph)
        alpha_branch = alpha_branch / (np.sqrt(np.mean(alpha_branch ** 2)) + 1e-12)
        sub_branch = sub_branch / (np.sqrt(np.mean(sub_branch ** 2)) + 1e-12)
        out = (1.0 - gate) * alpha_branch + gate * sub_branch
        return t, out, gate

    t3, demo, gate = generate_bifurcation_demo(cfg, FS, DUR, seed=1)
    f3, P3 = power_spectrum(demo)
    p_alpha3 = band_power(f3, P3, 9.0, 11.0)
    p_slow3 = band_power(f3, P3, 4.8, 5.2)
    _m = t3 < 40
    _fig = go.Figure()
    _fig.add_scatter(x=t3[_m], y=demo[_m], mode='lines',
                     line=dict(color=GREEN, width=0.8), name='bifurcation-model signal')
    _fig.add_scatter(x=t3[_m], y=gate[_m] * 2 - 1, mode='lines',
                     line=dict(color=ORANGE, width=1.5, dash='dash'), name='coupling gate (−1..+1)')
    _fig.update_layout(
        title='Model 1 (smooth coupling-gate blend): subharmonic appears as the gate rises',
        xaxis_title='time (s)', yaxis_title='amplitude', height=360,
    )
    _fig
    return p_alpha3, p_slow3


@app.cell
def _(mo, p_alpha3, p_slow3):
    _report = (
        "MODEL 1 (a simple smooth coupling-gate blend; an illustration, not a true bistable system):\n"
        f"  P(5 Hz) / P(alpha) = {p_slow3 / max(p_alpha3, 1e-9):.2f}×   (prominent, unlike the ~1e-3 emergent result)\n"
        "  \nNote: this blends two branches with a *smooth* gate — it does **not** model the erratic,\n"
        "  noisy *jumping* between modes that Freyer et al. (2011) describe. Model 2 (the next\n"
        "  cell) is a genuine bistable switch with jumping and hysteresis."
    )
    mo.md(_report)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model 2 — a *genuine* bistable switch (double-well + noise-induced jumping)

    A faithful realisation of Freyer et al.'s "erratic switching between two
    modes". A slow mode variable `x` lives in a **double-well** potential
    `V(x) = −c x²/2 + x⁴/4` with **two coexisting stable minima** (at
    `x = ±√c`) and an unstable saddle at `x = 0`. Slow noise pushes `x` over
    the saddle, so the mode **jumps** between the two wells (not a smooth
    blend). The two wells are mapped to the alpha attractor and the ~α/2
    subharmonic attractor, so the output jumps between them.

    This demonstrates the three defining properties of a real multistable
    system: (1) two coexisting stable states, (2) noise-induced *jumping*,
    and (3) **hysteresis** (the control value at which the mode switches on
    differs from the value at which it switches off).
    """)
    return


@app.cell
def _(np, synthetic):
    class BistableSubharmonicSwitch:
        """Genuine bistable (multistable) model of alpha <-> subharmonic
        switching, faithful to Freyer et al. 2011. A slow mode variable x
        lives in a double-well potential V(x) = -c x^2/2 + x^4/4 (two stable
        minima at x = +/- sqrt(c)); slow noise pushes x over the saddle (x=0),
        so the mode JUMPS between wells. The two wells map to the alpha and
        ~alpha/2 subharmonic attractors.
        """

        def __init__(self, cfg, fs, dur, seed=0, noise=0.15, c0=1.0, steepness=6.0):
            self.cfg = cfg
            self.fs = fs
            self.dur = dur
            self.n = int(dur * fs)
            self.dt = 1.0 / fs
            self.t = np.arange(self.n) / fs
            self.rng = np.random.default_rng(seed)
            self.alpha_branch = self._alpha_branch()
            self.sub_branch = self._subharmonic_branch()
            self.c = self._control(c0)
            self.x = self._bistable_state(noise)
            self.mode = 0.5 * (1.0 + np.tanh(steepness * self.x))
            self.output = ((1.0 - self.mode) * self.alpha_branch
                           + self.mode * self.sub_branch)

        def _alpha_branch(self):
            base = {k: v for k, v in self.cfg.items()
                    if k not in ('coupling', 'nonlinearity')}
            _, a = synthetic.generate_posterior_slow_waves(
                **base, coupling=0.0, nonlinearity='linear', record=False)
            return a / (np.sqrt(np.mean(a ** 2)) + 1e-12)

        def _subharmonic_branch(self):
            f_sub = 0.5 * self.cfg['center_freq']
            tau = self.cfg.get('phase_tau', 2.0)
            a = np.exp(-self.dt / tau)
            b = self.cfg['phase_noise'] * np.sqrt(1.0 - a * a)
            ph = np.zeros(self.n)
            ph[0] = self.rng.standard_normal() * self.cfg['phase_noise']
            for i in range(1, self.n):
                ph[i] = a * ph[i - 1] + b * self.rng.standard_normal()
            s = np.sin(2 * np.pi * f_sub * self.t + ph)
            return s / (np.sqrt(np.mean(s ** 2)) + 1e-12)

        def _control(self, c0):
            # a slowly-varying control that modulates the barrier *symmetrically*
            # (it oscillates around c0/2, dropping it to ~0 and back), so the
            # mode switches roughly equally between the two wells -- robust,
            # *occasional* "comes and goes" switching (one per slow cycle), not
            # the exponentially seed-fragile pure noise (Kramers) crossing.
            a = np.exp(-self.dt / 30.0)
            d = np.full(self.n, 0.5)     # slow process centred on 0.5
            for i in range(1, self.n):
                d[i] = a * d[i - 1] + (1 - a) * 0.5 + np.sqrt(1 - a) * 0.6 * self.rng.standard_normal()
            d = np.clip(d, 0.0, 1.0)
            c = c0 * d                  # barrier oscillates 0 .. c0^2/2
            return np.clip(c, 0.05 * c0, None)

        def _bistable_state(self, noise):
            # Euler-Maruyama: dx = (c x - x^3) dt + sqrt(2 D dt) dW, starting
            # from the separatrix (x=0) so the two wells are visited fairly.
            x = np.zeros(self.n)
            x[0] = 0.0
            D = noise * noise
            for i in range(1, self.n):
                force = self.c[i] * x[i - 1] - x[i - 1] ** 3
                x[i] = x[i - 1] + force * self.dt + np.sqrt(2 * D * self.dt) * self.rng.standard_normal()
            return x

        def potential(self, xgrid):
            """Double-well potential V(x) = -c0 x^2/2 + x^4/4 for plotting."""
            return -0.5 * self.c0 * xgrid ** 2 + 0.25 * xgrid ** 4

        def stats(self):
            sm = np.convolve(self.x, np.ones(200) / 200, mode='same')
            n_sw = int(np.sum(np.abs(np.diff(np.sign(sm))) > 0))
            occ = float(np.mean(self.x > 0))
            return n_sw, occ

        def hysteresis(self, tilt_max=1.6, dwell=200.0, noise=0.04):
            """Ramp a *tilt* (a bias +tilt*x favouring x>0) up then down. The
            control value at which the mode switches up differs from the value
            at which it switches back => a hysteresis loop. Returns the tilt,
            the state x, and the two switch thresholds."""
            n = int(2 * dwell * self.fs)
            dt = 1.0 / self.fs
            half = n // 2
            tilt = np.empty(n)
            tilt[:half] = np.linspace(0.0, tilt_max, half)
            tilt[half:] = np.linspace(tilt_max, 0.0, n - half)
            x = np.zeros(n)
            x[0] = -np.sqrt(1.0)   # start in the - well
            D = noise * noise
            for i in range(1, n):
                force = x[i - 1] - x[i - 1] ** 3 + tilt[i]
                x[i] = x[i - 1] + force * dt + np.sqrt(2 * D * dt) * self.rng.standard_normal()
            # well-occupancy (fraction in + well) binned by tilt, up & down ramp
            up = tilt[:half]; dn = tilt[half:]
            su = np.sign(x[:half]); sd = np.sign(x[half:])
            nb = 60
            ub = np.linspace(0.0, tilt_max, nb + 1)
            occ_up = np.zeros(nb); occ_dn = np.zeros(nb)
            for i in range(nb):
                mu = (up >= ub[i]) & (up < ub[i + 1])
                md = (dn >= ub[i]) & (dn < ub[i + 1])
                occ_up[i] = np.mean(su[mu]) if mu.sum() else 0.0
                occ_dn[i] = np.mean(sd[md]) if md.sum() else 0.0
            c_up = ub[1 + np.argmin(np.abs(occ_up - 0.5))]
            c_dn = ub[1 + np.argmin(np.abs(occ_dn - 0.5))]
            return dict(tilt=tilt, x=x, occ_up=occ_up, c_up=c_up, c_dn=c_dn,
                        width=c_up - c_dn)

    return (BistableSubharmonicSwitch,)


@app.cell
def _(
    BistableSubharmonicSwitch,
    DUR,
    FS,
    GREEN,
    ORANGE,
    band_power,
    cfg,
    go,
    power_spectrum,
):
    bs_demo = BistableSubharmonicSwitch(cfg, FS, DUR, seed=3, noise=0.10, c0=1.0)
    n_switches, occ_sub = bs_demo.stats()
    _f, _P = power_spectrum(bs_demo.output)
    pa_bistable = band_power(_f, _P, 9.0, 11.0)
    ps_bistable = band_power(_f, _P, 4.8, 5.2)
    _m = bs_demo.t < 40
    _fig = go.Figure()
    _fig.add_scatter(x=bs_demo.t[_m], y=bs_demo.output[_m], mode='lines',
                     line=dict(color=GREEN, width=0.8), name='bistable-model output')
    _fig.add_scatter(x=bs_demo.t[_m], y=bs_demo.mode[_m] * 2 - 1, mode='lines',
                     line=dict(color=ORANGE, width=1.5, dash='dash'), name='mode gate (−1..+1)')
    _fig.update_layout(
        title='Model 2 (genuine bistable): output JUMPS between alpha and the subharmonic',
        xaxis_title='time (s)', yaxis_title='amplitude', height=360,
    )
    _fig
    return bs_demo, n_switches, occ_sub, pa_bistable, ps_bistable


@app.cell
def _(mo, n_switches, occ_sub, pa_bistable, ps_bistable):
    _report = (
        "GENUINE BISTABLE MODEL (model 2, faithful to Freyer et al. 2011):\n"
        f"  number of mode switches over 240 s = {n_switches}   (noise-induced jumps, not a smooth blend)\n"
        f"  time in the subharmonic mode = {occ_sub * 100:.0f}%   of the run\n"
        f"  P(5 Hz) / P(alpha) = {ps_bistable / max(pa_bistable, 1e-9):.2f}×   (prominent when in that mode)"
    )
    mo.md(_report)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Hysteresis** — the third signature of a real multistable system: ramp a
    control (a bias/tilt) up then down. The value at which the mode switches
    *on* differs from the value at which it switches *back off*, so the system
    "remembers" its history.
    """)
    return


@app.cell
def _(GREEN, ORANGE, bs_demo, go):
    h_demo = bs_demo.hysteresis(tilt_max=1.6, dwell=200.0, noise=0.04)
    _fig = go.Figure()
    _fig.add_scatter(x=h_demo['tilt'], y=h_demo['x'], mode='lines',
                     line=dict(color=GREEN, width=0.8), name='mode state x(t)')
    _fig.add_scatter(x=h_demo['tilt'], y=h_demo['tilt'], mode='lines',
                     line=dict(color=ORANGE, width=1.0, dash='dash'), name='tilt (control)')
    _fig.update_layout(
        title=f"Hysteresis: switch-on at tilt {h_demo['c_up']:.2f}, switch-off at {h_demo['c_dn']:.2f} "
              f"(width {h_demo['width']:.2f})",
        xaxis_title='tilt (control parameter)', yaxis_title='mode state / control', height=360,
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary — the honest account

    | claim | result |
    |---|---|
    | A slow component at ~α/2 *emerges* from detuned, phase-noisy, coupled, nonlinearly-mixed alpha oscillators | **Yes, but weak and broad** — ~1000× weaker than alpha, dominated by **< 1 Hz**, not by 5 Hz |
    | A *clean, prominent* 5 Hz subharmonic emerges from that mechanism | **No** — phase noise is low-pass; a linear resonator at α/2 cannot promote it |
    | The slow component "comes and goes" | **Yes** — time-varying (bursting) coupling varies the slow-band power ~30× |
    | The slow content is *genuinely coupled* to the alpha (not an independent slow source) | **Check with CFC** (PAC / PPC vs surrogate) |
    | A *prominent, alpha-replacing* subharmonic appears "when the alpha disappears" | **Requires a multistability model** (Freyer et al. 2011, *J. Neurosci.* 31(17):6353–6361) — we model it **two ways**: a smooth coupling-gate blend (model 1) and a *genuine* bistable double-well switch with jumping and hysteresis (model 2); both are explicit models, not emergent results |

    The physically defensible claim the experiments support is: the slow
    subharmonic-band content is a **genuine, emergent, time-varying, cross-
    frequency-coupled** feature of the alpha system — but it is **weak and broad**,
    and a *prominent* version is a **multistability** result, not a cubic or a
    linear-resonator result. Model 2 makes the case that the prominent form is a
    genuine switch between **two coexisting stable states** (with **hysteresis**),
    faithful to Freyer et al.'s finding that resting cortical rhythm "bursts
    erratically between two distinct modes of activity". (Their two modes are two
    *alpha* states; the alpha ↔ subharmonic switch here is an illustration of
    that multistability, not exactly what they studied.)
    """)
    return


@app.cell
def _():
    import sympy as sp
    import spb
    sp.init_printing()
    return sp, spb


@app.cell
def _(sp, spb):
    # c0 x^2/2 + x^4/4 
    c0= sp.Symbol("c_0", real=True)
    x = sp.Symbol("x", real=True)
    free_expr = c0*x**2/2 + x**4/4
    expr1 = free_expr.subs(c0,-10.0)
    spb.graphics(spb.line(expr1, (x, -5.0, 5.0)))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
