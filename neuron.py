import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""
    # 🧠 Hodgkin–Huxley neuron with stochastic synaptic input

    A single-compartment [Hodgkin–Huxley (HH) model](https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model) neuron driven by noisy excitatory and inhibitory synaptic input. Its synaptic conductances are [Ornstein–Uhlenbeck (OU) processes](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process), stepped with the [Euler–Maruyama scheme](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method).

    #### How to use

    - **Controls** (left): every change recomputes the plots and stats live.
    - **Random**: draw a fresh noise realization at the current settings.
    - **Seed**: fix the noise for a reproducible run; the active seed is shown under the button.

    ---

    ### Model specification

    **Membrane potential** — $C_m$ is the membrane [capacitance](https://en.wikipedia.org/wiki/Capacitance); each $E$ is a [reversal potential](https://en.wikipedia.org/wiki/Reversal_potential).

    $$ C_m \frac{dV}{dt} = I_{ext} - I_{Na} - I_K - I_L - I_{syn} $$

    **Ionic currents**, with [gating variables](https://en.wikipedia.org/wiki/Voltage-gated_ion_channel) $m, h, n$.

    $$ I_{Na} = \bar g_{Na}\, m^3 h\,(V - E_{Na}), \qquad I_K = \bar g_K\, n^4\,(V - E_K), \qquad I_L = g_L\,(V - E_L) $$

    **Synaptic current**, where $g_E, g_I$ are the excitatory and inhibitory [conductances](https://en.wikipedia.org/wiki/Electrical_resistance_and_conductance).

    $$ I_{syn} = g_E(t)\,(V - E_E) + g_I(t)\,(V - E_I) $$

    **Gating kinetics** — $m, h, n$ follow [first-order kinetics](https://en.wikipedia.org/wiki/Rate_equation).

    $$ \frac{dm}{dt} = \alpha_m(V)(1 - m) - \beta_m(V)\,m, \qquad \frac{dh}{dt} = \alpha_h(V)(1 - h) - \beta_h(V)\,h, \qquad \frac{dn}{dt} = \alpha_n(V)(1 - n) - \beta_n(V)\,n $$

    **Synaptic conductances** — independent [OU processes](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) driven by [white noise](https://en.wikipedia.org/wiki/White_noise) $\xi(t)$.

    $$ \tau_E \frac{dg_E}{dt} = (\bar g_E - g_E) + \sigma_E\,\xi_E(t), \qquad \tau_I \frac{dg_I}{dt} = (\bar g_I - g_I) + \sigma_I\,\xi_I(t) $$

    **Integration** — this [stochastic differential equation](https://en.wikipedia.org/wiki/Stochastic_differential_equation) is stepped with the [Euler–Maruyama method](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method) (stationary variance $\sigma^2 \tau / 2$).

    $$ g(t + \Delta t) = g(t) + \frac{\Delta t}{\tau}\big(\bar g - g(t)\big) + \sigma \sqrt{\Delta t}\,\mathcal{N}(0, 1) $$
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _(np):
    def alpham(V):
        return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))


    def betam(V):
        return 4 * np.exp(-(V + 65) / 18)


    def alphah(V):
        return 0.07 * np.exp(-(V + 65) / 20)


    def betah(V):
        return 1 / (1 + np.exp(-(V + 35) / 10))


    def alphan(V):
        return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))


    def betan(V):
        return 0.125 * np.exp(-(V + 65) / 80)

    return alphah, alpham, alphan, betah, betam, betan


@app.cell
def _(alphah, alpham, alphan, betah, betam, betan, np):
    def integrate_hh_ou(
        dt,
        tMax,
        Cm,
        gNa,
        ENa,
        gK,
        EK,
        gL,
        EL,
        Vrest,
        gE_mean,
        gI_mean,
        tauE,
        tauI,
        sigmaE,
        sigmaI,
        EE,
        EI,
        Iext,
        seed,
    ):
        nSteps = int(tMax / dt) + 1

        timeWave = np.arange(nSteps) * dt
        V = np.zeros(nSteps)
        m = np.zeros(nSteps)
        h = np.zeros(nSteps)
        n = np.zeros(nSteps)
        gE = np.zeros(nSteps)
        gI = np.zeros(nSteps)
        Isyn = np.zeros(nSteps)

        # Initial conditions
        V[0] = Vrest
        m[0] = alpham(Vrest) / (alpham(Vrest) + betam(Vrest))
        h[0] = alphah(Vrest) / (alphah(Vrest) + betah(Vrest))
        n[0] = alphan(Vrest) / (alphan(Vrest) + betan(Vrest))
        gE[0] = gE_mean
        gI[0] = gI_mean

        sqrt_dt = np.sqrt(dt)

        rng = np.random.default_rng(seed)
        noiseE = rng.standard_normal(nSteps)
        noiseI = rng.standard_normal(nSteps)

        for i in range(1, nSteps):
            # Euler-Maruyama update for OU conductances
            gE[i] = (
                gE[i - 1]
                + dt / tauE * (gE_mean - gE[i - 1])
                + sigmaE * sqrt_dt * noiseE[i]
            )
            gI[i] = (
                gI[i - 1]
                + dt / tauI * (gI_mean - gI[i - 1])
                + sigmaI * sqrt_dt * noiseI[i]
            )

            Isyn[i - 1] = gE[i - 1] * (V[i - 1] - EE) + gI[i - 1] * (V[i - 1] - EI)

            # Standard HH currents
            INa = gNa * m[i - 1] ** 3 * h[i - 1] * (V[i - 1] - ENa)
            IK = gK * n[i - 1] ** 4 * (V[i - 1] - EK)
            IL = gL * (V[i - 1] - EL)

            dVdt = (Iext - INa - IK - IL - Isyn[i - 1]) / Cm
            V[i] = V[i - 1] + dt * dVdt

            # Gating variable updates
            m[i] = m[i - 1] + dt * (alpham(V[i - 1]) * (1 - m[i - 1]) - betam(V[i - 1]) * m[i - 1])
            h[i] = h[i - 1] + dt * (alphah(V[i - 1]) * (1 - h[i - 1]) - betah(V[i - 1]) * h[i - 1])
            n[i] = n[i - 1] + dt * (alphan(V[i - 1]) * (1 - n[i - 1]) - betan(V[i - 1]) * n[i - 1])

        return timeWave, V, m, h, n, gE, gI, Isyn

    return (integrate_hh_ou,)


@app.cell
def _(mo):
    tMax = mo.ui.slider(100, 2000, value=1000, step=100, label="Duration tMax (ms)")
    dt = mo.ui.slider(0.005, 0.05, value=0.01, step=0.005, label="Time step dt (ms)")

    seed = mo.ui.number(value=0, label="Seed")
    reseed = mo.ui.button(
        label="Random",
        value=0,
        on_click=lambda n: n + 1,
        tooltip="Draw a fresh noise realization at the current settings",
        kind="success",
    )

    Iext = mo.ui.slider(-20, 40, value=0, step=1, label="Injected current Iext (uA/cm^2)")

    Cm = mo.ui.slider(0.1, 5, value=1, step=0.1, label="Membrane capacitance Cm (uF/cm^2)")
    Vrest = mo.ui.slider(-100, -40, value=-70, step=5, label="Resting potential Vrest (mV)")

    gNa = mo.ui.slider(0, 200, value=120, step=10, label="Sodium conductance gNa (mS/cm^2)")
    ENa = mo.ui.slider(-50, 100, value=50, step=5, label="Sodium reversal ENa (mV)")
    gK = mo.ui.slider(0, 100, value=36, step=2, label="Potassium conductance gK (mS/cm^2)")
    EK = mo.ui.slider(-120, 0, value=-90, step=5, label="Potassium reversal EK (mV)")
    gL = mo.ui.slider(0, 5, value=0.6, step=0.1, label="Leak conductance gL (mS/cm^2)")
    EL = mo.ui.slider(-100, 0, value=-70, step=5, label="Leak reversal EL (mV)")

    gE_mean = mo.ui.slider(0, 0.5, value=0.05, step=0.01, label="Mean excitatory gE (nS)")
    tauE = mo.ui.slider(1, 50, value=6, step=1, label="Excitatory time constant tauE (ms)")
    sigmaE = mo.ui.slider(0, 0.5, value=0.05, step=0.01, label="Excitatory noise sigmaE (nS)")
    EE = mo.ui.slider(-20, 20, value=0, step=1, label="Excitatory reversal EE (mV)")

    gI_mean = mo.ui.slider(0, 0.5, value=0.05, step=0.01, label="Mean inhibitory gI (nS)")
    tauI = mo.ui.slider(1, 50, value=10, step=1, label="Inhibitory time constant tauI (ms)")
    sigmaI = mo.ui.slider(0, 0.5, value=0.05, step=0.01, label="Inhibitory noise sigmaI (nS)")
    EI = mo.ui.slider(-100, 0, value=-75, step=5, label="Inhibitory reversal EI (mV)")

    controls_panel = mo.accordion(
        {
            "Simulation": mo.vstack(
                [tMax, dt, mo.hstack([seed, reseed], justify="start", align="end", gap=1)]
            ),
            "Membrane & leak": mo.vstack([Cm, Vrest, gL, EL]),
            "Na / K channels": mo.vstack([gNa, ENa, gK, EK]),
            "Excitatory synapse (OU)": mo.vstack([gE_mean, tauE, sigmaE, EE]),
            "Inhibitory synapse (OU)": mo.vstack([gI_mean, tauI, sigmaI, EI]),
        },
        multiple=True,
    )
    return (
        Cm,
        EE,
        EI,
        EK,
        EL,
        ENa,
        Iext,
        Vrest,
        controls_panel,
        dt,
        gE_mean,
        gI_mean,
        gK,
        gL,
        gNa,
        reseed,
        seed,
        sigmaE,
        sigmaI,
        tMax,
        tauE,
        tauI,
    )


@app.cell
def seed_control(mo, reseed, seed):
    effective_seed = int(seed.value) + 1_000_003 * int(reseed.value)
    seed_readout = mo.md(
        f"Active seed: **{effective_seed}**  \n"
        "<span style='color:#888'>Type a base seed to reproduce a run, or press Random for a new trial.</span>"
    )
    return effective_seed, seed_readout


@app.cell
def _(
    Cm,
    EE,
    EI,
    EK,
    EL,
    ENa,
    Iext,
    Vrest,
    dt,
    effective_seed,
    gE_mean,
    gI_mean,
    gK,
    gL,
    gNa,
    integrate_hh_ou,
    np,
    plt,
    sigmaE,
    sigmaI,
    tMax,
    tauE,
    tauI,
):
    time, V, m, h, n, gE, gI, Isyn = integrate_hh_ou(
        dt=dt.value,
        tMax=tMax.value,
        Cm=Cm.value,
        gNa=gNa.value,
        ENa=ENa.value,
        gK=gK.value,
        EK=EK.value,
        gL=gL.value,
        EL=EL.value,
        Vrest=Vrest.value,
        gE_mean=gE_mean.value,
        gI_mean=gI_mean.value,
        tauE=tauE.value,
        tauI=tauI.value,
        sigmaE=sigmaE.value,
        sigmaI=sigmaI.value,
        EE=EE.value,
        EI=EI.value,
        Iext=Iext.value,
        seed=effective_seed,
    )

    # Panel 1: membrane potential and synaptic current
    fig_membrane, (ax_v, ax_isyn) = plt.subplots(
        2, 1, figsize=(7, 4.2), sharex=True, constrained_layout=True
    )
    ax_v.plot(time, V, color="#1f77b4", linewidth=1.0)
    ax_v.axhline(0, color="#888888", linewidth=0.7, linestyle="--")
    ax_v.set_ylabel("V (mV)")
    ax_v.set_ylim(-120, 60)
    ax_v.grid(alpha=0.3)
    ax_isyn.plot(time, Isyn, color="#d62728", linewidth=0.8)
    ax_isyn.set_ylabel("Isyn (pA)")
    ax_isyn.set_xlabel("Time (ms)")
    isyn_pad = 0.1 * (Isyn.max() - Isyn.min() + 1e-9)
    ax_isyn.set_ylim(Isyn.min() - isyn_pad, Isyn.max() + isyn_pad)
    ax_isyn.grid(alpha=0.3)

    # Panel 2: excitatory and inhibitory synaptic conductances
    fig_synaptic, (ax_ge, ax_gi) = plt.subplots(
        2, 1, figsize=(7, 4.2), sharex=True, constrained_layout=True
    )
    g_span = max(np.abs(gE).max(), np.abs(gI).max(), 0.1) * 1.1
    ax_ge.plot(time, gE, color="#2ca02c", linewidth=0.8)
    ax_ge.set_ylabel("gE (nS)")
    ax_ge.set_ylim(-g_span, g_span)
    ax_ge.grid(alpha=0.3)
    ax_gi.plot(time, gI, color="#9467bd", linewidth=0.8)
    ax_gi.set_ylabel("gI (nS)")
    ax_gi.set_xlabel("Time (ms)")
    ax_gi.set_ylim(-g_span, g_span)
    ax_gi.grid(alpha=0.3)

    # Panel 3: gating variables
    fig_gating, ax_g = plt.subplots(figsize=(7, 3.2), constrained_layout=True)
    ax_g.plot(time, m, color="#2ca02c", linewidth=1.0, label="m  (Na activation)")
    ax_g.plot(time, h, color="#9467bd", linewidth=1.0, label="h  (Na inactivation)")
    ax_g.plot(time, n, color="#ff7f0e", linewidth=1.0, label="n  (K activation)")
    ax_g.set_xlabel("Time (ms)")
    ax_g.set_ylabel("Gating variable")
    ax_g.set_ylim(-0.05, 1.05)
    ax_g.legend(loc="center right", framealpha=0.9)
    ax_g.grid(alpha=0.3)
    return V, fig_gating, fig_membrane, fig_synaptic, gE, gI, time


@app.cell(hide_code=True)
def _(V, gE, gI, mo, np, time):
    spike_threshold_mV = 0.0
    spike_mask = (V[:-1] <= spike_threshold_mV) & (V[1:] > spike_threshold_mV)
    spike_count = int(np.count_nonzero(spike_mask))
    duration_s = float(time[-1] - time[0]) / 1000.0
    firing_rate_hz = spike_count / duration_s if duration_s > 0 else 0.0

    readouts = mo.hstack(
        [
            mo.stat(value=f"{spike_count}", label="Spikes", caption=f"in {duration_s * 1000:.0f} ms", bordered=True),
            mo.stat(value=f"{firing_rate_hz:.1f} Hz", label="Firing rate", bordered=True),
            mo.stat(value=f"{gE.mean():.3f} nS", label="Mean gE", bordered=True),
            mo.stat(value=f"{gI.mean():.3f} nS", label="Mean gI", bordered=True),
        ],
        justify="start",
        wrap=True,
        gap=0.75,
    )

    return (readouts,)


@app.cell
def _(
    controls_panel,
    fig_gating,
    fig_membrane,
    fig_synaptic,
    mo,
    readouts,
    seed_readout,
):
    controls = mo.vstack(
        [mo.md("### ⚙️ Controls"), controls_panel, seed_readout],
        align="start",
        gap=0.5,
    )

    panels = mo.vstack(
        [
            readouts,
            mo.md(
                "**Membrane potential & synaptic current**  \n"
                "Blue: membrane potential $V(t)$. Red: synaptic current $I_{syn}(t)$. "
                "The dashed line marks 0 mV, a common spike threshold."
            ),
            mo.mpl.interactive(fig_membrane),
            mo.md(
                "**Synaptic conductances (OU)**  \n"
                "Green: excitatory $g_E(t)$. Purple: inhibitory $g_I(t)$."
            ),
            mo.mpl.interactive(fig_synaptic),
            mo.md(
                "**Gating variables**  \n"
                "$m$ = Na activation, $h$ = Na inactivation, $n$ = K activation."
            ),
            mo.mpl.interactive(fig_gating),
        ],
        gap=1.0,
    )

    dashboard = mo.hstack(
        [controls, panels],
        widths=[1, 3],
        align="start",
        gap=1.5,
    )

    dashboard
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### 🔬 Things to try

    - **Draw new noise realizations.** Keep every parameter fixed and press
      **Random** several times. How much does the spike count change?
      That spread is trial-to-trial variability caused purely by synaptic noise.
    - **Push the neuron across threshold.** Raise **Iext** slowly from 0. At some
      value the firing switches from sporadic, noise-driven spikes to regular tonic
      firing — find that transition. Then set `sigmaE = sigmaI = 0` to see the
      sharp, deterministic onset.
    - **Balance excitation and inhibition.** Increase **gE** and **gI** together.
      Strong E and I can cancel on average while greatly increasing the
      *fluctuations*: watch the membrane potential get more jagged even though its
      mean barely moves.
    - **Change the input timescales.** `tauE` and `tauI` set how long synaptic
      conductance excursions last: small `tau` gives fast, uncorrelated noise;
      large `tau` gives slow drifts. See how this reshapes the membrane trace.
    - **Break the channels.** Set **gNa = 0** (no spikes) or **gK = 0** (no
      repolarisation) and describe what happens — the classic ablation experiments
      from the original Hodgkin–Huxley work.

    > **Tip:** the *active seed* under the button identifies the exact noise
    > realization. Note it down to re-create a trace later.
    """)
    return


if __name__ == "__main__":
    app.run()
