import marimo

__generated_with = "0.16.5"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""# Hodgkin-Huxley neuron with Ornstein-Uhlenbeck synaptic input""")
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt


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


    def integrate_hh_ou():
        dt = 0.01 # ms
        tMax = 1000  # ms
        Cm = 1  # uF/cm^2
        gNa, ENa = 120, 50 # mS/cm^2, mV
        gK, EK = 36, -90
        gL, EL = 0.6, -70
        Vrest = -70
        nSteps = int(tMax / dt) + 1
        gE_mean, gI_mean = 0.05, 0.05 # nS
        tauE, tauI = 6, 10 # ms
        sigmaE, sigmaI = 0.05, 0.05 # stdev of excitatory and inhibitory synaptic conductances
        EE, EI = 0, -75 # mV, theis reversal potentials
        Iext = 0  # injected current in uA/cm^2

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

        sqrt_dt = np.sqrt(dt) # Precompute sqrt(dt) because why not?

        noiseE = np.random.normal(0, 1, nSteps) # Same here
        noiseI = np.random.normal(0, 1, nSteps) # and here

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

        return timeWave, V, m, h, n, gE, gI, Isyn, sineWave


    time, V, m, h, n, gE, gI, Isyn, sineWave = integrate_hh_ou()

    fig, axs = plt.subplots(4, 1)

    axs[0].plot(time, V, linewidth=1)
    axs[0].set(ylim=(-120, 60), ylabel="Membrane potential (mV)")
    axs[1].plot(time, Isyn, linewidth=0.8)
    axs[1].set(ylim=(-50, 50), ylabel="Synaptic current (pA)")
    axs[2].plot(time, gE, linewidth=0.8)
    axs[2].set(ylim=(-0.5, 0.5), ylabel="Excitatory conductance (nS)")
    axs[3].plot(time, gI, linewidth=0.8)
    axs[3].set(ylim=(-0.5, 0.5), ylabel="Inhibitory conductance (nS)")

    for ax in axs.flat:
        ax.set(xlabel="Time (ms)")
        ax.label_outer()
        ax.sharex(axs[0])

    fig.set_figwidth(8)
    fig.set_figheight(10)
    fig.tight_layout()
    fig.align_ylabels(axs)

    mo.mpl.interactive(fig)
    return (mo,)


if __name__ == "__main__":
    app.run()
