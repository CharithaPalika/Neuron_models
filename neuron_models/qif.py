import numpy as np
from matplotlib import pyplot as plt
import gradio as gr
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
import io

from matplotlib import rcParams
rcParams["font.family"] = "Times New Roman"
rcParams["font.size"] = 12
rcParams["axes.spines.top"] =  False
rcParams["axes.spines.right"] = False


import numpy as np

class QIFModel:
    """
    Quadratic Integrate-and-Fire neuron model (ms version).

    Equation:
        dV/dt = (V^2 + I_ext) / tau

    Spike rule:
        If V >= V_peak → register spike, reset to V_reset,
        optionally hold neuron at V_reset for a refractory period.

    Parameters
    ----------
    tau      : membrane time constant (ms)
    V_peak   : spike threshold (mV)
    V_reset  : reset voltage after spike (mV)
    T_ref    : refractory duration (ms)
    """

    def __init__(self, tau=10.0, V_peak=30.0, V_reset=-60.0, T_ref=2.0):
        self.tau = tau          # ms
        self.V_peak = V_peak    # mV
        self.V_reset = V_reset  # mV
        self.T_ref = T_ref      # ms

    def dv_dt(self, V, I_ext):
        """
        RHS of the QIF ODE:
        dV/dt = (V^2 + I_ext) / tau   (all in ms units)
        """
        return (V**2 + I_ext) / self.tau

    def solve(self, V0, dt, iterations, I_ext=0.0, use_refractory=False):
        """
        Integrate the QIF neuron using forward Euler.

        Parameters
        ----------
        V0           : initial voltage (mV)
        dt           : timestep in ms
        iterations   : number of updates
        I_ext        : external input (scalar or array of length iterations+1)
        use_refractory : hold V at reset for T_ref after spike

        Returns
        -------
        time        : numpy array (ms)
        V_trace     : numpy array (mV)
        spike_times : list of spike times (ms)
        """

        N = iterations + 1
        V = np.empty(N, dtype=float)
        V[0] = float(V0)

        # Input current preparation
        if np.isscalar(I_ext):
            I = np.ones(N, dtype=float) * float(I_ext)
        else:
            I = np.asarray(I_ext, dtype=float)
            if I.shape[0] != N:
                raise ValueError("I_ext array must be length iterations+1")

        spike_times = []

        # Convert refractory duration to number of steps
        ref_steps = int(round(self.T_ref / dt)) if (use_refractory and self.T_ref > 0) else 0

        for i in range(1, N):

            # Detect spike based on previous step value
            if V[i-1] >= self.V_peak:
                spike_times.append((i-1) * dt)
                V[i-1] = self.V_reset

                # apply refractory hold
                if ref_steps > 0:
                    end_idx = min(N, i - 1 + ref_steps)
                    V[i-1:end_idx] = self.V_reset
                    i = end_idx
                    if i >= N:
                        break

            # Euler update
            dV = self.dv_dt(V[i-1], I[i-1])
            V[i] = V[i-1] + dV * dt

        time = np.arange(0, N * dt, dt)

        return time, V, spike_times




# -------------------------------------------------------------
# QIF simulation wrapper for Gradio
# -------------------------------------------------------------
def run_qif_simulation(
    V0,
    I_in,
    tau,
    V_peak,
    V_reset,
    T_ref,
    duration,
    dt,
    use_refractory,
):
    """
    Runs QIFModel and returns three matplotlib figures:
    - Input current (constant)
    - Voltage trace
    - Spike raster
    """

    # iterations from duration and dt (both in ms)
    iterations = int(duration / dt)

    # instantiate model
    model = QIFModel(
        tau=float(tau),
        V_peak=float(V_peak),
        V_reset=float(V_reset),
        T_ref=float(T_ref),
    )

    # solve
    time, V, spike_times = model.solve(
        V0=float(V0),
        dt=float(dt),
        iterations=iterations,
        I_ext=float(I_in),
        use_refractory=use_refractory,
    )

    # 1) Current (constant)
    fig_I, ax_I = plt.subplots(figsize=(6, 3))
    ax_I.plot(time, np.ones_like(time) * I_in, color="blue")
    ax_I.set_title("Input Current")
    ax_I.set_xlabel("Time (ms)")
    ax_I.set_ylabel("I (a.u.)")
    ax_I.grid(True)

    # 2) Voltage trace
    fig_V, ax_V = plt.subplots(figsize=(6, 3))
    ax_V.plot(time, V, color="black")
    ax_V.set_title(
        f"QIF Voltage Trace "
        + ("(Refractory ON)" if use_refractory else "(Refractory OFF)")
    )
    ax_V.set_xlabel("Time (ms)")
    ax_V.set_ylabel("V (mV)")
    ax_V.grid(True)

    # 3) Spike raster
    fig_S, ax_S = plt.subplots(figsize=(6, 2))
    if len(spike_times) > 0:
        ax_S.scatter(spike_times, np.ones_like(spike_times),
                     marker="|", s=200, color="red")
    ax_S.set_ylim(0, 2)
    ax_S.set_title("Spike Raster")
    ax_S.set_xlabel("Time (ms)")
    ax_S.set_yticks([])
    ax_S.grid(False)

    return fig_I, fig_V, fig_S


# -------------------------------------------------------------
# GRADIO APP
# -------------------------------------------------------------
app = gr.Interface(
    fn=run_qif_simulation,
    inputs=[
        gr.Number(value=-20.0, label="Initial V₀ (mV)"),
        gr.Slider(-20, 20, value=20.0, step=1.0, label="Input Current I_ext (a.u.)"),

        gr.Slider(1.0, 50.0, value=10.0, step=0.5, label="τ (ms)"),
        gr.Slider(-20.0, 80.0, value=60.0, step=1.0, label="V_peak (mV)"),
        gr.Slider(-100.0, 0.0, value=-20.0, step=1.0, label="V_reset (mV)"),
        gr.Slider(0.0, 20.0, value=2.0, step=0.5, label="Refractory T_ref (ms)"),

        gr.Slider(10.0, 2000.0, value=500.0, step=10.0, label="Duration (ms)"),
        gr.Slider(0.01, 1.0, value=0.1, step=0.01, label="Time Step dt (ms)"),

        gr.Checkbox(value=True, label="Use Refractory"),
    ],
    outputs=[
        gr.Plot(label="Input Current"),
        gr.Plot(label="Voltage Trace"),
        gr.Plot(label="Spike Raster"),
    ],
    title="Quadratic Integrate-and-Fire (QIF) Neuron Simulator",
    description="Simulate a QIF neuron with adjustable parameters and view current, voltage, and spike raster.",
    flagging_mode= "never",
)


def launch_qif_app():
    app.launch(share=False)
