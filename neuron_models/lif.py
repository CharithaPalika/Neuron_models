import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from tqdm.autonotebook import tqdm
from matplotlib import rcParams
import numpy as np

import gradio as gr
import numpy as np
import plotly.graph_objects as go


class LIFModel:
    """
    Leaky Integrate-and-Fire neuron model
    C * dV/dt = I - g_l * (V - El)

    Parameters (set at construction):
      gl     : leak conductance
      Cm     : membrane capacitance
      El     : resting potential (mV)
      thresh : spike threshold (mV)
      T_ref  : refractory period (s) -- time to hold V at El after a spike
    """

    def __init__(self, gl=0.16, Cm=0.0049, El=-65.0, thresh=-50.0, T_ref=0.002):
        self.gl = gl
        self.Cm = Cm
        self.El = El
        self.thresh = thresh
        self.T_ref = T_ref

    def solve(self, V0, dt, iterations, I_ext=0.0,
              reset_to_El=True, use_refractory=False):
        """
        Run the simulation.

        Args:
          V0 (float): initial membrane potential (mV)
          dt (float): timestep (s)
          iterations (int): number of time steps to simulate
          I_ext (float or array-like): external input current
                 - scalar: constant current
                 - array: length must be iterations+1
          reset_to_El (bool): if True, on spike set V to El
          use_refractory (bool): if True, mimic original Ref behavior:
                                 after a spike, hold V at El for T_ref

        Returns:
          V (np.ndarray): voltage trace (length iterations+1, includes V0)
          spike_times (list of float): spike times in seconds
        """
        N = iterations + 1
        V = np.empty(N)
        V[0] = V0

        # Prepare input current array
        if np.isscalar(I_ext):
            I = np.ones(N) * I_ext
        else:
            I = np.asarray(I_ext)
            if I.shape[0] != N:
                raise ValueError("I_ext array length must equal iterations+1")

        spike_times = []

        # number of steps for refractory period
        ref_steps = int(round(self.T_ref / dt)) if (use_refractory and self.T_ref > 0) else 0

        i = 1
        while i < N:
            # Euler update
            dV = (I[i] - self.gl * (V[i-1] - self.El)) / self.Cm
            V[i] = V[i-1] + dV * dt

            # Check for spike
            if V[i] > self.thresh:
                spike_times.append(i * dt)

                # reset voltage
                if reset_to_El:
                    V[i] = self.El

                # apply refractory if requested
                if ref_steps > 0:
                    for j in range(1, ref_steps + 1):
                        if i + j < N:
                            V[i + j] = self.El
                    i = i + ref_steps  # skip ahead past refractory period

            i += 1

        return V, spike_times



# -------------------------------------------------------------
# LIF simulation wrapper for Gradio
# -------------------------------------------------------------
def run_lif_simulation(V0, I_in, duration, dt, use_refractory, T_ref):
    """
    Runs LIF model and returns plots.
    """

    iterations = int(duration / dt)

    # instantiate model with given refractory period
    model = LIFModel(gl=0.16, Cm=0.0049, El=-65, thresh=-50, T_ref=T_ref)

    # solve
    V, spike_times = model.solve(
        V0=float(V0),
        dt=dt,
        iterations=iterations,
        I_ext=float(I_in),
        reset_to_El=True,
        use_refractory=use_refractory,
    )

    t = np.arange(0, iterations + 1) * dt

    # --------- Plot Voltage ---------
    fig_v, ax1 = plt.subplots(figsize=(6, 3))
    ax1.plot(t, V, color="black")
    ax1.set_title(
        f"Membrane Voltage Trace " +
        ("(Refractory ON)" if use_refractory else "(Refractory OFF)")
    )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Voltage (mV)")
    ax1.grid(True)

    # --------- Plot Current ---------
    fig_I, ax2 = plt.subplots(figsize=(6, 3))
    ax2.plot(t, np.ones_like(t) * I_in, color="blue")
    ax2.set_title("Input Current")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Current (a.u.)")
    ax2.grid(True)

    # --------- Spike Raster ---------
    fig_s, ax3 = plt.subplots(figsize=(6, 2))
    if len(spike_times) > 0:
        ax3.scatter(spike_times, np.ones_like(spike_times),
                    marker="|", s=200, color="red")
    ax3.set_ylim(0, 2)
    ax3.set_title("Spike Raster")
    ax3.set_xlabel("Time (s)")
    ax3.set_yticks([])
    ax3.grid(False)

    return fig_I, fig_v, fig_s



# -------------------------------------------------------------
# GRADIO APP
# -------------------------------------------------------------
app = gr.Interface(
    fn=run_lif_simulation,
    inputs=[
        gr.Number(value=-65, label="Initial Voltage V0 (mV)"),
        gr.Slider(-20, 20, value=5, step=1, label="Input Current (a.u.)"),
        gr.Slider(0.01, 1.0, value=0.1, step=0.01, label="Simulation Duration (s)"),
        gr.Slider(0.00001, 0.01, value=0.00002, step=0.00001, label="Time Step dt (s)"),

        gr.Checkbox(value=False, label="Enable Refractory Period"),
        gr.Slider(0.0, 0.05, value=0.002, step=0.001, label="Refractory Time T_ref (s)"),
    ],
    outputs=[
        gr.Plot(label="External Current"),
        gr.Plot(label="Voltage Trace"),
        gr.Plot(label="Spike Raster"),
    ],
    title="LIF Neuron Simulator",
    description="Simulates a Leaky Integrate-and-Fire neuron with adjustable input current and refractory period.",
)

def launch_lif_app():
    app.launch(share = False)

