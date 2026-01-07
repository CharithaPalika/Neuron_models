import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from tqdm.autonotebook import tqdm
from matplotlib import rcParams
import numpy as np
import plotly.graph_objects as go
import gradio as gr
rcParams["font.family"] = "Times New Roman"
rcParams["font.size"] = 14
rcParams["axes.spines.top"] =  False
rcParams["axes.spines.right"] = False


neuron_models = {
    "RS":  {"a": 0.02,  "b": 0.2,  "c": -65, "d": 8},    # Regular Spiking
    "IB":  {"a": 0.02,  "b": 0.2,  "c": -55, "d": 4},    # Intrinsically Bursting
    "CH":  {"a": 0.02,  "b": 0.2,  "c": -50, "d": 2},    # Chattering
    "FS":  {"a": 0.1,   "b": 0.2,  "c": -65, "d": 2},    # Fast Spiking
    "LTS": {"a": 0.02,  "b": 0.25, "c": -65, "d": 2},    # Low-Threshold Spiking
    "TC":  {"a": 0.02,  "b": 0.25, "c": -65, "d": 0.05}, # Thalamocortical
    "RZ":  {"a": 0.1,   "b": 0.26, "c": -65, "d": 2},    # Resonator
    "PH":  {"a": 0.02,  "b": 0.25, "c": -65, "d": 6}     # Phasic Spiking
}



class IzhikevichModel:
    def __init__(self, I, params):
        """
        I: 1D numpy array or list of external current values (length = number of timesteps)
        params: dict with keys "a","b","c","d"
        """
        self.I = np.array(I, dtype=float)
        self.a = params["a"]
        self.b = params["b"]
        self.c = params["c"]
        self.d = params["d"]

    def dv_dt(self, v, u, I_val):
        """
        dv/dt for Izhikevich model (right-hand-side)
        Equivalent to: 0.04*v**2 + 5*v + 140 - u + I
        """
        return 0.04 * v ** 2 + 5 * v + 140 - u + I_val

    def du_dt(self, v, u):
        """
        du/dt for Izhikevich model
        Equivalent to: a*(b*v - u)
        """
        return self.a * (self.b * v - u)

    def solve(self, v0, u0, dt, iter, t0=0.0):
        """
        Integrate the Izhikevich equations with simple Euler forward method,
        including the spike reset rule v >= 30 mV -> v = c, u += d.

        v0, u0: initial conditions (scalars)
        dt: timestep (ms)
        iter: number of timesteps to run
        t0: initial time (default 0.0)

        Returns:
            time: numpy array of times (length iter)
            v_trace: numpy array of v over time (length iter)
            u_trace: numpy array of u over time (length iter)
            spikes: list of times when spikes occurred (times where v reached threshold)
        """
        # Ensure I has correct length; if shorter, pad with zeros; if longer, truncate
        if len(self.I) < iter:
            I_used = np.concatenate([self.I, np.zeros(iter - len(self.I))])
        else:
            I_used = self.I[:iter]

        v = float(v0)
        u = float(u0)
        v_trace = np.empty(iter, dtype=float)
        u_trace = np.empty(iter, dtype=float)
        spikes = []

        t = t0
        for j in range(iter):
            I_val = I_used[j]

            # Spike check BEFORE integrating next step (consistent with your original code)
            if v >= 30.0:
                spikes.append(t)
                v = self.c
                u += self.d

            # Euler integration
            dv = self.dv_dt(v, u, I_val)
            v = v + dv * dt
            du = self.du_dt(v, u)
            u = u + du * dt

            v_trace[j] = v
            u_trace[j] = u
            t += dt

        time = np.arange(t0, t0 + iter * dt, dt)
        return time, v_trace, u_trace, spikes


def build_current(
    duration, dt,
    current_mode,
    I_const,
    I_step_low, I_step_high, step_t_on, step_t_off,
    pulse_freq, pulse_width, pulse_amp,
    pois_rate, pois_amp, pois_width,
    sin_freq, sin_amp, sin_offset,
):
    """
    duration, dt in ms.
    Returns time (ms) and I_ext array.
    """
    iters = int(duration / dt)
    t = np.arange(0, duration, dt)
    I_ext = np.zeros(iters)

    if current_mode == "Constant":
        I_ext[:] = I_const

    elif current_mode == "Step":
        I_ext[:] = I_step_low
        on_idx = int(step_t_on / dt)
        off_idx = int(step_t_off / dt)
        on_idx = max(0, min(on_idx, iters))
        off_idx = max(on_idx, min(off_idx, iters))
        I_ext[on_idx:off_idx] = I_step_high

    elif current_mode == "Pulsatile":
        # frequency in Hz -> period in ms
        if pulse_freq > 0:
            pulse_period = 1000.0 / pulse_freq
        else:
            pulse_period = duration + dt  # effectively "no pulses"
        for i, ti in enumerate(t):
            if ti >= 0:  # can add a 'start time' if you want
                phase = ti % pulse_period
                if phase < pulse_width:
                    I_ext[i] = pulse_amp
                else:
                    I_ext[i] = 0.0
            else:
                I_ext[i] = 0.0

    elif current_mode == "Poisson":
        # rate in Hz, dt in ms
        p_spike = pois_rate * dt / 1000.0
        width_steps = max(1, int(pois_width / dt))
        for i in range(iters):
            if np.random.rand() < p_spike:
                j_end = min(iters, i + width_steps)
                I_ext[i:j_end] += pois_amp

    elif current_mode == "Sinusoidal":
        # sin_freq in Hz, t in ms -> convert to seconds inside sin
        I_ext = sin_amp * np.sin(2 * np.pi * sin_freq * (t / 1000.0)) + sin_offset

    return t, I_ext


# -------------------------------------------------------------------
# Simulation execution
# -------------------------------------------------------------------
def run_izhikevich_simulation(
    v0, u0,
    neuron_type,
    duration,
    dt,
    current_mode,
    I_const,
    I_step_low,
    I_step_high,
    step_t_on,
    step_t_off,
    pulse_freq,
    pulse_width,
    pulse_amp,
    pois_rate,
    pois_amp,
    pois_width,
    sin_freq,
    sin_amp,
    sin_offset,
):

    # --- Build current waveform ---
    time, I_ext = build_current(
        duration, dt,
        current_mode,
        I_const,
        I_step_low, I_step_high, step_t_on, step_t_off,
        pulse_freq, pulse_width, pulse_amp,
        pois_rate, pois_amp, pois_width,
        sin_freq, sin_amp, sin_offset,
    )
    iters = len(time)

    # --- Choose neuron params (exclude PH in UI, but dict can still contain it) ---
    params = neuron_models[neuron_type]
    neuron = IzhikevichModel(I=I_ext, params=params)

    # Initial conditions from user
    v0 = float(v0)
    u0 = float(u0)

    # --- Solve model ---
    t_sim, v_time, u_time, spike_times = neuron.solve(
        v0=v0, u0=u0, dt=dt, iter=iters
    )

    # --- Plot 1: External current ---
    fig_I = go.Figure()
    fig_I.add_trace(go.Scatter(
        x=t_sim, y=I_ext,
        mode='lines', name='I_ext'
    ))
    fig_I.update_layout(
        title="External Current I(t)",
        xaxis_title="Time (ms)",
        yaxis_title="Current (a.u.)",
        template="plotly_white",
        height=300,
    )

    # --- Plot 2: Membrane voltage ---
    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(
        x=t_sim, y=v_time,
        mode='lines', name='v(t)', line=dict(color='black')
    ))
    fig_v.update_layout(
        title=f"Membrane Voltage v(t) — {neuron_type}",
        xaxis_title="Time (ms)",
        yaxis_title="Voltage (mV)",
        template="plotly_white",
        height=300,
    )

    # --- Plot 3: Spike raster with vertical bars ---
    fig_spikes = go.Figure()

    if len(spike_times) > 0:
        # Plot spikes as big vertical markers (squares) so they are clearly visible
        fig_spikes.add_trace(go.Scatter(
            x=spike_times,
            y=[1.0] * len(spike_times),
            mode='markers',
            marker=dict(
                symbol='line-ns-open',   # very safe & visible marker
                size=10,
                color='black'
            ),
            showlegend=False
        ))
    else:
        # If no spikes, add a small text annotation so it's clear
        fig_spikes.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            text="No spikes for these parameters",
            showarrow=False,
            font=dict(size=14)
        )

    fig_spikes.update_layout(
        title="Spike Times",
        xaxis_title="Time (ms)",
        yaxis=dict(showticklabels=False, range=[0, 2]),
        template="plotly_white",
        height=250,
    )
    return fig_I, fig_v, fig_spikes


# -------------------------------------------------------------------
# Gradio interface (simple, clean)
# -------------------------------------------------------------------
# Filter out PH from neuron types
neuron_choices = [k for k in neuron_models.keys() if k != "PH"]

inputs = [
    # Initial conditions
    gr.Number(value=-65.0, label="Initial v0 (mV)"),
    gr.Number(value=0.1, label="Initial u0"),

    # Neuron + time
    gr.Dropdown(neuron_choices, value="RS", label="Neuron Type"),
    gr.Slider(100, 2000, value=1000, step=50, label="Duration (ms)"),
    gr.Slider(0.01, 1, value=0.1, step=0.01, label="Time Step (ms)"),

    # Current mode
    gr.Radio(
        ["Constant", "Step", "Pulsatile", "Poisson", "Sinusoidal"],
        value="Constant",
        label="Current Type"
    ),

    # Constant current
    gr.Slider(-5, 20, value=10, step=0.5, label="Constant I"),

    # Step current
    gr.Slider(-20, 20, value=0, step=0.5, label="Step: I_low"),
    gr.Slider(-20, 20, value=10, step=0.5, label="Step: I_high"),
    gr.Slider(0, 1000, value=100, step=10, label="Step: t_on (ms)"),
    gr.Slider(0, 1000, value=600, step=10, label="Step: t_off (ms)"),

    # Pulsatile (frequency-based)
    gr.Slider(0.1, 10, value=3, step=0.1, label="Pulsatile: frequency (Hz)"),
    gr.Slider(1, 500, value=100, step=1, label="Pulsatile: pulse width (ms)"),
    gr.Slider(0, 20, value=10, step=0.5, label="Pulsatile: amplitude"),

    # Poisson
    gr.Slider(0, 200, value=20, step=1, label="Poisson: rate (Hz)"),
    gr.Slider(0, 20, value=1, step=0.1, label="Poisson: amplitude"),
    gr.Slider(1, 50, value=5, step=1, label="Poisson: pulse width (ms)"),

    # Sinusoidal
    gr.Slider(0.1, 200, value=10, step=0.1, label="Sinusoidal: frequency (Hz)"),
    gr.Slider(0, 20, value=5, step=0.5, label="Sinusoidal: amplitude"),
    gr.Slider(-20, 20, value=1, step=0.5, label="Sinusoidal: DC offset"),
]

outputs = [
    gr.Plot(label="External Current"),
    gr.Plot(label="Voltage Trace"),
    gr.Plot(label="Spike Raster"),
]

app = gr.Interface(
    fn=run_izhikevich_simulation,
    inputs=inputs,
    outputs=outputs,
    title="Izhikevich Neuron Simulator",
    description="Simulate RS, FS, IB, CH, LTS, TC, RZ neuron firing under different current inputs.",
    flagging_mode="never", 

)

def launch_izhikevich_app():
    app.launch(share=False)
