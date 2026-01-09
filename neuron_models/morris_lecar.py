import numpy as np
from matplotlib import pyplot as plt
import gradio as gr
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
import io
from matplotlib import rcParams
rcParams["font.size"] = 12
rcParams["axes.spines.top"] =  False
rcParams["axes.spines.right"] = False

class MorrisLecarModel:
    def __init__(self, I_ext):
        self.gL = 2
        self.VL = -50
        self.gCa = 4
        self.VCa = 100
        self.gK = 8
        self.VK = -70
        self.C = 20
        self.I_ext = I_ext
        self.lambda_m = 1.0    # lambda M
        self.lambda_n = 0.1    # lambda N

        self.V1 = 0
        self.V2 = 15
        self.V3 = 10
        self.V4 = 10

        self.v_timemon = []
        self.n_timemon = []
        self.dv_mon = []
        self.dn_mon = []

    def m_inf(self, V):
        return 0.5 * (1 + np.tanh((V - self.V1) / self.V2))

    def n_inf(self, V):
        return 0.5 * (1 + np.tanh((V - self.V3) / self.V4))

    def tau_n(self, V):
        return 1 / np.cosh((V - self.V3) / (2 * self.V4))
    
    def dn_dt(self, V, n):
        return self.lambda_n * (self.n_inf(V) - n) / self.tau_n(V)
    
    def dv_dt(self, v , n):
        m = self.m_inf(v)
        I_L = self.gL * (v - self.VL)
        I_Ca = self.gCa * m * (v - self.VCa)
        I_K = self.gK * n * (v - self.VK)
        return self.lambda_m * ( -I_L - I_Ca - I_K + self.I_ext) / self.C
    
    def solve(self, v0, n0, dt, iter):
        v = v0
        n = n0
        self.v_timemon = []
        self.n_timemon = []
        self.dv_mon = []
        self.dn_mon = []

        for i in range(iter):
            dv = self.dv_dt(v, n)
            dn = self.dn_dt(v, n)
            v = v + dv * dt
            n = n + dn * dt
            self.v_timemon.append(v)
            self.n_timemon.append(n)
            self.dv_mon.append(dv)
            self.dn_mon.append(dn)
        return self.v_timemon, self.n_timemon, self.dv_mon, self.dn_mon
    
    def v_nullcline(self,v):
        pass

    def n_nullcline(self,n):
        pass

    def plot_nullclines(self):
        pass
    
    def phaseplot(self, v0, n0, startpoints, density):
        V, N = np.meshgrid(v0, n0)
        dv = self.dv_dt(V, N)
        dn = self.dn_dt(V, N)
        plt.streamplot(V,N,dv, dn , density=density, start_points=startpoints, 
                       integration_direction='forward', broken_streamlines=False, maxlength=20, linewidth=0.5)
                    #    maxlength=100)
        # plt.xlim(-25, 25)
        # plt.ylim(-25,25)




# ---------------------------------------------------------
# Convert Matplotlib figure → PIL image for Gradio
# ---------------------------------------------------------
def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)


# ---------------------------------------------------------
# Simulation and Plot Function
# ---------------------------------------------------------
def run_morrislecar_simulation(Iext, v0, n0, duration, dt, show_plots):

    iter_count = int(duration / dt)

    # Create model with selected external current
    model = MorrisLecarModel(I_ext=Iext)

    # Solve ODE
    v_time, n_time, dv, dn = model.solve(
        v0=v0, n0=n0, dt=dt, iter=iter_count
    )
    t = np.arange(0, duration, dt)

    # --------------------------------------------------------
    # (1) Time plot (Plotly)
    # --------------------------------------------------------
    fig_v = go.Figure()
    if "Time" in show_plots:
        fig_v.add_trace(go.Scatter(
            x=t, y=v_time, mode='lines',
            name="V(t)", line=dict(color="firebrick")
        ))
        fig_v.add_trace(go.Scatter(
            x=t, y=n_time, mode='lines',
            name="n(t)", line=dict(color="royalblue")
        ))

    fig_v.update_layout(
        title="V(t) and n(t)",
        xaxis_title="Time (ms)",
        yaxis_title="Values",
        template="plotly_white",
        height=400
    )

    # --------------------------------------------------------
    # (2) Trajectory (Plotly)
    # --------------------------------------------------------
    fig_traj = go.Figure()
    if "Trajectory" in show_plots:
        fig_traj.add_trace(go.Scatter(
            x=v_time, y=n_time,
            mode="lines",
            name="Trajectory", line=dict(color="green")
        ))
        fig_traj.add_trace(go.Scatter(
            x=[v_time[0]], y=[n_time[0]],
            mode="markers",
            marker=dict(size=6, color="red"),
            name="Start"
        ))

    fig_traj.update_layout(
        title="Trajectory in Phase Space",
        xaxis_title="V",
        yaxis_title="n",
        template="plotly_white",
        height=400
    )

    # --------------------------------------------------------
    # (3) Phase plane (Matplotlib streamplot, no nullclines)
    # --------------------------------------------------------
    if "Phase plane" in show_plots:
        fig, ax = plt.subplots(figsize=(5, 5))

        # Choose a reasonable grid for Morris–Lecar
        V_vals = np.linspace(-80, 60, 30)
        N_vals = np.linspace(0, 1, 30)
        V, N = np.meshgrid(V_vals, N_vals)

        dV = model.dv_dt(V, N)
        dN = model.dn_dt(V, N)

        ax.streamplot(V, N, dV, dN,
                      density=0.6,
                      color="gray",
                      linewidth=0.8,
                      arrowsize=1)

        # Trajectory overlay
        ax.plot(v_time, n_time,
                color="green", linewidth=1.2,
                label="Trajectory")

        ax.set_title("Phase Plane (V–n)")
        ax.set_xlabel("V")
        ax.set_ylabel("n")

        phase_plane_img = fig_to_image(fig)
        plt.close(fig)
    else:
        phase_plane_img = None

    return fig_v, fig_traj, phase_plane_img


# ---------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------
with gr.Blocks(title="Morris–Lecar Neuron Simulator") as demo:

    gr.Markdown("## ⚡ Morris–Lecar Neuron Simulator")

    # Initial conditions & current
    with gr.Row():
        Iext = gr.Slider(-50, 200, value=80, step=1, label="External Current Iext")
        v0 = gr.Number(value=-60.0, label="Initial V₀ (mV)")
        n0 = gr.Number(value=0.0, label="Initial n₀")

    # Simulation controls
    with gr.Row():
        duration = gr.Slider(50, 2000, value=500, step=10, label="Duration (ms)")
        dt = gr.Slider(0.01, 1.0, value=0.1, step=0.01, label="Time Step (ms)")

    # Plot selections
    show_plots = gr.CheckboxGroup(
        ["Time", "Trajectory", "Phase plane"],
        value=["Time"],
        label="Select plots to display"
    )

    run_button = gr.Button("Run Simulation 🚀")

    # Outputs
    with gr.Row():
        output_time = gr.Plot(label="V(t), n(t)")
    with gr.Row():
        output_traj = gr.Plot(label="Trajectory V–n")
    with gr.Row():
        output_phase = gr.Image(type="pil", label="Phase Plane")

    run_button.click(
        fn=run_morrislecar_simulation,
        inputs=[Iext, v0, n0, duration, dt, show_plots],
        outputs=[output_time, output_traj, output_phase],
    )


# Launch the app
def launch_morrislecar_app():
    demo.launch(share=False)
