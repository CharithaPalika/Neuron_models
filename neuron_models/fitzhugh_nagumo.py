import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from tqdm.autonotebook import tqdm
from matplotlib import rcParams
import numpy as np
import plotly.graph_objects as go
import gradio as gr
from PIL import Image
import io


class FNModel():
    '''

    Class to simulate  FitzHugh-Nagumo neuron model

    Attributes:
      a (float): Parameter, Default = 0.5
      b (float): Parameter, Default = 0.1
      r (float): Parameter, Default = 0.1
      Iext (float): External current, Default = 0
    '''
    def __init__(self, a = 0.5, b = 0.1 , r = 0.1, Iext = 0):

        self.a = a
        self.b = b
        self.r = r
        self.Iext = Iext
        self.v_timemon = []
        self.w_timemon = []
        self.dw_mon = []
        self.dv_mon = []

    def dv_dt(self, v, w):
        '''
        Function to compute dv/dt

        Args:
          v (float): current time step voltage value
          w (float): current time step w value

        Returns:
          (float): derivative wrt to v

        '''
        fv = v * (self.a - v) * (v - 1)
        return (fv - w + self.Iext)

    def dw_dt(self, v, w):
        '''
        Function to compute dw/dt

        Args:
          v (float): current time step voltage value
          w (float): current time step w value

        Returns:
          (float): derivative wrt to w

        '''
        return (self.b * v - self.r * w)

    def v_nullcline(self, v):
        '''
        Function to compute values of v null_cline

        Args:
          v (np.ndarray): voltage values

        Returns:
          w (np.ndarray): w values corresponding to v

        '''
        fv = v * (self.a - v) * (v - 1)
        w = fv + self.Iext
        return w

    def w_nullcline(self, v):
        '''
        Function to compute values of w null_cline

        Args:
          v (np.ndarray): voltage values

        Returns:
          w (np.ndarray): w values corresponding to v

        '''
        w = (self.b * v)/ self.r
        return w

    def solve(self, v0, w0, dt, iter):
        '''
        Function to solve ODE using Euler method

        Args:
          v0 (float): initial voltage
          w0 (float): initial w
          dt (float): Euler step
          iter (int): Number of iterations

        Returns:
          None

        '''
        v = v0
        w = w0
        self.v_timemon = []
        self.w_timemon = []
        self.dv_mon = []
        self.dw_mon = []
        for i in range(iter):
            dv = self.dv_dt(v, w)
            dw = self.dw_dt(v, w)
            self.dv_mon.append(dv)
            self.dw_mon.append(dw)
            v = v + dt * dv
            w = w + dt * dw
            self.v_timemon.append(v)
            self.w_timemon.append(w)

        return self.v_timemon, self.w_timemon, self.dv_mon, self.dw_mon

    def plotnullclines(self, v):
      '''
        Function to Plot v and w nullclines

        Args:
          v (np.ndarray): voltage values

        Returns:
          None

        '''

      #v = np.linspace(-0.05,1,100)
      v_null = self.v_nullcline(v)
      w_null = self.w_nullcline(v)
      plt.axvline(x=0, c="black", lw = 0.3)
      plt.axhline(y=0, c="black", lw = 0.3)
      plt.plot(v, v_null, label = 'v nullcline')
      plt.plot(v, w_null, label = 'w nullcline')
      plt.legend(fontsize = 8)
      plt.title('Nullclines for Iext = '+ (str(self.Iext)) ,fontsize = 10)
      plt.xlabel('v', fontsize = 12)
      plt.ylabel('w', fontsize = 12)

    def phaseplot(self, v0, w0, startpoints, density):
      '''
        Function to plot phase portrait superimposed on nullclines

        Args:
          v_range (np.ndarray): Array of voltage values to plot nullclines
          v0 (np.ndarray): Array of voltage values for phase plot
          w0 (np.ndarray): Array of w values for phase plot
          startpoints (list): Either None or list indicating the start points
          density (float): Phase portrait density

        Returns:
          None

        '''
      # Plotting trajectories
      V, W = np.meshgrid(v0, w0)
      dv = self.dv_dt(V,W)
      dw = self.dw_dt(V,W)
      plt.streamplot(V, W, dv, dw,
                     density = density,
                     maxlength = 10,
                     linewidth = 0.5,
                     start_points= startpoints,
                     integration_direction = 'forward',
                     broken_streamlines = False)





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
def run_fn_simulation(a, b, r, v0, w0, Iext, duration, dt, show_plots, show_nullclines):

    iter_count = int(duration / dt)

    # Create model with selected params
    model = FNModel(a=a, b=b, r=r, Iext=Iext)

    # Solve ODE
    v_time, w_time, dv, dw = model.solve(
        v0=v0, w0=w0, dt=dt, iter=iter_count
    )
    t = np.arange(0, duration, dt)

    # --------------------------------------------------------
    # (1) Time plot (Plotly)
    # --------------------------------------------------------
    fig_v = go.Figure()
    if "Time" in show_plots:
        fig_v.add_trace(go.Scatter(
            x=t, y=v_time, mode='lines',
            name="v(t)", line=dict(color="firebrick")
        ))
        fig_v.add_trace(go.Scatter(
            x=t, y=w_time, mode='lines',
            name="w(t)", line=dict(color="royalblue")
        ))

    fig_v.update_layout(
        title="v(t) and w(t)",
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
            x=v_time, y=w_time,
            mode="lines",
            name="Trajectory", line=dict(color="green")
        ))
        fig_traj.add_trace(go.Scatter(
            x=[v_time[0]], y=[w_time[0]],
            mode="markers",
            marker=dict(size=6, color="red"),
            name="Start"
        ))

    fig_traj.update_layout(
        title="Trajectory in Phase Space",
        xaxis_title="v",
        yaxis_title="w",
        template="plotly_white",
        height=400
    )

    # --------------------------------------------------------
    # (3) Phase plane (Matplotlib streamplot)
    # --------------------------------------------------------
    if "Phase plane" in show_plots:

        fig, ax = plt.subplots(figsize=(5, 5))

        # Grid for streamplot
        v_vals = np.linspace(-0.5, 1.5, 30)
        w_vals = np.linspace(-0.5, 1.5, 30)
        V, W = np.meshgrid(v_vals, w_vals)

        dV = model.dv_dt(V, W)
        dW = model.dw_dt(V, W)

        ax.streamplot(V, W, dV, dW,
                      density=0.3,
                      color="gray",
                      linewidth=0.8,
                      arrowsize=1,
                      maxlength=10,
                    integration_direction = 'forward',
                     broken_streamlines = False)

        # Optional nullclines
        if show_nullclines:
            v_nc = np.linspace(-0.5, 1.5, 200)
            ax.plot(v_nc, model.v_nullcline(v_nc),
                    label="v-nullcline", color="red")
            ax.plot(v_nc, model.w_nullcline(v_nc),
                    label="w-nullcline", color="blue")
            ax.legend(fontsize=8)

        # Trajectory overlay
        ax.plot(v_time, w_time,
                color="green", linewidth=1.2,
                label="Trajectory")

        ax.set_title("Phase Plane (2D Streamplot)")
        ax.set_xlabel("v")
        ax.set_ylabel("w")

        phase_plane_img = fig_to_image(fig)
        plt.close(fig)
    else:
        phase_plane_img = None

    return fig_v, fig_traj, phase_plane_img


# ---------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------
with gr.Blocks(title="FitzHugh–Nagumo Neuron Simulator") as demo:

    gr.Markdown("## ⚡ FitzHugh–Nagumo Neuron Simulator")

    # Model parameters
    with gr.Row():
        a = gr.Number(value=0.5, label="a")
        b = gr.Number(value=0.1, label="b")
        r = gr.Number(value=0.1, label="r")

    # Initial conditions
    with gr.Row():
        v0 = gr.Number(value=0.3, label="Initial v₀")
        w0 = gr.Number(value=0.0, label="Initial w₀")

    # Simulation controls
    with gr.Row():
        Iext = gr.Slider(0, 1, value=0.5, step=0.01, label="External Current Iext")
        duration = gr.Slider(100, 5000, value=1000, step=100, label="Duration (ms)")
        dt = gr.Slider(0.001, 0.1, value=0.01, step=0.001, label="Time Step (ms)")

    # Plot selections
    show_plots = gr.CheckboxGroup(
        ["Time", "Trajectory", "Phase plane"],
        value=["Time"],
        label="Select plots to display"
    )

    # Toggle nullclines
    show_nullclines = gr.Checkbox(
        label="Show nullclines in Phase Plane",
        value=True
    )

    run_button = gr.Button("Run Simulation 🚀")

    # Outputs
    with gr.Row():
        output_time = gr.Plot(label="v(t), w(t)")
    with gr.Row():
        output_traj = gr.Plot(label="Trajectory v–w")
    with gr.Row():
        output_phase = gr.Image(type="pil", label="Phase Plane")

    run_button.click(
        fn=run_fn_simulation,
        inputs=[a, b, r, v0, w0, Iext, duration, dt, show_plots, show_nullclines],
        outputs=[output_time, output_traj, output_phase],
    )



def launch_FN_app():
  demo.launch(share = False)
