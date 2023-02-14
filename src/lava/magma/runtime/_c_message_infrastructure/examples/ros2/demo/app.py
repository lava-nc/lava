# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

#
# To run this in bokeh (with a web interface):
# bokeh serve app.py --port 18886
#
# Or run directly in command line, with openCV windows popping out:
# python3 app.py
#

import os
import subprocess  # nosec
import cv2

import numpy as np
from threading import Thread
from multiprocessing import Pipe
from functools import partial

from bokeh.plotting import figure, curdoc
from bokeh.layouts import gridplot, Spacer
from bokeh.models import LinearColorMapper, ColorBar, Title, Button
from bokeh.models.ranges import DataRange1d

from lava.proc.lif.process import LIF
from lava.magma.compiler.compiler import Compiler
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.runtime.message_infrastructure.message_interface_enum import (
    ActorType,
)
from lava.magma.runtime.runtime import Runtime
from lava.magma.core.run_conditions import RunSteps


from lava.lib.dnf.connect.connect import _configure_ops, _compute_weights
from lava.lib.dnf.kernels.kernels import MultiPeakKernel, SelectiveKernel
from lava.lib.dnf.operations.operations import Convolution
from sparse.process import Syn

from process_out.process import ProcessOut, DataRelayerPM
from ros_realsense_input.process import RosRealsenseInput, RosRealsenseInputPM
from rate_reader.process import RateReader


def through_bokeh():
    def get_pname():
        pname = ""
        with subprocess.Popen(
            ["ps -o cmd= {}".format(os.getpid())],
            stdout=subprocess.PIPE,
            shell=True,  # nosec
        ) as p:
            pname = str(p.communicate()[0])
        return pname

    return "bokeh" in get_pname()


running_in_bokeh = through_bokeh()


# ==========================================================================
# Parameters
# ==========================================================================
# number of time steps to be run in demo
num_steps = 4800 if running_in_bokeh else 3

# RosRealsenseInput Params
true_height = 480
true_width = 640
down_sample_factor = 16
diff_thresh = 250

down_sampled_shape = (
    true_width // down_sample_factor,
    true_height // down_sample_factor,
)

num_neurons = (true_height // down_sample_factor) * (
    true_width // down_sample_factor
)
down_sampled_flat_shape = (num_neurons,)

true_shape = (true_width, true_height)

# RateReader Params
buffer_size_rate_reader = 10

# Network Params
multipeak_params = {
    "amp_exc": 14,
    "width_exc": [5, 5],
    "amp_inh": -10,
    "width_inh": [10, 10],
}

selective_kernel_params = {"amp_exc": 7, "width_exc": [7, 7], "global_inh": -5}
sparse1_weights = 8
sparse2_weights = 20
dv_selective = 2047
du_selective = 809
selective_threshold = 30
du_multipeak = 2000
dv_multipeak = 2000
multipeak_threshold = 30

# MultiPeak DNF Params
kernel_multi_peak = MultiPeakKernel(**multipeak_params)
ops_multi_peak = [Convolution(kernel_multi_peak)]
_configure_ops(ops_multi_peak, down_sampled_shape, down_sampled_shape)
weights_multi_peak = _compute_weights(ops_multi_peak)

# Selective DNF Params
kernel_selective = SelectiveKernel(**selective_kernel_params)
ops_selective = [Convolution(kernel_selective)]
_configure_ops(ops_selective, down_sampled_shape, down_sampled_shape)
weights_selective = _compute_weights(ops_selective)

print("weights_multi_peak:", weights_multi_peak.shape)
print("weights_selective:", weights_selective.shape)

# ==========================================================================
# Instantiating Pipes
# ==========================================================================
recv_pipe, send_pipe = Pipe()

# ==========================================================================
# Instantiate Processes Running on CPU
# ==========================================================================
roscam_input = RosRealsenseInput(
    true_height=true_height,
    true_width=true_width,
    down_sample_factor=down_sample_factor,
    num_steps=num_steps,
    diff_thresh=diff_thresh,
)

rate_reader_multi_peak = RateReader(
    shape=down_sampled_shape,
    buffer_size=buffer_size_rate_reader,
    num_steps=num_steps,
)

rate_reader_selective = RateReader(
    shape=down_sampled_shape,
    buffer_size=buffer_size_rate_reader,
    num_steps=num_steps,
)

# sends data to pipe for plotting
data_relayer = ProcessOut(
    shape_roscam_frame=down_sampled_shape,
    shape_dnf=down_sampled_shape,
    send_pipe=send_pipe,
)

# ==========================================================================
# Instantiate Processes Running on Loihi 2
# ==========================================================================
sparse_1 = Syn(
    weights=np.eye(num_neurons) * sparse1_weights, synname="sparse_1"
)
dnf_multi_peak = LIF(
    shape=down_sampled_shape,
    du=du_multipeak,
    dv=dv_multipeak,
    vth=multipeak_threshold,
)
connections_multi_peak = Syn(
    weights=weights_multi_peak, synname="connections_multi_peak"
)
sparse_2 = Syn(
    weights=np.eye(num_neurons) * sparse2_weights, synname="sparse_2"
)
dnf_selective = LIF(
    shape=down_sampled_shape,
    du=du_selective,
    dv=dv_selective,
    vth=selective_threshold,
)
connections_selective = Syn(
    weights=weights_selective, synname="connections_selective"
)


# ==========================================================================
# Connecting Processes
# ==========================================================================
# Connecting Input Processes
roscam_input.event_frame_out.reshape(down_sampled_flat_shape).connect(
    sparse_1.s_in
)
sparse_1.a_out.reshape(new_shape=down_sampled_shape).connect(
    dnf_multi_peak.a_in
)
dnf_multi_peak.s_out.reshape(new_shape=down_sampled_flat_shape).connect(
    sparse_2.s_in
)  # * down_sampled_flat_shape: (num_neurons,)
sparse_2.a_out.reshape(new_shape=down_sampled_shape) \
    .connect(dnf_selective.a_in)

# Recurrent-connecting MultiPeak DNF
con_ip = connections_multi_peak.s_in
dnf_multi_peak.s_out.reshape(new_shape=con_ip.shape) \
    .connect(con_ip)
con_op = connections_multi_peak.a_out
con_op.reshape(new_shape=dnf_multi_peak.a_in.shape) \
    .connect(dnf_multi_peak.a_in)

# Recurrent-connecting Selective DNF
con_ip = connections_selective.s_in
dnf_selective.s_out.reshape(new_shape=con_ip.shape).connect(con_ip)
con_op = connections_selective.a_out
con_op.reshape(new_shape=dnf_selective.a_in.shape).connect(dnf_selective.a_in)

# Connect C Reader Processes
sparse_1.a_out.reshape(new_shape=down_sampled_shape).connect(
    rate_reader_multi_peak.in_port
)
dnf_selective.s_out.connect(rate_reader_selective.in_port)

# Connecting ProcessOut (data relayer)
roscam_input.event_frame_out.connect(data_relayer.roscam_frame_port)
rate_reader_multi_peak.out_port.connect(data_relayer.dnf_multipeak_rates_port)
rate_reader_selective.out_port.connect(data_relayer.dnf_selective_rates_port)

# ==========================================================================
# Runtime Creation and Compilation
# ==========================================================================

exception_pm_map = {
    RosRealsenseInput: RosRealsenseInputPM,
    ProcessOut: DataRelayerPM,
}
run_cfg = Loihi2SimCfg(exception_proc_model_map=exception_pm_map)
run_cnd = RunSteps(num_steps=num_steps, blocking=False)

# Compilation
compiler = Compiler()
executable = compiler.compile(roscam_input, run_cfg=run_cfg)

# Initializing runtime
mp = ActorType.MultiProcessing
runtime = Runtime(exe=executable, message_infrastructure_type=mp)
runtime.initialize()

# ==========================================================================
# Main block
# ==========================================================================
if running_in_bokeh:
    # ==========================================================================
    # Bokeh Helpers
    # ==========================================================================
    def callback_run():
        runtime.start(run_condition=run_cnd)

    def create_plot(plot_base_width, data_shape, title):
        x_range = DataRange1d(
            start=0,
            end=data_shape[0],
            bounds=(0, data_shape[0]),
            range_padding=50,
            range_padding_units="percent",
        )
        y_range = DataRange1d(
            start=0,
            end=data_shape[1],
            bounds=(0, data_shape[1]),
            range_padding=50,
            range_padding_units="percent",
        )

        pw = plot_base_width
        ph = int(pw * data_shape[1] / data_shape[0])
        plot = figure(
            width=pw,
            height=ph,
            x_range=x_range,
            y_range=y_range,
            match_aspect=True,
            tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],
            toolbar_location=None,
        )

        image = plot.image(
            [],
            x=0,
            y=0,
            dw=data_shape[0],
            dh=data_shape[1],
            palette="Viridis256",
            level="image",
        )

        plot.add_layout(Title(text=title, align="center"), "above")

        x_grid = list(range(data_shape[0]))
        plot.xgrid[0].ticker = x_grid
        y_grid = list(range(data_shape[1]))
        plot.ygrid[0].ticker = y_grid
        plot.xgrid.grid_line_color = None
        plot.ygrid.grid_line_color = None

        color = LinearColorMapper(palette="Viridis256", low=0, high=1)
        image.glyph.color_mapper = color

        cb = ColorBar(color_mapper=color)
        plot.add_layout(cb, "right")

        return plot, image

    # ==========================================================================
    # Instantiating Bokeh document
    # ==========================================================================
    bokeh_document = curdoc()

    # create plots
    roscam_frame_p, roscam_frame_im = create_plot(
        400, down_sampled_shape, "ROS input (events)"
    )
    dnf_multipeak_rates_p, dnf_multipeak_rates_im = create_plot(
        400, down_sampled_shape, "DNF multi-peak (spike rates)"
    )
    dnf_selective_rates_p, dnf_selective_rates_im = create_plot(
        400, down_sampled_shape, "DNF selective (spike rates)"
    )

    # add a button widget and configure with the call back
    button_run = Button(label="Run")
    button_run.on_click(callback_run)

    # finalize layout (with spacer as placeholder)
    spacer = Spacer(height=40)
    bokeh_document.add_root(
        gridplot(
            [
                [button_run, None, None],
                [None, spacer, None],
                [roscam_frame_p, dnf_multipeak_rates_p, dnf_selective_rates_p],
            ],
            toolbar_options=dict(logo=None),
        )
    )

    # ==========================================================================
    # Bokeh Update
    # ==========================================================================
    def update(
        roscam_frame_ds_image,
        dnf_multipeak_rates_ds_image,
        dnf_selective_rates_ds_image,
    ):
        roscam_frame_im.data_source.data["image"] = [roscam_frame_ds_image]
        dnf_multipeak_rates_im.data_source.data["image"] = [
            dnf_multipeak_rates_ds_image
        ]
        dnf_selective_rates_im.data_source.data["image"] = [
            dnf_selective_rates_ds_image
        ]

    # ==========================================================================
    # Bokeh Main loop
    # ==========================================================================
    def main_loop():
        while True:
            data_for_plot_dict = recv_pipe.recv()
            bokeh_document.add_next_tick_callback(
                partial(update, **data_for_plot_dict)
            )

    thread = Thread(target=main_loop)
    thread.start()
else:
    runtime.start(run_condition=run_cnd)

    def main_loop():
        for i in range(num_steps):
            print("step", i)
            data_for_plot_dict = recv_pipe.recv()
            for name, img_data in data_for_plot_dict.items():
                cv2.imshow(str(i) + "_" + name, img_data * 255)
                # * If you prefer saving the image..
                # cv2.imwrite(
                #   "~/img_" + str(i) + "_" + name + ".jpg",
                #   img_data * 255)

    thread = Thread(target=main_loop)
    thread.start()
    thread.join()
    runtime.stop()
