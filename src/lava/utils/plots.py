# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import matplotlib.pyplot as plt


def raster_plot(spks, stride=6, fig=None, color='b', alpha=1):
    """Generate raster plot of spiking activity.

    Parameters
    ----------
    spks : np.ndarray shape (num_neurons, num_timesteps)
        Spiking activity of neurons, a spike is indicated by a one
    stride : int
        Stride for plotting neurons
    """
    num_neurons = spks.shape[0]
    num_time_steps = spks.shape[1]

    if stride >= num_neurons:
        raise ValueError("Stride must be less than the number of neurons")

    time_steps = np.arange(0, num_time_steps, 1)
    if fig is None:
        fig = plt.figure(figsize=(10, 5))

    plt.xlim(-1, num_time_steps)
    plt.yticks([])

    plt.xlabel('Time steps')
    plt.ylabel('Neurons')

    for i in range(0, num_neurons, stride):
        spike_times = time_steps[spks[i] == 1]
        plt.plot(spike_times,
                 i * np.ones(spike_times.shape),
                 linestyle=' ',
                 marker='o',
                 markersize=1.5,
                 color=color,
                 alpha=alpha)

    return fig
