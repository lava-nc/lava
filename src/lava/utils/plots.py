# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import FigureBase

_DEFAULT_FIGSIZE = (10, 5)


def raster_plot(
    spikes: np.ndarray,
    stride: int = 6,
    fig: ty.Optional[FigureBase] = None,
    figsize: ty.Optional[ty.Tuple[int, int]] = None,
    marker: str = "o",
    markersize: float = 1.5,
    color: ty.Any = "b",
    alpha: float = 1,
    xlabel: str = "Time Steps",
    ylabel: str = "Neurons",
) -> FigureBase:
    """Generate raster plot of spiking activity.

    Parameters
    ----------
    spikes : np.ndarray
        Spiking activity of neurons. Shape is (number of neurons, number of
        timestemps). spikes[i][j] represents the spiking activity of neuron i
        at time step j. 1 indicates a spike, 0 indicates no spike.
    stride : int
        Stride for plotting neurons. E.g. a stride of 6 means plot the spike
        train of every 6th neuron. Default is 6.
    fig: FigureBase, optional
        Active matplotlib figure to use. Passing None will create a new one.
        Cannot be used together with figsize.
    figsize: (float, float), optional
        Width, height in inches to use to create new figure. Cannot be used
        together with fig.
    marker: str
        The style of the markers representing the spikes. Default is 'o'.
    markersize: float
        The size of the markers representing the spikes. Default is 1.5.
    color: any
        Value specifying the color of the markers. See
        https://matplotlib.org/stable/tutorials/colors/colors.html for details.
    alpha: float
        Alpha value to use. Must be in between 0 and 1 (inclusive). Default is
        1.
    xlabel: str
        The label of the x axis. Default is 'Time Steps'.
    ylabel: str
        The label of the y axis. Default is 'Neurons'.
    """

    if len(spikes.shape) != 2 or 0 in spikes.shape:
        raise ValueError(
            "Parameter <spikes> must have exactly two dimensions and "
            "they must be non-empty."
        )

    if ((spikes != 0) & (spikes != 1)).any():
        raise ValueError("All values in spikes must be either 0 or 1.")

    num_neurons = spikes.shape[0]
    num_time_steps = spikes.shape[1]

    if stride > num_neurons:
        raise ValueError(
            "Stride must not be greater than the number of neurons."
        )

    if fig is not None and figsize is not None:
        raise ValueError("Must use at most one of the following: fig, "
                         "figsize.")

    time_steps = np.arange(0, num_time_steps, 1)

    if fig is None:
        if figsize is None:
            figsize = _DEFAULT_FIGSIZE
        fig = plt.figure(figsize=figsize)

    plt.xlim(-1, num_time_steps)
    plt.yticks([])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for i in range(0, num_neurons, stride):
        spike_times = time_steps[spikes[i] == 1]
        plt.plot(
            spike_times,
            i * np.ones(spike_times.shape),
            linestyle=" ",
            marker=marker,
            markersize=markersize,
            color=color,
            alpha=alpha,
        )

    return fig
