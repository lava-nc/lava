# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import matplotlib.pyplot as plt
import numpy as np

from IPython.display import display, clear_output
from typing import List, Tuple

from lava.proc.io.extractor import Extractor


class Figure:
    def __init__(self,
                 plots: List) -> None:
        self.plots = plots
        self.exists = False
        self.closed = False
        self.use_ipython = False
        try:
            get_ipython().__class__.__name__
            self.use_ipython = True
        except NameError:
            pass


    def show(self):
        if not self.exists:
            self.create_figure()
        while not self.closed:
            self.update()

    def close(self, evt = None):
        if not self.closed and self.fig.get_visible():
            plt.close()
        self.closed = True

    def create_figure(self):
        if not self.use_ipython:
            plt.ion()
        self.fig = plt.figure(figsize=(15, 5), dpi=90)
        for plot in self.plots:
            plot.create_subplot(self.fig)
        if not self.use_ipython:
            self.fig.canvas.mpl_connect('close_event', self.close)
        self.fig.tight_layout()
        plt.show(block=False)
        self.exists = True
        plt.pause(0.1)

    def update(self):
        for plot in self.plots:
            plot.update()
        if self.use_ipython:
            clear_output(wait=True)
            display(self.fig)
        else:
            plt.pause(0.01)


class AbstractPlot:
    def __init__(self, subplot: int):
        self.fig = None
        self.ax = None
        self.subplot = subplot
        self.exists = False

    def create_subplot(self, fig: plt.Figure):
        self.fig = fig
        self.ax = self.fig.add_subplot(self.subplot)
        self.draw()
        self.exists = True

    def update(self):
        if self.receive_data():
            self.draw()

    def draw(self):
        raise NotImplemented()

    def receive_data(self):
        raise NotImplemented()


class Raster(AbstractPlot):
    """StreamingRasterPlot visualizes streaming spike data.
    
    Parameters
    ----------
    length: int, default = 1000
        The number of timesteps that will be visualized.
    """
    def __init__(self,
                 shape: Tuple[int, ...],
                 length: int = 1000,
                 subplot: int = 111):
        super().__init__(subplot=subplot)
        self.length = length
        self.extractor = Extractor(shape=shape)
        self.spk_in = self.extractor.in_port
        data_shape = shape + (length,)
        self.data = np.zeros(shape=data_shape)
        self.timestep = 0

    def draw(self):
        if np.count_nonzero(self.data) > 0:
            y, x = np.where(self.data)
        else:
            y, x = [], []
        self.ax.clear()
        self.ax.scatter(x, y, c='k', marker='|')
        self.ax.set_xlabel('Timestep')
        self.ax.set_ylabel('Neuron Idx')
        self.ax.set_xlim(0, self.length)
        self.ax.set_ylim(-0.5, self.spk_in.shape[0] - 0.5)

    def receive_data(self) -> int:
        received = 0
        while self.extractor.can_receive():
            t_data = self.extractor.receive()
            t = (self.timestep - 1) % self.length
            self.timestep += 1
            self.data[..., t] = t_data
            received += 1
        return received


class ImageView(AbstractPlot):
    """ImageView visualizes streaming images."""

    def __init__(self,
                 shape: Tuple,
                 bias: float,
                 range: float,
                 transpose: List = [0, 1, 2],
                 subplot: int = 111) -> None:
        super().__init__(subplot=subplot)
        self.extractor = Extractor(shape)
        self.img_in = self.extractor.in_port
        self.data = np.zeros(shape=shape)
        self.bias = bias
        self.range = range
        self.transpose = transpose
        self.transform_data()

    def draw(self):
        self.ax.clear()
        self.ax.imshow(self.img, aspect='auto', interpolation='nearest',
                       origin='upper')
        self.ax.set_xticks([])
        self.ax.set_yticks([])

    def receive_data(self) -> int:
        received = 0
        while self.extractor.can_receive():
            np.copyto(self.data, self.extractor.receive())
            received += 1
        if received > 0:
            self.transform_data()
        return received

    def transform_data(self):
        self.img = self.data.transpose(self.transpose)
        self.img -= self.bias
        self.img /= 2 * self.range
        self.img += 0.5


class LinePlot(AbstractPlot):
    """LinePlot visualizes streaming continuous data."""

    def __init__(self,
                 length: int,
                 min: float,
                 max: float,
                 num_lines: int = 1,
                 subplot: int = 111):
        super().__init__(subplot=subplot)
        self.extractors = list([Extractor(shape=(1,))
                                for _ in range(num_lines)])
        self.y_in = list([ext.in_port for ext in self.extractors])
        self.length = length
        self.min = min
        self.max = max
        self.x = np.zeros(shape=(1, length))
        self.y = np.zeros(shape=(num_lines, length))

    def draw(self):
        self.ax.clear()
        self.ax.plot(self.x.T, self.y.T)
        #self.ax.set_xlim(0, self.length)
        self.ax.set_ylim(self.min, self.max)
        self.ax.set_xlabel('Timestep')
        self.ax.set_ylabel('Value')

    def receive_data(self) -> int:
        received = 0
        while all([ext.can_receive() for ext in self.extractors]):
            self.x = np.roll(self.x, shift=-1, axis=1)
            self.x[0,-1] = self.x[0,-2] + 1
            self.y = np.roll(self.y, shift=-1, axis=1)
            vals = [ext.receive() for ext in self.extractors]
            self.y[:,-1] = np.array(vals).flatten()
            received += 1
        return received
