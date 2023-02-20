# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
import matplotlib.pyplot as plt
from lava.utils.plots import raster_plot

np.random.seed(0)
_SPIKES = np.random.randint(2, size=(10, 20))


class TestInputValidation(unittest.TestCase):
    def test_return_figure_on_valid_input(self) -> None:
        fig = raster_plot(_SPIKES)
        self.assertIsInstance(fig, plt.FigureBase)

        fig = raster_plot(_SPIKES, fig=plt.figure())
        self.assertIsInstance(fig, plt.FigureBase)

        fig = raster_plot(_SPIKES, figsize=(10, 10))
        self.assertIsInstance(fig, plt.FigureBase)

    def test_bad_spikes_shape(self) -> None:
        spikes = np.array([0, 1, 2])

        with self.assertRaises(ValueError) as cm:
            raster_plot(spikes)

        self.assertEquals(
            str(cm.exception),
            "Parameter <spikes> must have exactly two dimensions and "
            "they must be non-empty."
        )

    def test_non_binary_values(self) -> None:
        error_msg = "All values in spikes must be either 0 or 1."

        spikes = np.array([[0, 2], [0, 0]])
        with self.assertRaises(ValueError) as cm:
            raster_plot(spikes)

        self.assertEquals(str(cm.exception), error_msg)

        spikes = np.array([[0, -1], [0, 0]])
        with self.assertRaises(ValueError) as cm:
            raster_plot(spikes)

        self.assertEquals(str(cm.exception), error_msg)

    def test_bad_stride(self) -> None:
        with self.assertRaises(ValueError) as cm:
            raster_plot(_SPIKES, stride=11)

        self.assertEquals(
            str(cm.exception),
            "Stride must not be greater than the number of neurons.",
        )

    def test_both_fig_and_figsize_provided(self) -> None:
        with self.assertRaises(ValueError) as cm:
            raster_plot(_SPIKES, fig=plt.figure(), figsize=(10, 10))

        self.assertEquals(
            str(cm.exception),
            "Must use at most one of the following: fig, figsize.",
        )
