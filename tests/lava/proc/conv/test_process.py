# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
from lava.proc.conv.process import Conv


class TestConvProcess(unittest.TestCase):
    """Tests for Conv class"""
    def test_init(self) -> None:
        """Tests instantiation of Conv"""
        weight = np.random.randint(256, size=[8, 3, 5, 3]) - 128
        conv = Conv(
            weight=weight,
            input_shape=(100, 60, 3),
            padding=(1, 2),
            stride=1,
        )

        self.assertEqual(conv.output_shape, (100, 60, 8))
        self.assertEqual(conv.input_shape, (100, 60, 3))
        self.assertEqual(conv.weight.shape, weight.shape)
        self.assertEqual(conv.padding.init, (1, 2))
        self.assertEqual(conv.stride.init, (1, 1))
        self.assertEqual(conv.dilation.init, (1, 1))
        self.assertEqual(conv.groups.init, 1)
        self.assertTrue(np.abs(conv.weight.init - weight).sum() == 0)
