# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.proc.max_pooling.process import MaxPooling


class TestProcessMaxPooling(unittest.TestCase):
    def test_init(self):
        """Tests instantiation of DownSamplingDense."""
        max_pooling = MaxPooling(shape_in=(240, 180, 1),
                                 kernel_size=2)

        self.assertIsInstance(max_pooling, MaxPooling)
        self.assertEqual(max_pooling.kernel_size.init, (2, 2))
        self.assertEqual(max_pooling.stride.init, (2, 2))
        self.assertEqual(max_pooling.padding.init, (0, 0))

    def test_invalid_shape_in_throws_exception(self):
        with self.assertRaises(ValueError):
            MaxPooling(shape_in=(240, 180, 1, 1),
                       kernel_size=2)

    def test_invalid_kernel_size_throws_exception(self):
        with self.assertRaises(ValueError):
            MaxPooling(shape_in=(240, 180, 1),
                       kernel_size=(2, 2, 1))

    def test_invalid_stride_throws_exception(self):
        with self.assertRaises(ValueError):
            MaxPooling(shape_in=(240, 180, 1),
                       kernel_size=2,
                       stride=(2, 2, 1))

    def test_invalid_padding_throws_exception(self):
        with self.assertRaises(ValueError):
            MaxPooling(shape_in=(240, 180, 1),
                       kernel_size=2,
                       padding=(0, 0, 0))

    def test_negative_shape_in_element_throws_exception(self):
        with self.assertRaises(ValueError):
            MaxPooling(shape_in=(-240, 180, 1),
                       kernel_size=2)

    def test_negative_kernel_size_throws_exception(self):
        with self.assertRaises(ValueError):
            MaxPooling(shape_in=(240, 180, 1),
                       kernel_size=-1)

    def test_negative_stride_throws_exception(self):
        with self.assertRaises(ValueError):
            MaxPooling(shape_in=(240, 180, 1),
                       kernel_size=2,
                       stride=-1)

    def test_negative_padding_throws_exception(self):
        with self.assertRaises(ValueError):
            MaxPooling(shape_in=(240, 180, 1),
                       kernel_size=2,
                       padding=-1)


if __name__ == '__main__':
    unittest.main()