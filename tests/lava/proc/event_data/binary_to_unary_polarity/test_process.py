# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.proc.event_data.binary_to_unary_polarity.process \
    import BinaryToUnaryPolarity


class TestProcessBinaryToUnaryPolarity(unittest.TestCase):
    def test_init(self):
        """Tests instantiation of BinaryToUnaryPolarity."""
        binary_to_unary_polarity = BinaryToUnaryPolarity(shape=(43200,))

        self.assertIsInstance(binary_to_unary_polarity, BinaryToUnaryPolarity)

    def test_invalid_shape_throws_exception(self):
        """Tests whether an invalid shape (not one-dimensional) throws an
        exception."""
        invalid_shape = (240, 180)
        with(self.assertRaises(ValueError)):
            BinaryToUnaryPolarity(shape=invalid_shape)

    def test_negative_size_throws_exception(self):
        """Tests whether shape with a negative size throws an exception."""
        invalid_shape = (-43200,)
        with(self.assertRaises(ValueError)):
            BinaryToUnaryPolarity(shape=invalid_shape)


if __name__ == '__main__':
    unittest.main()
