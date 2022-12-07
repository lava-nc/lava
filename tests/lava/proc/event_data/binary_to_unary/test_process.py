# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.proc.event_data.binary_to_unary_polarity.process \
    import BinaryToUnaryPolarity


class TestProcessBinaryToUnaryPolarity(unittest.TestCase):
    def test_init(self):
        """Tests instantiation of BinaryToUnaryPolarity."""
        converter = BinaryToUnaryPolarity(shape=(43200,))

        self.assertIsInstance(converter, BinaryToUnaryPolarity)

    def test_invalid_shape_throws_exception(self):
        """Tests whether a shape argument with an invalid shape
        throws an exception."""
        with(self.assertRaises(ValueError)):
            BinaryToUnaryPolarity(shape=(240, 180))

    def test_negative_size_throws_exception(self):
        """Tests whether a shape argument with a negative size
        throws an exception."""
        with(self.assertRaises(ValueError)):
            BinaryToUnaryPolarity(shape=(-43200,))


if __name__ == '__main__':
    unittest.main()
