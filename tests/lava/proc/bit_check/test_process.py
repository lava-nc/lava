# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
from lava.proc.bit_check.process import BitCheck


class TestBitCheck(unittest.TestCase):
    """Tests for BitCheck class"""

    def test_init(self) -> None:
        """Tests instantiation of BitCheck"""
        bc = BitCheck(shape=(2, 3, 4), layerid=10, debug=1, bits=12)
        self.assertEqual(bc.shape, (2, 3, 4))
        self.assertEqual(bc.layerid.get(), 10)
        self.assertEqual(bc.debug.get(), 1)
        self.assertEqual(bc.bits.get(), 12)
        bc01 = BitCheck(shape=(1,))
        self.assertEqual(bc01.shape, (1,))
        self.assertEqual(bc01.layerid.init, None)
        self.assertEqual(bc01.debug.init, 0)
        self.assertEqual(bc01.bits.init, 24)
        bc02 = BitCheck()
        self.assertEqual(bc02.shape, (1,))

        bitcheckers = []
        for i in range(1, 31):
            bitcheckers.append(BitCheck(bits=i))
            self.assertEqual(bitcheckers[i - 1].bits.init, i)

    def test_init_bits_exception(self) -> None:
        """Tests instantiation of BitCheck throws
        error on incorrect bits parameter"""
        with self.assertRaises(ValueError):
            BitCheck(bits=0)
        with self.assertRaises(ValueError):
            BitCheck(bits=32)
