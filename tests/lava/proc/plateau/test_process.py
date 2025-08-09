# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


import unittest
from lava.proc.plateau.process import Plateau


class TestPlateauProcess(unittest.TestCase):
    """Tests for Plateau class"""
    def test_init(self):
        """Tests instantiation of Plateau"""
        plat = Plateau(
            shape=(100,),
            dv_dend=100,
            dv_soma=1,
            vth_dend=10,
            vth_soma=1,
            up_dur=10,
            name="Plat"
        )

        self.assertEqual(plat.name, "Plat")
        self.assertEqual(plat.dv_dend.init, 100)
        self.assertEqual(plat.dv_soma.init, 1)
        self.assertEqual(plat.vth_dend.init, 10)
        self.assertEqual(plat.vth_soma.init, 1)
        self.assertEqual(plat.up_dur.init, 10)
