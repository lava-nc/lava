# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
from lava.proc.filter.process import ExpFilter 

class TestExpFilterProcess(unittest.TestCase):
    """Tests for ExpFilter class"""
    def test_init(self):
        """Tests instantiation of ExpFilter"""
        filter = ExpFilter(shape=(100,),
                           tau=0.5,
                           name="ExpFilter")

        self.assertEqual(filter.name, "ExpFilter")
        self.assertEqual(filter.tau.init, 0.5)
        self.assertEqual(filter.proc_params["shape"], (100,))