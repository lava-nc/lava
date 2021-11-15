# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest
import typing as ty
import numpy as np

from lava.magma.core.model.c.model import AbstractCProcessModel

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


class ArrayPM(AbstractCProcessModel):
    source_files = ["test_array.c"]


class Test_Creation(unittest.TestCase):
    def test_array(self):
        class PM(AbstractCProcessModel):
            source_files = ["test_array.c"]

        pm = PM(np.zeros(10))
        pm.run()
        self.assertEqual(np.sum(pm.x), 10)


if __name__ == "__main__":
    unittest.main()
