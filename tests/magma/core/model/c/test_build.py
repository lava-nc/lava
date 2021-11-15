# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest

from lava.magma.core.model.c.model import AbstractCProcessModel

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


class Test_Build(unittest.TestCase):
    """
    def test_boring(self):
        class PM(AbstractCProcessModel):
            source_files = ["test_boring.c"]

        pm = PM()
        pm.run()
    """

    def test_runstate(self):
        class PM(AbstractCProcessModel):
            source_files = ["test_runstate.c"]

        pm = PM()
        pm.run()


if __name__ == "__main__":
    unittest.main()
