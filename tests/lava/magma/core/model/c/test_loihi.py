# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest
import os

from lava.magma.core.model.c.model import AbstractCProcessModel
from lava.proc.dense.models import PyDenseModel

from mockports import MockServicePort


class Test_loihi(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        return super().setUpClass()

    def test_loihi(self):
        class LoihiPM(AbstractCProcessModel, PyDenseModel):
            service_to_process_cmd: MockServicePort = MockServicePort(10)
            source_files = ["test_loihi.c"]

        pm = LoihiPM()
        # test for error due to mock service port not being a loihi syncronizer
        self.assertRaises(ValueError, pm.run)


if __name__ == "__main__":
    unittest.main()
