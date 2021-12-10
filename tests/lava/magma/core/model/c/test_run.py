# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import os

from lava.magma.core.model.c.model import AbstractCProcessModel

from . import mockports


class Test_run(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # os.chdir(os.path.dirname(os.path.abspath(__file__)))
        # sys.path.append(os.getcwd())
        return super().setUpClass()

    def test_run(self):
        class RunPM(AbstractCProcessModel):
            service_to_process_cmd: mockports.MockServicePort = (
                mockports.MockServicePort(10)
            )
            source_files = [
                os.path.dirname(os.path.abspath(__file__)) + "/test_run.c"
            ]

        pm = RunPM()
        pm.run()
        self.assertEqual(pm.service_to_process_cmd.phase, 0)


if __name__ == "__main__":
    unittest.main()
