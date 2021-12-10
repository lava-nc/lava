# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import os
import sys

from lava.magma.core.model.c.model import AbstractCProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.decorator import implements

from . import mockports


@unittest.skipIf(sys.platform.startswith("win"), "can't build on windows")
class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # os.chdir(os.path.dirname(os.path.abspath(__file__)))
        return super().setUpClass()

    @unittest.skip("avoid problems with parallel compile")
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

    @unittest.skipUnless(sys.platform.startswith("darwin"), "run this on mac")
    def test_io(self):
        class IOPM(AbstractCProcessModel):
            service_to_process_cmd: mockports.MockServicePort = (
                mockports.MockServicePort()
            )
            port: mockports.MockDataPort = mockports.MockDataPort()
            source_files = [
                os.path.dirname(os.path.abspath(__file__)) + "/test_io.c"
            ]

        pm = IOPM()
        pm.run()
        self.assertFalse(pm.port.recd is None)
        self.assertFalse(pm.port.sent is None)
        self.assertEqual(pm.port.recd, pm.port.sent)

    @unittest.skipUnless(sys.platform.startswith("linux"), "run this on linux")
    def test_loihi(self):
        @implements(protocol=LoihiProtocol)
        class PyLoihiDummy(PyLoihiProcessModel):
            service_to_process_cmd: mockports.MockNpServicePort = (
                mockports.MockNpServicePort(4)
            )
            process_to_service_ack: mockports.MockNpServicePort = (
                mockports.MockNpServicePort(4)
            )
            service_to_process_req: mockports.MockNpServicePort = (
                mockports.MockNpServicePort(4)
            )
            current_ts: int = 0
            var_ports = []

        class LoihiPM(AbstractCProcessModel, PyLoihiDummy):
            source_files = [
                os.path.dirname(os.path.abspath(__file__)) + "/test_loihi.c"
            ]

        pm = LoihiPM()
        # test for error on phase zero to end test
        self.assertRaises(BaseException, pm.run)


if __name__ == "__main__":
    unittest.main()
