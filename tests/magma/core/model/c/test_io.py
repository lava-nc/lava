# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest
import os

from lava.magma.core.model.c.model import AbstractCProcessModel

from mockports import MockDataPort, MockServicePort


class Test_io(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        return super().setUpClass()

    def test_io(self):
        class IOPM(AbstractCProcessModel):
            service_to_process_cmd: MockServicePort = MockServicePort()
            port: MockDataPort = MockDataPort()
            source_files = ["test_io.c"]

        pm = IOPM()
        pm.run()
        self.assertFalse(pm.port.recd is None)
        self.assertFalse(pm.port.sent is None)
        self.assertEqual(pm.port.recd, pm.port.sent)


if __name__ == "__main__":
    unittest.main()
