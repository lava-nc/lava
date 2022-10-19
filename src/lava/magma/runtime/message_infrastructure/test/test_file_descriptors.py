# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import os
from lava.magma.core.run_conditions import RunSteps
from lava.proc.lif.process import LIF
from lava.magma.core.run_configs import Loihi1SimCfg
from time import sleep
from subprocess import run


def run_process():
    du = 10
    dv = 100
    vth = 4900

    # Create processes
    lif2 = LIF(shape=(2, ),
            vth=vth,
            dv=dv,
            du=du,
            bias_mant=0,
            name='lif2')

    lif2.run(condition=RunSteps(num_steps=1),
        run_cfg=Loihi1SimCfg(select_tag="fixed_pt"))

    lif2.stop()


def get_file_descriptor_usage():
    result = run("lsof 2>/dev/null | grep python | grep FIFO | wc -l",
                 shell=True)
    sleep(0.1)
    return result.stdout


class TestFileDescriptors(unittest.TestCase):
    num_iterations = 1000

    @unittest.skipIf(os.name != "posix",
        "Checking file descriptor only for POSIX systems.")
    def test_file_descriptor_usage(self):
        # Check initial state that file descriptor usage is zero
        file_descriptor_usage = get_file_descriptor_usage()
        self.assertEqual(file_descriptor_usage, None)


        for iteration in range(self.num_iterations):    
            # Run Process
            run_process()

            # Check file descriptor usage after running processes
            file_descriptor_usage = get_file_descriptor_usage()
            self.assertEqual(file_descriptor_usage, None)


if __name__ == "__main__":
    unittest.main()
