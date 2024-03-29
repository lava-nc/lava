# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/
import threading
import unittest
import typing as ty

import numpy as np

from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort, PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort, InPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg


class Process1(AbstractProcess):
    def __init__(self, shape_1: ty.Tuple, **kwargs):
        super().__init__(shape_1=shape_1, **kwargs)

        self.in_1 = InPort(shape=shape_1)
        self.out_1 = OutPort(shape=shape_1)


@implements(proc=Process1, protocol=LoihiProtocol)
@requires(CPU)
class LoihiDenseSpkPyProcess1PM(PyLoihiProcessModel):
    in_1: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out_1: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)

    def run_spk(self):
        print(f"Receiving in Process...")
        data_1 = self.in_1.recv()
        print(f"Received {data_1} in Process...")

        print(f"Sending {data_1} from Process...")
        self.out_1.send(data_1)
        print(f"Sent {data_1} from Process!")


class TestExternalPipeIO(unittest.TestCase):
    def test_run_steps_non_blocking(self):
        data = [[1], [2], [3], [4], [5]]

        relay = Process1(shape_1=(1,))
        # Control buffer size with buffer_size arg, default is 64
        relay.in_1.flag_external_pipe()
        # Control buffer size with buffer_size arg, default is 64
        relay.out_1.flag_external_pipe()

        run_cfg = Loihi2SimCfg()
        run_condition = RunSteps(num_steps=5, blocking=False)

        def thread_inject_fn() -> None:
            for send_data_single_item in data:
                print(f"Sending {send_data_single_item} from thread_inject...")
                # Use probe() before send() to know whether or not send() will
                # block (i.e if the buffer of external_pipe_csp_send_port
                # is full).
                relay.in_1.external_pipe_csp_send_port.send(
                    np.array(send_data_single_item))
                print(f"Sent {send_data_single_item} from thread_inject!")

        def thread_extract_fn() -> None:
            for _ in range(len(data)):
                print(f"Receiving in thread_extract...")
                # Use probe() before recv() to know whether or not recv() will
                # block (i.e if the buffer of external_pipe_csp_recv_port
                # is empty).
                received_data = relay.out_1.external_pipe_csp_recv_port.recv()
                print(f"Received {received_data} in thread_extract!")

        thread_inject = threading.Thread(target=thread_inject_fn,
                                         daemon=True)
        thread_extract = threading.Thread(target=thread_extract_fn,
                                          daemon=True)

        relay.run(condition=run_condition, run_cfg=run_cfg)

        thread_inject.start()
        thread_extract.start()

        relay.wait()
        relay.stop()


if __name__ == "__main__":
    unittest.main()
