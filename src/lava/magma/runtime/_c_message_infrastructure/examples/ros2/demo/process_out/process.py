# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
import numpy as np


class ProcessOut(AbstractProcess):
    """Process that receives (1) the raw ROS camera events, (2) the spike rates
    of the selective as well as (3) the multi-peak DNF per pixel. It sends
    these values through a multiprocessing pipe (rather than a Lava OutPort)
    to allow for plotting."
    """

    def __init__(self, shape_roscam_frame, shape_dnf, send_pipe):
        super().__init__(
            shape_roscam_frame=shape_roscam_frame,
            shape_dnf=shape_dnf,
            send_pipe=send_pipe,
        )
        self.roscam_frame_port = InPort(shape=shape_roscam_frame)
        self.dnf_multipeak_rates_port = InPort(shape=shape_dnf)
        self.dnf_selective_rates_port = InPort(shape=shape_dnf)


@implements(proc=ProcessOut, protocol=LoihiProtocol)
@requires(CPU)
class DataRelayerPM(PyLoihiProcessModel):
    roscam_frame_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    dnf_multipeak_rates_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    dnf_selective_rates_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self._send_pipe = proc_params["send_pipe"]

    def run_spk(self):
        roscam_frame = self.roscam_frame_port.recv()
        dnf_multipeak_rates = self.dnf_multipeak_rates_port.recv()
        dnf_selective_rates = self.dnf_selective_rates_port.recv()

        roscam_frame_ds_image = np.rot90(roscam_frame)
        dnf_multipeak_rates_ds_image = np.rot90(dnf_multipeak_rates)
        dnf_selective_rates_ds_image = np.rot90(dnf_selective_rates)

        data_dict = {
            "roscam_frame_ds_image": roscam_frame_ds_image,
            "dnf_multipeak_rates_ds_image": dnf_multipeak_rates_ds_image,
            "dnf_selective_rates_ds_image": dnf_selective_rates_ds_image,
        }

        self._send_pipe.send(data_dict)
