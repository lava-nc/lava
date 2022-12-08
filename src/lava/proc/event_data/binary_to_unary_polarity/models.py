# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.event_data.binary_to_unary_polarity.process \
    import BinaryToUnaryPolarity


@implements(proc=BinaryToUnaryPolarity, protocol=LoihiProtocol)
@requires(CPU)
class PyBinaryToUnaryPolarityPM(PyLoihiProcessModel):
    """PyLoihiProcessModel implementing the BinaryToUnaryPolarity Process.

    Transforms event-based data with binary polarity (0 for negative events,
    1 for positive events) to unary polarity (1 for negative and positive
    events)."""
    in_port: PyInPort = LavaPyType(PyInPort.VEC_SPARSE, int)
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_SPARSE, int)

    def run_spk(self) -> None:
        data, indices = self.in_port.recv()
        data[data == 0] = 1
        self.out_port.send(data, indices)
