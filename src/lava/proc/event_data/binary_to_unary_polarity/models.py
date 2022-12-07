# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

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
    1 for positive events) coming from its in_port to unary polarity
    (1 for negative and positive events) and sends it through its out_port.
    """
    in_port: PyInPort = LavaPyType(PyInPort.VEC_SPARSE, int)
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_SPARSE, int)

    def run_spk(self) -> None:
        data, indices = self.in_port.recv()

        data = self._encode(data)

        self.out_port.send(data, indices)

    @staticmethod
    def _encode(data: np.ndarray) -> np.ndarray:
        """Transform event-based data with binary polarity to unary polarity.

        Parameters
        ----------
        data : ndarray
            Event-based data with binary polarity.

        Returns
        ----------
        result : ndarray
            Event-based data with unary polarity.
        """
        data[data == 0] = 1

        return data
