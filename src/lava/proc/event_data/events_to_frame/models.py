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
from lava.proc.event_data.events_to_frame.process import EventsToFrame


@implements(proc=EventsToFrame, protocol=LoihiProtocol)
@requires(CPU)
class PyEventsToFramePM(PyLoihiProcessModel):
    """PyLoihiProcessModel implementing the EventsToFrame Process.

    Transforms a collection of (sparse) events with unary or binary polarity
    into a (dense) frame of shape (W, H, 1) or (W, H, 2).
    """
    in_port: PyInPort = LavaPyType(PyInPort.VEC_SPARSE, int)
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    def run_spk(self) -> None:
        data, indices = self.in_port.recv()
        dense_data = self._transform(data, indices)
        self.out_port.send(dense_data)

    def _transform(self, data: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Transforms from an event-based representation to a frame-based
        representation.

        Parameters
        ----------
        data : ndarray
            Array of events.
        indices : ndarray
            Array of event indices.

        Returns
        ----------
        result : ndarray
            Frame of events.
        """
        shape_out = self.out_port.shape
        dense_data = np.zeros(shape_out)

        xs, ys = np.unravel_index(indices, shape_out[:-1])

        dense_data[xs[data == 0], ys[data == 0], 0] = 1
        dense_data[xs[data == 1], ys[data == 1], shape_out[-1] - 1] = 1

        return dense_data
