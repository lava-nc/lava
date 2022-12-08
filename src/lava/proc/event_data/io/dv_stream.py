# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
from dv import NetworkNumpyEventPacketInput

from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol


class DvStream(AbstractProcess):
    def __init__(self,
                 *,
                 address: str,
                 port: int,
                 shape_out: ty.Tuple[int],
                 **kwargs) -> None:
        super().__init__(address=address,
                         port=port,
                         shape_out=shape_out,
                         **kwargs)
        self._validate_address(address)
        self._validate_port(port)
        self._validate_shape(shape_out)

        self.out_port = OutPort(shape=shape_out)

    @staticmethod
    def _validate_address(address: str) -> None:
        """Check that address is not an empty string or None."""
        if not address:
            raise ValueError("Address parameter not specified."
                             "The address must be an IP address or domain.")

    @staticmethod
    def _validate_port(port: int) -> None:
        """Check whether the given port number is valid."""
        _min = 0
        _max = 65535
        if not (_min <= port <= _max):
            raise ValueError(f"Port number must be an integer between {_min=} "
                             f"and {_max=}; got {port=}.")

    @staticmethod
    def _validate_shape(shape: ty.Tuple[int]) -> None:
        """Check that shape one-dimensional with a positive size."""
        if len(shape) != 1:
            raise ValueError(f"Shape of the OutPort should be (n,); "
                             f"got {shape=}.")
        if shape[0] <= 0:
            raise ValueError(f"Size of the shape (maximum number of events) "
                             f"must be positive; got {shape=}.")


@implements(proc=DvStream, protocol=LoihiProtocol)
@requires(CPU)
class DvStreamPM(PyLoihiProcessModel):
    """Python ProcessModel of the DvStream Process"""
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_SPARSE, int)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        self._address = proc_params["address"]
        self._port = proc_params["port"]
        self._shape_out = proc_params["shape_out"]
        self._event_stream = proc_params.get("event_stream")
        if not self._event_stream:
            self._event_stream = NetworkNumpyEventPacketInput(
                address=self._address,
                port=self._port
            )
