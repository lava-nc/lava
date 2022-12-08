# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel


class RecvSparse(AbstractProcess):
    """Process that receives arbitrary sparse data.

    Parameters
    ----------
    shape: tuple
        Shape of the InPort and Vars.
    """
    def __init__(self,
                 shape: ty.Tuple[int]) -> None:
        super().__init__(shape=shape)

        self.in_port = InPort(shape=shape)

        self.data = Var(shape=shape, init=np.zeros(shape, dtype=int))
        self.idx = Var(shape=shape, init=np.zeros(shape, dtype=int))


@implements(proc=RecvSparse, protocol=LoihiProtocol)
@requires(CPU)
class PyRecvSparsePM(PyLoihiProcessModel):
    """Receives sparse data from PyInPort and stores a padded version of
    received data and indices in Vars."""
    in_port: PyInPort = LavaPyType(PyInPort.VEC_SPARSE, int)

    data: np.ndarray = LavaPyType(np.ndarray, int)
    idx: np.ndarray = LavaPyType(np.ndarray, int)

    def run_spk(self) -> None:
        data, idx = self.in_port.recv()

        self.data = np.pad(
            data,
            pad_width=(0, self.in_port.shape[0] - data.shape[0])
        )
        self.idx = np.pad(
            idx,
            pad_width=(0, self.in_port.shape[0] - data.shape[0])
        )


class RecvDense(AbstractProcess):
    """Process that receives arbitrary dense data.

    Parameters
    ----------
    shape: tuple
        Shape of the InPort and Var.
    """
    def __init__(self,
                 shape: ty.Union[
                     ty.Tuple[int, int], ty.Tuple[int, int, int]]) -> None:
        super().__init__(shape=shape)

        self.in_port = InPort(shape=shape)

        self.data = Var(shape=shape, init=np.zeros(shape, dtype=int))


@implements(proc=RecvDense, protocol=LoihiProtocol)
@requires(CPU)
class PyRecvDensePM(PyLoihiProcessModel):
    """Receives dense data from PyInPort and stores it in a Var."""
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)

    data: np.ndarray = LavaPyType(np.ndarray, int)

    def run_spk(self) -> None:
        data = self.in_port.recv()

        self.data = data


class SendSparse(AbstractProcess):
    """Process that sends arbitrary sparse data.

    Parameters
    ----------
    shape: tuple
        Shape of the OutPort.
    """
    def __init__(self,
                 shape: ty.Tuple[int],
                 data: np.ndarray,
                 indices: np.ndarray) -> None:
        super().__init__(shape=shape, data=data, indices=indices)

        self.out_port = OutPort(shape=shape)


@implements(proc=SendSparse, protocol=LoihiProtocol)
@requires(CPU)
class PySendSparsePM(PyLoihiProcessModel):
    """Sends sparse data to PyOutPort."""
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_SPARSE, int)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self._data = proc_params["data"]
        self._indices = proc_params["indices"]

    def run_spk(self) -> None:
        self.out_port.send(data=self._data, indices=self._indices)
