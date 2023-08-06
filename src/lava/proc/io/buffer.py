# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from typing import Optional, Union, Type

from lava.magma.core.process.variable import Var
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort, RefPort

from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort, PyRefPort


class Buffer(AbstractProcess):
    """Buffer receives data from OutPorts or VarPorts in each
    timestep.

    To add a connection, call `connect` with either an InPort
    or a Var from another process.

    To read the contents of the buffer, call `get` on the Var
    returned by a call to `connect` while the process model
    is still active (i.e. before calling `process.stop`).

    TODO: Implement 'wrap_around', 'reallocate', 'read_write_file'
    TODO: Support OutPorts also

    Parameters
    ----------
    length: int, default = 100
        The number of timesteps that can be recorded in the buffer.
    overflow: str, default = 'raise_exception'
        The desired behavior when the buffer overflows. Options are
        'raise_exception', 'wrap_around', and 'reallocate'.
    """
    def __init__(self,
                 *,
                 length: int = 100,
                 overflow: str = 'raise_error') -> None:
        super().__init__(length=length, overflow=overflow,
            map_out=[], map_in=[], map_ref=[])
        self.length = length
        self.overflow = overflow
        self.map_out = []
        self.map_in = []
        self.map_ref = []
        self.index = 1000

    def connect(self, other: Union[InPort, OutPort, Var],
                init: Optional[np.ndarray] = None) -> Var:
        """Connect the buffer to an OutPort or Var from another process.
        Calling this method will instantiate a new InPort or RefPort as
        needed in the buffer and a corresponding Var of the appropriate
        shape and length.
        Parameters
        ----------
        other: Union[OutPort, Var]
            The other port or var to connect to and store in the buffer.
        init: Optional[ndarray]
            The initial value of the buffer Var. This will determine the
            values sent from an InPort buffer and the default values for
            an OutPort or RefPort buffer.
        Returns
        -------
        The Var which will store the buffered data.
        """
        index = self.index
        var_shape = other.shape + (self.length,)
        if init is None:
            init = np.zeros(var_shape)
        var = Var(shape=var_shape, init=init)
        var.name = f'Var{index}'
        setattr(self, var.name, var)

        if isinstance(other, InPort):
            port = OutPort(shape=other.shape)
            port.name = f'Out{index}'
            other.connect_from(port)
            self.map_out.append((var.name, port.name))
            self.proc_params.overwrite('map_out', self.map_out)
        elif isinstance(other, OutPort):
            port = InPort(shape=other.shape)
            port.name = f'In{index}'
            other.connect(port)
            self.map_in.append((var.name, port.name))
            self.proc_params.overwrite('map_in', self.map_in)
        elif isinstance(other, Var):
            port = RefPort(shape=other.shape)
            port.name = f'Ref{index}'
            port.connect_var(other)
            self.map_ref.append((var.name, port.name))
            self.proc_params.overwrite('map_ref', self.map_ref)
        else:
            raise ValueError(f'Other {other} is not an InPort, OutPort, '
                             'or Var.')
        setattr(self, port.name, port)
        self._post_init()
        self.index += 1
        return var


class MetaPyBuffer(type(PyLoihiProcessModel)):
    """This metaclass allows dynamic port and var generation."""
    def __getattr__(cls, name):
        if 'In' in name:
            return LavaPyType(PyInPort.VEC_DENSE, float)
        elif 'Out' in name:
            return LavaPyType(PyOutPort.VEC_DENSE, float)
        elif 'Ref' in name:
            return LavaPyType(PyRefPort.VEC_DENSE, float)
        elif 'Var' in name:
            return LavaPyType(np.ndarray, float)
        else:
            raise AttributeError(name=name, obj=cls)


@implements(proc=Buffer, protocol=LoihiProtocol)
@requires(CPU)
class PyBuffer(PyLoihiProcessModel, metaclass=MetaPyBuffer):
    """Python CPU model for Buffer. Uses dense floating point numpy
    arrays for buffer storage and operations."""
    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.length = proc_params['length']
        self.overflow = proc_params['overflow']
        self.map_in = proc_params['map_in']
        self.map_out = proc_params['map_out']
        self.map_ref = proc_params['map_ref']

        for var, port in self.map_in:
            setattr(self, var, LavaPyType(np.ndarray, float))
            setattr(self, port, LavaPyType(PyInPort.VEC_DENSE, float))

        for var, port in self.map_out:
            setattr(self, var, LavaPyType(np.ndarray, float))
            setattr(self, port, LavaPyType(PyOutPort.VEC_DENSE, float))

        for var, port in self.map_ref:
            setattr(self, var, LavaPyType(np.ndarray, float))
            setattr(self, port, LavaPyType(PyRefPort.VEC_DENSE, float))

    def run_spk(self) -> None:
        """Read InPorts and write to buffer Vars and read from buffer
        Vars to write to OutPorts."""
        t = self.time_step - 1
        if t >= self.length:
            self.do_overflow()
        for var, port in self.map_in:
            data = getattr(self, port).recv()
            getattr(self, var)[..., t] = data
        for var, port in self.map_out:
            data = getattr(self, var)[..., t]
            getattr(self, port).send(data)

    def post_guard(self) -> None:
        """Do management phase only if needed for RefPort reads."""
        return len(self.map_ref) > 0

    def run_post_mgmt(self) -> None:
        """Read RefPorts and write to buffer Vars."""
        t = self.time_step - 1
        if t >= self.length:
            self.do_overflow()
        for var, port in self.map_ref:
            data = getattr(self, port).read()
            getattr(self, var)[..., t] = data

    def do_overflow(self) -> None:
        """Implement overflow behavior."""
        if self.overflow == 'raise_error':
            raise RuntimeError(f'PyBuffer overflow: timestep {self.time_step}'
                                ' is greater than length {self.length}')
        else:
            raise NotImplementedError(f'PyBuffer overflow: overflow '
                                       '{self.overflow} is not implemented.')
