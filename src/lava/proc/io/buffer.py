# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from typing import Optional, Tuple, Type, Union

from lava.magma.core.process.variable import Var
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import (
    AbstractPort,
    InPort,
    OutPort,
    RefPort,
)

from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort, PyRefPort


class Buffer(AbstractProcess):
    """Buffer stores data in Vars that can be connected to Vars and
    ports in other processes.

    Buffer Vars have a final dimension equal to the length of the Buffer.
    If a buffer Var is connected to an OutPort, it will store the vector
    sent by that OutPort on each timestep. If a buffer Var is connected to
    an InPort, it will send


    To add a connection, call `connect` with either an InPort
    or a Var from another process.

    To read the contents of the buffer, call `get` on the Var
    returned by a call to `connect` while the process model
    is still active (i.e. before calling `process.stop`).

    TODO: Implement 'wrap_around', 'reallocate', 'read_write_file'

    Parameters
    ----------
    length: int, default = 100
        The number of timesteps that can be recorded in the buffer.
    overflow: str, default = 'raise_exception'
        The desired behavior when the buffer overflows. Options are
        'raise_exception', 'wrap_around', and 'reallocate'.
    """

    def __init__(
        self, *, length: int = 100, overflow: str = "raise_error"
    ) -> None:
        super().__init__(
            length=length,
            overflow=overflow,
            map_out=[],
            map_in=[],
            map_read=[],
            map_write=[],
        )
        self.length = length
        self.overflow = overflow
        self.map_out = []
        self.map_in = []
        self.map_read = []
        self.map_write = []
        self.index = 0

    def add_var(
        self,
        name: str,
        shape: Optional[Tuple] = None,
        init: Optional[np.ndarray] = None,
    ) -> Var:
        """Add a buffer Var.
        Parameters
        ----------
        name: str
            Name of the Var to create.
        shape: Optional[Tuple], default = None
            The port shape which this buffer Var can store. The Var shape
            will include an additional dimension corresponding to the length
            of the buffer in timesteps.
        init: Optional[ndarray], default = None
            The initial value of the buffer Var. This will determine the
            values sent to any InPort connected to this Var, or the default
            values that will be incrementally overwritten if this is connected
            to an OutPort.
        Returns
        -------
        The buffer Var.
        """
        if not shape and not init:
            raise ValueError("shape and init cannot both be None. "
                             f"{shape}, {init}")
        elif (
            shape is not None and init is not None and init.shape[:-1] != shape
        ):
            raise ValueError(
                "shape and init.shape are not compatible: "
                f"{shape} != {init.shape}"
            )
        if init is not None and init.shape[-1] != self.length:
            raise ValueError(
                "init.shape is not compatible with length: "
                f"Last dim {init.shape[:-1]} != {self.length}"
            )
        if init is None:
            init = np.zeros(shape + (self.length,))
        var = Var(shape=init.shape, init=init)
        setattr(self, f"{name}Var", var)
        self._post_init()
        setattr(self, name, var)
        return var

    def add_outport(self, buffer: Union[Var, str]) -> OutPort:
        """Create an OutPort mapped to the buffer Var indicated by name or
        reference.
        Parameters
        ----------
        buffer : Union[Var, str]
            A reference to a buffer Var in this Buffer or the name of a Var.
        Returns
        -------
        The newly mapped OutPort.
        """
        if isinstance(buffer, str):
            buffer = getattr(self, buffer)
        port = self._create_port(OutPort, buffer, buffer.shape[:-1])
        self._map_buffer_to_port(buffer, port, "map_out")
        return port

    def add_inport(self, buffer: Union[Var, str]) -> InPort:
        """Create an InPort mapped to the buffer Var indicated by name or
        reference.
        Parameters
        ----------
        buffer : Union[Var, str]
            A reference to a buffer Var in this Buffer or the name of a Var.
        Returns
        -------
        The newly mapped InPort.
        """
        if isinstance(buffer, str):
            buffer = getattr(self, buffer)
        port = self._create_port(InPort, buffer, buffer.shape[:-1])
        self._map_buffer_to_port(buffer, port, "map_in")
        return port

    def add_refport(self, buffer: Union[Var, str], mode: str) -> RefPort:
        """Create a RefPort mapped to the buffer Var indicated by name or
        reference.
        Parameters
        ----------
        buffer : Union[Var, str]
            A reference to a buffer Var in this Buffer or the name of a Var.
        Returns
        -------
        The newly mapped RefPort.
        """
        if mode not in ["read", "write"]:
            raise ValueError("mode not in [read, write]: {mode}")
        if isinstance(buffer, str):
            buffer = getattr(self, buffer)
        port = self._create_port(RefPort, buffer, buffer.shape[:-1])
        self._map_buffer_to_port(buffer, port, f"map_{mode}")
        return port

    def connect(
        self,
        name: str,
        other: Union[InPort, Var],
        init: Optional[np.ndarray] = None,
    ) -> Var:
        """Connect a buffer Var to an InPort or Var from another process.

        Calling this method will create a new buffer Var if it doesn't exist
        and it will create an OutPort or RefPort as needed.

        Raises an error if the buffer Var exists and the shapes are
        incompatible.

        Parameters
        ----------
        name: str
            The name of the buffer Var to connect. This will be created
            if it does not exist.
        other: Union[InPort, Var]
            The other port or var to connect to the buffer.
        init: Optional[ndarray]
            The initial value of the buffer Var. This will determine the
            values sent to the connected InPort or Var.
        Returns
        -------
        The Var which will write buffered data.
        """
        if hasattr(self, name):
            var = getattr(self, name)
            if var.shape[:-1] != other.shape:
                raise ValueError(
                    "var.shape and other.shape are not "
                    f"compatible: {var.shape} != {other.shape}"
                )
            if init:
                raise ValueError("var exists but init is not None. "
                                 f"{init}")
        else:
            var = self.add_var(name, other.shape, init)
        if isinstance(other, InPort):
            port = self.add_outport(var)
            port.connect(other)
        else:
            port = self.add_refport(var, "write")
            port.connect_var(other)
        return var

    def connect_from(
        self,
        name: str,
        other: Union[OutPort, Var],
        init: Optional[np.ndarray] = None,
    ) -> Var:
        """Connect a buffer Var to an OutPort or Var from another process.

        Calling this method will create a new buffer Var if the named Var
        does not exist and it will create an InPort or RefPort as needed.

        Raises an error if the buffer Var exists and the shapes are
        incompatible.

        Parameters
        ----------
        name: str
            The name of the buffer Var to connect. This will be created
            if it does not exist.
        other: Union[InPort, Var]
            The other port or var to connect to the buffer.
        init: Optional[ndarray]
            The initial value of the buffer Var. This will determine the
            values sent to the connected InPort or Var.
        Returns
        -------
        The Var which will store the buffered data.
        """
        if hasattr(self, name):
            var = getattr(self, name)
            if var.shape[:-1] != other.shape:
                raise ValueError(
                    f"var.shape and other.shape are not "
                    f"compatible: {var.shape} != {other.shape}"
                )
            if init:
                raise ValueError("var exists but init is not None.")
        else:
            var = self.add_var(name, other.shape, init)
        if isinstance(other, OutPort):
            port = self.add_inport(var)
            port.connect_from(other)
        else:
            port = self.add_refport(var, "read")
            port.connect_var(other)
        return var

    def _create_port(
        self, port_cls: Type[AbstractPort], buffer: Var, shape: Tuple
    ):
        """Create a port to connect the buffer to another port or var."""
        port = port_cls(shape=shape)
        port.name = f"{buffer.name[:-3]}{port_cls.__name__}{self.index}"
        self.index += 1
        setattr(self, port.name, port)
        self._post_init()
        return port

    def _map_buffer_to_port(
        self, buffer: Var, port: Type[AbstractPort], map_name: str
    ):
        """Map the buffer Var to a corresponding port."""
        getattr(self, map_name).append((buffer.name, port.name))
        self.proc_params.overwrite(map_name, getattr(self, map_name))


class MetaPyBuffer(type(PyLoihiProcessModel)):
    """This metaclass allows dynamic port and var generation."""

    def __getattr__(cls, name):
        if "InPort" in name:
            return LavaPyType(PyInPort.VEC_DENSE, float)
        elif "OutPort" in name:
            return LavaPyType(PyOutPort.VEC_DENSE, float)
        elif "RefPort" in name:
            return LavaPyType(PyRefPort.VEC_DENSE, float)
        elif "Var" in name:
            return LavaPyType(np.ndarray, float)
        else:
            print(name)
            raise AttributeError(name, cls)


@implements(proc=Buffer, protocol=LoihiProtocol)
@requires(CPU)
class PyBuffer(PyLoihiProcessModel, metaclass=MetaPyBuffer):
    """Python CPU model for Buffer. Uses dense floating point numpy
    arrays for buffer storage and operations."""

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.length = proc_params["length"]
        self.overflow = proc_params["overflow"]
        self.do_overflow = self.get_overflow_func()
        self.map_in = proc_params["map_in"]
        self.map_out = proc_params["map_out"]
        self.map_read = proc_params["map_read"]
        self.map_write = proc_params["map_write"]

        for var, port in self.map_in:
            setattr(self, var, LavaPyType(np.ndarray, float))
            setattr(self, port, LavaPyType(PyInPort.VEC_DENSE, float))

        for var, port in self.map_out:
            setattr(self, var, LavaPyType(np.ndarray, float))
            setattr(self, port, LavaPyType(PyOutPort.VEC_DENSE, float))

        for var, port in self.map_read:
            setattr(self, var, LavaPyType(np.ndarray, float))
            setattr(self, port, LavaPyType(PyRefPort.VEC_DENSE, float))

        for var, port in self.map_write:
            setattr(self, var, LavaPyType(np.ndarray, float))
            setattr(self, port, LavaPyType(PyRefPort.VEC_DENSE, float))

    def run_spk(self) -> None:
        """Read InPorts and write to buffer Vars and read from buffer
        Vars to write to OutPorts."""
        i = self.get_index_with_overflow()
        for var, port in self.map_in:
            data = getattr(self, port).recv()
            getattr(self, var)[..., i] = data
        for var, port in self.map_out:
            data = getattr(self, var)[..., i]
            getattr(self, port).send(data)

    def post_guard(self) -> None:
        """Do management phase only if needed for RefPort reads."""
        return len(self.map_read) + len(self.map_write) > 0

    def run_post_mgmt(self) -> None:
        """Read RefPorts and write to buffer Vars."""
        i = self.get_index_with_overflow()
        for var, port in self.map_read:
            data = getattr(self, port).read()
            getattr(self, var)[..., i] = data
        for var, port in self.map_write:
            data = getattr(self, var)[..., i]
            getattr(self, port).write(data)

    def get_index_with_overflow(self) -> int:
        """Get the index to read or write in the buffer and implement overflow
        behavior if needed.
        """
        index = self.time_step - 1
        return self.do_overflow(index)

    def get_overflow_func(self) -> None:
        """Return a function to apply overflow behavior."""
        if self.overflow == "raise_error":
            return self.do_raise_on_overflow
        elif self.overflow == "wrap_around":
            return self.do_wrap_around
        else:
            raise NotImplementedError(
                "PyBuffer overflow: overflow "
                f"{self.overflow} is not implemented."
            )

    def do_raise_on_overflow(self, i):
        if i > self.length:
            raise RuntimeError(
                f"PyBuffer overflow: timestep {self.time_step}"
                f" is greater than length {self.length}"
            )
        return i

    def do_wrap_around(self, i):
        return i % self.length
