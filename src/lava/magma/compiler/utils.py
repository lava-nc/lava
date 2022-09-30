import functools as ft
import typing as ty
import numpy as np
from dataclasses import dataclass
from enum import IntEnum

from lava.magma.compiler.mappable_interface import Mappable
from lava.magma.compiler.subcompilers.address import NcLogicalAddress, \
    NcVirtualAddress
from lava.magma.compiler.var_model import LoihiVarModel
from lava.magma.core.model.spike_type import SpikeType


@dataclass
class VarInitializer:
    name: str
    shape: ty.Tuple[int, ...]
    value: ty.Any
    var_id: int


@dataclass
class PortInitializer:
    name: str
    shape: ty.Tuple[int, ...]
    d_type: type
    port_type: str
    size: int
    transform_funcs: ty.Dict[str, ty.List[ft.partial]] = None

    @property
    def bytes(self) -> int:
        return np.prod(self.shape) * np.dtype(self.d_type).itemsize


# check if can be a subclass of PortInitializer
@dataclass
class VarPortInitializer:
    name: str
    shape: ty.Tuple[int, ...]
    var_name: str
    d_type: type
    port_type: str
    size: int
    port_cls: type
    transform_funcs: ty.Dict[str, ty.List[ft.partial]] = None

    @property
    def bytes(self) -> int:
        return np.prod(self.shape) * np.dtype(self.d_type).itemsize


@dataclass
class LoihiVarInitializer(VarInitializer):
    d_type: type


@dataclass
class LoihiPortInitializer(PortInitializer, Mappable):
    """This address needs to be defined based on var model"""
    var_model: ty.Optional[LoihiVarModel] = None

    def get_logical(self) -> ty.List[NcLogicalAddress]:
        """

        Returns
        -------
        Returns logical address of the port initializer.
        """
        return [NcLogicalAddress(chip_id=addr.logical_chip_id,
                                 core_id=addr.logical_core_id) for addr in
                self.var_model.address]

    def set_virtual(self, addrs: ty.List[NcVirtualAddress]):
        """
        Sets physical address of the port initializer
        Parameters
        ----------
        addrs: List of address

        Returns
        -------

        """
        if len(addrs) != len(self.var_model.address):
            raise ValueError("Length of list of address provided doesn't "
                             "match size of the address list of the port "
                             "initializer.")
        for idx, addr in enumerate(addrs):
            self.var_model.address[idx].physical_chip_id = addr.chip_id
            self.var_model.address[idx].physical_core_id = addr.core_id


class LoihiConnectedPortType(IntEnum):
    """Types of port connectivity; direction does not matter"""
    # Denotes port is associated with C/NC Process
    C_NC = 1
    # Denotes port is associated with C/C Process
    C_C = 2
    # Denotes port is associated with C/PY Process
    C_PY = 3


class LoihiConnectedPortEncodingType(IntEnum):
    """Encoding type of the connected port - Required in case of C_PY"""
    # Denotes data fmt is VEC_DENSE
    VEC_DENSE = 1
    # Denotes data fmt is SEQ_DENSE
    SEQ_DENSE = 2
    # Denotes data fmt is VEC_SPARSE
    VEC_SPARSE = 3
    # Denotes data fmt is SEQ_SPARSE
    SEQ_SPARSE = 4


@dataclass
class LoihiIOPortInitializer(LoihiPortInitializer):
    """Port Initializer for a I/O Port for C/NC Models"""
    connected_port_type: ty.Optional[LoihiConnectedPortType] = None
    connected_port_encoding_type: ty.Optional[LoihiConnectedPortEncodingType] \
        = None
    spike_type: ty.Optional[SpikeType] = None


@dataclass
class LoihiInPortInitializer(LoihiIOPortInitializer):
    """Port Initializer for a InPort for C/NC Models"""
    pass


@dataclass
class LoihiCInPortInitializer(LoihiIOPortInitializer):
    embedded_core = 0
    embedded_counters = None


@dataclass
class LoihiOutPortInitializer(LoihiIOPortInitializer):
    """Port Initializer for a OutPort for C/NC Models"""
    pass


@dataclass
class LoihiVarPortInitializer(LoihiPortInitializer):
    # This address needs to be defined based on var model
    pass
