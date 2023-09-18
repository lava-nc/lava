# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

import functools as ft
import typing as ty
from dataclasses import dataclass
from enum import IntEnum

from lava.magma.compiler.mappable_interface import Mappable
from lava.magma.compiler.subcompilers.address import NcLogicalAddress, \
    NcVirtualAddress
from lava.magma.compiler.var_model import LoihiVarModel, ConvInVarModel
from lava.magma.core.model.spike_type import SpikeType
from lava.magma.core.process.ports.connection_config import ConnectionConfig


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
        # TODO: Need to clean this
        if isinstance(self.var_model, ConvInVarModel):
            return self.var_model.get_logical()

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
        # TODO: Need to clean this
        if isinstance(self.var_model, ConvInVarModel):
            self.var_model.set_virtual(addrs)
            return

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
    # Denotes port is associated with PY/NC Process
    PY_NC = 4


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


@dataclass
class LoihiCInPortInitializer(LoihiIOPortInitializer):
    embedded_core = 0
    embedded_counters = None


@dataclass
class LoihiPyInPortInitializer(LoihiCInPortInitializer):
    connection_config: ty.Optional[ConnectionConfig] = None


@dataclass
class LoihiOutPortInitializer(LoihiIOPortInitializer):
    """Port Initializer for a OutPort for C/NC Models"""


@dataclass
class LoihiVarPortInitializer(LoihiPortInitializer):
    # This address needs to be defined based on var model
    pass
