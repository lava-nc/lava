# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/
from __future__ import annotations

import typing as ty
from abc import ABC
from dataclasses import InitVar, dataclass

from lava.magma.compiler.mappable_interface import Mappable
from lava.magma.compiler.subcompilers.address import (NcLogicalAddress,
                                                      NcVirtualAddress)

if ty.TYPE_CHECKING:
    pass
from lava.magma.core.process.variable import Var


@dataclass
class LoihiAddress:
    # physical chip id of the var
    physical_chip_id: int
    # physical core id of the nc var or lmt_id of the spike counter
    physical_core_id: int
    # logical chip id used in compilation, before mapping to hardware addresses
    logical_chip_id: int
    # logical core id used in compilation, before mapping to hardware addresses
    logical_core_id: int
    # logical address/index of the var; used with nodesets for get/set
    logical_idx_addr: int
    # length of the contiguous addresses of var on core_id on chip_id
    length: int
    # stride of the contiguous addresses of var on core_id on chip_id
    stride: int


@dataclass
class LoihiNeuronAddress(LoihiAddress):
    # Which Neuron Group on the core neuron belongs to
    neuron_group_id: int


@dataclass
class AbstractVarModel(ABC):
    var: InitVar[Var] = None
    node_id: int = -1  # Default value signifying unset
    runtime_srv_id: int = -1  # Default value signifying unset

    def __post_init__(self, var: Var) -> None:
        if var is not None:
            self.var_id: int = var.id
            self.name: str = var.name
            self.shape: ty.Tuple[int, ...] = var.shape
            self.proc_id: int = var.process.id


@dataclass
class PyVarModel(AbstractVarModel):
    pass


@dataclass
class LoihiVarModel(AbstractVarModel, Mappable):
    # Physical Address this var points to on hardware
    address: ty.List[LoihiAddress] = None
    # field name mapping for var to register subfield
    field_name: ty.Optional[str] = None
    # name of the loihi register
    register_name: ty.Optional[str] = None
    # Length of the register
    register_length: ty.Optional[int] = None
    # Offset of the subfield within the register
    variable_offset: ty.Optional[int] = None
    # Length of the subfield of the register
    variable_length: ty.Optional[int] = None
    # Set in case the register is overloaded
    union_type: ty.Optional[bool] = False
    # Type of the union in case the register is overloaded
    sub_type: ty.Optional[str] = None

    def get_logical(self) -> ty.List[NcLogicalAddress]:
        """

        Returns
        -------
        Returns logical address of the port initializer.
        """
        return [
            NcLogicalAddress(chip_id=addr.logical_chip_id, core_id=addr.logical_core_id)
            for addr in self.address
        ]

    def set_virtual(self, addrs: ty.List[NcVirtualAddress]):
        """
        Sets physical address of the port initializer
        Parameters
        ----------
        addrs: List of address

        Returns
        -------

        """
        if len(addrs) != len(self.address):
            raise ValueError(
                "Length of list of address provided doesn't "
                "match size of the address list of the port "
                "initializer."
            )
        for idx, addr in enumerate(addrs):
            self.address[idx].physical_chip_id = addr.chip_id
            self.address[idx].physical_core_id = addr.core_id


@dataclass
class LoihiNeuronVarModel(LoihiVarModel):
    pass


@dataclass
class CVarModel(LoihiVarModel):
    pass


@dataclass
class NcVarModel(LoihiVarModel):
    pass
