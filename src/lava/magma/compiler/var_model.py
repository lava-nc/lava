# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

from __future__ import annotations

import typing as ty
from abc import ABC
from dataclasses import dataclass, InitVar

from lava.magma.compiler.mappable_interface import Mappable
from lava.magma.compiler.subcompilers.address import NcLogicalAddress, \
    NcVirtualAddress
from lava.magma.core.process.ports.connection_config import SpikeIOInterface, \
    SpikeIOPort, SpikeIOMode

if ty.TYPE_CHECKING:
    pass
from lava.magma.core.process.variable import Var

ChipIdx: int
CoreIdx: int


@dataclass
class LoihiAddress:
    # physical chip id of the var
    physical_chip_id: ChipIdx
    # physical core id of the nc var or lmt_id of the spike counter
    physical_core_id: CoreIdx
    # logical chip id used in compilation, before mapping to hardware addresses
    logical_chip_id: ChipIdx
    # logical core id used in compilation, before mapping to hardware addresses
    logical_core_id: CoreIdx
    # logical address/index of the var; used with nodesets for get/set
    logical_idx_addr: int
    # length of the contiguous addresses of var on core_id on chip_id
    length: int
    # stride of the contiguous addresses of var on core_id on chip_id
    stride: int


@dataclass
class LoihiNeuronAddress(LoihiAddress):
    # To which Neuron Group on the core a neuron belongs
    neuron_group_id: int


@dataclass
class LoihiSynapseAddress(LoihiAddress):
    # To which SynEntry on the core a synapse belongs
    syn_entry_id: int


@dataclass
class LoihiInAxonAddress(LoihiAddress):
    # To which Profile on the core a synapse belongs
    profile_id: int


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
            self.dtype = type(var.init)


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
        return [NcLogicalAddress(chip_id=addr.logical_chip_id,
                                 core_id=addr.logical_core_id) for addr in
                self.address]

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
            raise ValueError("Length of list of address provided doesn't "
                             "match size of the address list of the port "
                             "initializer.")
        for idx, addr in enumerate(addrs):
            self.address[idx].physical_chip_id = addr.chip_id
            self.address[idx].physical_core_id = addr.core_id


@dataclass
class LoihiNeuronVarModel(LoihiVarModel):
    pass


@dataclass
class LoihiSynapseVarModel(LoihiVarModel):
    pass


@dataclass
class CVarModel(LoihiVarModel):
    pass


@dataclass
class NcVarModel(LoihiVarModel):
    pass


@dataclass
class Region:
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    logical_chip_idx: ChipIdx
    logical_core_idx: CoreIdx
    physical_chip_idx: ChipIdx = None
    physical_core_idx: CoreIdx = None


@dataclass
class ConvInVarModel(AbstractVarModel, Mappable):
    x_dim: int = 0
    y_dim: int = 0
    f_dim: int = 0
    x_split: int = 0
    f_split: int = 0
    regions: ty.List[Region] = None

    def get_logical(self) -> ty.List[NcLogicalAddress]:
        """

        Returns
        -------
        Returns logical address of the port initializer.
        """
        return [NcLogicalAddress(chip_id=region.logical_chip_idx,
                                 core_id=region.logical_core_idx) for region in
                self.regions]

    def set_virtual(self, addrs: ty.List[NcVirtualAddress]):
        """
        Sets physical address of the port initializer
        Parameters
        ----------
        addrs: List of address

        Returns
        -------

        """
        if len(addrs) != len(self.regions):
            raise ValueError("Length of list of address provided doesn't "
                             "match size of the regions list of the port "
                             "initializer.")
        for idx, addr in enumerate(addrs):
            self.regions[idx].physical_chip_idx = addr.chip_id
            self.regions[idx].physical_core_idx = addr.core_id


@dataclass
class ConvNeuronVarModel(LoihiNeuronVarModel):
    alloc_dims: ty.List[ty.Tuple[int, int, int]] = None
    valid_dims: ty.List[ty.Tuple[int, int, int]] = None
    var_shape: ty.Tuple[int, int, int] = None


@dataclass
class ByteEncoder:
    """Encodes ptr, len, base"""
    base: int = 0
    len: int = 0
    ptr: int = 0


@dataclass
class CoreEncoder:
    """ Encodes a core xyp """
    x: ByteEncoder = ByteEncoder()
    y: ByteEncoder = ByteEncoder()
    p: ByteEncoder = ByteEncoder()


@dataclass
class ChipEncoder:
    """ Encoding for chip field """
    x: ByteEncoder = ByteEncoder()
    y: ByteEncoder = ByteEncoder()
    z: ByteEncoder = ByteEncoder()


@dataclass
class AxonEncoder:
    """ Encoding for axon field """
    hi: ByteEncoder = ByteEncoder()
    lo: ByteEncoder = ByteEncoder()


@dataclass
class TimeCompare:
    """Used by SpikeBlock to determine when to inject spikes"""
    time_mode: int
    num_time_bits: int
    time_len: int
    time_ptr: int


@dataclass
class DecodeConfig:
    receive_mode: int
    decode_mode: int


@dataclass
class SpikeEncoder:
    islong: int
    core: CoreEncoder
    axon: AxonEncoder
    chip: ChipEncoder
    payload: ty.List[ByteEncoder]


@dataclass
class NcSpikeIOVarModel(NcVarModel):
    msg_queue_id: int = 0
    num_message_bits: int = 8
    interface: SpikeIOInterface = SpikeIOInterface.ETHERNET
    spike_io_port: SpikeIOPort = SpikeIOPort.ETHERNET
    spike_io_mode: SpikeIOMode = SpikeIOMode.TIME_COMPARE
    ethernet_chip_id: ty.Optional[ty.Tuple[int, int, int]] = None
    ethernet_chip_idx: ty.Optional[int] = None
    decode_config: ty.Optional[DecodeConfig] = None
    time_compare: ty.Optional[TimeCompare] = None
    spike_encoder: ty.Optional[SpikeEncoder] = None


@dataclass
class NcConvSpikeInVarModel(NcSpikeIOVarModel):
    # Tuple will be in the order of [atom_paylod, atom_axon, addr_idx]
    region_map: ty.List[ty.List[ty.Tuple[int, int, int]]] = None
