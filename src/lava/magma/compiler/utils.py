import functools as ft
import typing as ty
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
