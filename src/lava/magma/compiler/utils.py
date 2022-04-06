import typing as ty
import functools as ft
from dataclasses import dataclass


@dataclass
class VarInitializer:
    name: str
    shape: ty.Tuple[int, ...]
    value: ty.Any
    var_id: int
    parent_list_name: ty.Optional[ty.AnyStr] = None


@dataclass
class PortInitializer:
    name: str
    shape: ty.Tuple[int, ...]
    d_type: type
    port_type: str
    size: int
    parent_list_name: ty.Optional[ty.AnyStr] = None
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
    parent_list_name: ty.Optional[ty.AnyStr] = None
    transform_funcs: ty.Dict[str, ty.List[ft.partial]] = None
