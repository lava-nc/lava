import typing as ty
from dataclasses import dataclass

import numpy as np


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
    d_type: ty.Type[np.intc]
    port_type: str
    size: int
