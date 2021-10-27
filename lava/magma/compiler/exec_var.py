# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from __future__ import annotations
import typing as ty
from abc import ABC
from dataclasses import dataclass

if ty.TYPE_CHECKING:
    from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var


@dataclass
class AbstractExecVar(ABC):
    var: Var
    node_id: int
    runtime_srv_id: int

    @property
    def var_id(self) -> int:
        return self.var.id

    @property
    def name(self) -> str:
        return self.var.name

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        return self.var.shape

    @property
    def process(self) -> AbstractProcess:
        return self.var.process

    @property
    def proc_id(self) -> int:
        return self.var.process.id


@dataclass
class PyExecVar(AbstractExecVar):
    pass


@dataclass
class CExecVar(AbstractExecVar):
    pass


@dataclass
class NcExecVar(AbstractExecVar):
    chip_id: int
    core_id: int
    register_base_addr: int
    entry_id: int
    field: str
