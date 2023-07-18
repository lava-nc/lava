# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
from abc import ABC, abstractmethod
from lava.magma.compiler.channels.interfaces import AbstractCspPort


class AbstractPortImplementation(ABC):
    def __init__(
        self,
        process_model: "AbstractProcessModel",  # noqa: F821
        shape: ty.Tuple[int, ...] = tuple(),
        d_type: type = int,
    ):
        self._process_model = process_model
        self._shape = shape
        self._d_type = d_type

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        """Returns the shape of the port"""
        return self._shape

    @property
    @abstractmethod
    def csp_ports(self) -> ty.List[AbstractCspPort]:
        """Returns all csp ports of the port."""

    def start(self):
        """Start all csp ports."""
        for csp_port in self.csp_ports:
            csp_port.start()

    def join(self):
        """Join all csp ports"""
        for csp_port in self.csp_ports:
            csp_port.join()
