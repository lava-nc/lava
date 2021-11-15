# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from abc import ABCMeta, abstractmethod


class AbstractPortImplementation(metaclass=ABCMeta):
    def __init__(
        self,
        process_model: "AbstractProcessModel",  # noqa: F821
        shape: ty.Tuple[int, ...] = tuple(),
        d_type: type = int,
    ):
        self._process_model = process_model
        self._shape = shape
        self._d_type = d_type

    @abstractmethod
    def start(self):
        # start all csp ports
        ...

    @abstractmethod
    def join(self):
        # join all csp ports
        ...
