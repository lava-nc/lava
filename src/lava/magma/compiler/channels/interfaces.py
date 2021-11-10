# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from enum import IntEnum

import numpy as np
from abc import ABC, abstractmethod


# ToDo: (AW) probe and peek methods are missing
class AbstractCspPort(ABC):
    """Abstract base class for CSP channel."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def d_type(self) -> np.dtype:
        pass

    @property
    @abstractmethod
    def shape(self) -> ty.Tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def join(self):
        pass


class AbstractCspSendPort(AbstractCspPort):
    @abstractmethod
    def send(self, data: np.ndarray):
        pass


class AbstractCspRecvPort(AbstractCspPort):
    @abstractmethod
    def recv(self) -> np.ndarray:
        pass


class Channel(ABC):
    @property
    @abstractmethod
    def src_port(self) -> AbstractCspSendPort:
        pass

    @property
    @abstractmethod
    def dst_port(self) -> AbstractCspRecvPort:
        pass


class ChannelType(IntEnum):
    """Type of a channel given the two process models"""
    PyPy = 0
    CPy = 1
    PyC = 2
