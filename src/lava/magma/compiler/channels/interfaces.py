# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

import typing as ty
from abc import ABC, abstractmethod
from enum import IntEnum

import numpy as np


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

    def is_msg_size_static(self) -> bool:
        return True


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
    CNc = 3
    NcC = 4
    CC = 3
    NcNc = 5
    NcPy = 6
    PyNc = 7
