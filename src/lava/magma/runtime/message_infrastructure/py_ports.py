# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

import typing as ty
from abc import ABC, abstractmethod

import numpy as np


class AbstractTransferPort(ABC):
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


class AbstractSendPort(AbstractTransferPort):
    @abstractmethod
    def send(self, data: np.ndarray):
        pass


class AbstractRecvPort(AbstractTransferPort):
    @abstractmethod
    def recv(self) -> np.ndarray:
        pass
