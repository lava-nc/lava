# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from __future__ import annotations
import typing as ty
from abc import ABC, abstractmethod
import math

if ty.TYPE_CHECKING:
    from lava.magma.core.process.process import AbstractProcess


class AbstractProcessMember(ABC):
    """A member of a process has a reference to its parent process, a name
    and a shape because it is generally tensor-valued."""

    def __init__(self, shape: ty.Tuple):
        if not isinstance(shape, tuple):
            raise AssertionError("'shape' must be a tuple.")
        self.shape: ty.Tuple = shape
        self._process: ty.Optional[AbstractProcess] = None
        self._name: ty.Optional[str] = None

    @property
    def size(self) -> int:
        """Returns the size of the tensor-valued ProcessMember which is the
        product of all elements of its shape."""
        return math.prod(self.shape)

    @property
    def name(self) -> str:
        """Returns name of ProcessMember."""
        return self._name

    @name.setter
    def name(self, val: str):
        """Sets name of ProcessMember."""
        if not isinstance(val, str):
            raise AssertionError("'name' must be a string.")
        self._name = val

    @property
    def process(self) -> AbstractProcess:
        """Returns parent process of ProcessMember."""
        return self._process

    @process.setter
    def process(self, val: AbstractProcess):
        """Sets parent process of ProcessMember."""
        from lava.magma.core.process.process import AbstractProcess

        if not isinstance(val, AbstractProcess):
            raise AssertionError("Not a process!")
        self._process = val


class IdGeneratorSingleton(ABC):
    """A singleton class that generates globally unique ids to distinguish
    other unique objects."""

    @abstractmethod
    def __new__(cls):
        pass

    @classmethod
    def reset_singleton(cls):
        """Resets singleton."""
        cls.instance = None
        cls.is_not_initialized = True

    def __init__(self):
        self._next_id = 0

    def get_next_id(self) -> int:
        """Returns next id."""
        next_id = self._next_id
        self._next_id += 1
        return next_id
