# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause
from abc import ABC, abstractmethod

from lava.magma.core.model.model import AbstractProcessModel


class AbstractCProcessModel(AbstractProcessModel, ABC):
    """Abstract interface for a C ProcessModels.

    Example for how variables and ports might be initialized:
        a_in:  LavaCType(CInPort, 'short', precision=16)
        s_out: LavaCType(COutPort, 'short', precision=1)
        u:     LavaCType(array, 'signed long int', precision=24)
        v:     LavaCType(array, 'signed long int, precision=24)
        bias:  LavaCType(array, 'signed short int', precision=12)
        du:    LavaCType(scalar, 'short int', precision=12)
    """

    @property
    @abstractmethod
    def source_file_name(self) -> str:
        """Returns file name of *.h and *.c file containing implementation of
        Process behavior.
        By default, it should be in same directory as Python module of
        Process and Executable.
        """
        pass 


class AbstractParallelCProcessModel(AbstractCProcessModel, ABC):
    """Abstract interface for a C ProcessModel that can be distributed over
    multiple cores.
    Idea: This should be an implementation of a CProcessModel that can in
    particular be distributed over multiple embedded processors to accelerate
    sequential behavior or to provide larger contiguous virtual space of
    spike counters.
    """

    pass
