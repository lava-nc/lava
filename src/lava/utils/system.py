# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import os

from typing import Optional, Callable, TypeVar

# NOTE: Awkward, but the std lib deprecated decorator is not available yet,
# so I'm adding it here to deprecate this module :/
_T = TypeVar("_T")


def deprecated(__msg: str) -> Callable[[_T], _T]:
    def decorator(__arg: _T) -> _T:
        __arg.__deprecated__ = __msg
        return __arg
    return decorator

# NOTE: This module is deprecated as of v0.8.0 in favor of lava.utils.loihi


class staticproperty(property):
    """Wraps static member function of a class as a static property of that
    class.
    """

    def __get__(self, cls, owner):
        return staticmethod(self.fget).__get__(None, owner)()


@deprecated('This class is deprecated. Use lava.utils.loihi instead.')
class Loihi2:
    preferred_partition: str = None

    @deprecated('This method is deprecated. Use lava.utils.loihi instead.')
    @staticmethod
    def set_environ_settings(partititon: Optional[str] = None) -> None:
        """Sets the os environment for execution on Loihi.
        Parameters
        ----------
        partititon : str, optional
            Loihi partition name, by default None.
        """
        if 'SLURM' not in os.environ and 'NOSLURM' not in os.environ:
            os.environ['SLURM'] = '1'
        if 'LOIHI_GEN' not in os.environ:
            os.environ['LOIHI_GEN'] = 'N3B3'
        if 'PARTITION' not in os.environ and partititon is not None:
            os.environ['PARTITION'] = partititon

    @deprecated('This method is deprecated. Use lava.utils.loihi instead.')
    @staticproperty
    def is_loihi2_available() -> bool:
        """Checks if Loihi2 compiler is available and sets the environment
        vairables.
        Returns
        -------
        bool
            Flag indicating whether Loih 2 is available or not.
        """
        try:
            from lava.magma.compiler.subcompilers.nc.ncproc_compiler import \
                CompilerOptions
            CompilerOptions.verbose = True
        except ModuleNotFoundError:
            # Loihi2 compiler is not availabe
            return False
        Loihi2.set_environ_settings(Loihi2.preferred_partition)
        return True

    @deprecated('This method is deprecated. Use lava.utils.loihi instead.')
    @staticproperty
    def partition():
        """Get the partition information."""
        if 'PARTITION' in os.environ.keys():
            return os.environ['PARTITION']
        return 'Unspecified'
