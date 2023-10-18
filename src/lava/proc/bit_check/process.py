# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty

from lava.magma.core.process.process import LogConfig, AbstractProcess
from lava.magma.core.process.ports.ports import RefPort
from lava.magma.core.process.variable import Var


class BitCheck(AbstractProcess):
    def __init__(
        self,
        shape: ty.Tuple[int, ...] = (1,),
        layerid: ty.Optional[int] = None,
        debug: ty.Optional[int] = 0,
        bits: ty.Optional[int] = 24,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
        **kwargs,
    ) -> None:
        """BitCheck process.
        This process is used for quick checking of
        bit-accurate process run as to whether bits will
        overflow when running on bit sensitive hardware.

        Parameters
        ----------
        shape: Tuple
            shape of the sigma process.
            Default is (1,).
        layerid: int or float
            layer number of network.
            Default is None.
        debug: 0 or 1
            Enable (1) or disable (0) debug print.
            Default is 0.
        bits: int
            bits to use when checking overflow, 1-32
            Default is 24.
        """
        super().__init__(
            shape=shape,
            name=name,
            log_config=log_config,
            **kwargs,
        )
        super().__init__(shape=shape, **kwargs)

        self.ref = RefPort(shape=shape)

        self.layerid = Var(shape=shape, init=layerid)
        self.debug = Var(shape=shape, init=debug)
        if bits <= 31 and bits >= 1:
            self.bits = Var(shape=shape, init=bits)
        else:
            raise ValueError("bits value is \
                             {} but should be 1-31".format(bits))
        """
        overflowed: int
            0 by default, changed to 1 if overflow occurs
            Default is 0.
        """
        self._overflowed: ty.Type(Var) = Var(shape=shape, init=0)

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        """Return shape of the Process."""
        return self.proc_params["shape"]

    @property
    def overflowed(self) -> ty.Type[int]:
        """Return overflow Var of Process.
        1 is overflowed, 0 is not overflowed"""
        return self._overflowed.get()
