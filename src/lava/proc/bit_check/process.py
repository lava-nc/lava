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
        *,
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
        overflow when running on bit limited hardware.

        Parameters
        ----------
        shape: Tuple
            Shape of the sigma process.
            Default is (1,).
        layerid: int or float
            Layer number of network.
            Default is None.
        debug: 0 or 1
            Enable (1) or disable (0) debug print.
            Default is 0.
        bits: int
            Bits to use when checking overflow, 1-32.
            Default is 24.
        """
        super().__init__(
            name=name,
            log_config=log_config,
            **kwargs
        )

        initial_shape = (1,)
        self.state = RefPort(initial_shape)

        self.layerid: ty.Type(Var) = Var(shape=initial_shape, init=layerid)
        self.debug: ty.Type(Var) = Var(shape=initial_shape, init=debug)
        if bits <= 31 and bits >= 1:
            self.bits: ty.Type(Var) = Var(shape=initial_shape, init=bits)
        else:
            raise ValueError("bits value is \
                             {} but should be 1-31".format(bits))
        self._overflowed: ty.Type(Var) = Var(shape=initial_shape, init=0)

    def connect_var(self, var: Var) -> None:
        self.state = RefPort(var.shape)
        self.state.connect_var(var)

        self.layerid = Var(name="layerid",
                           shape=var.shape,
                           init=self.layerid.init
                           )
        self.debug = Var(name="debug",
                         shape=var.shape,
                         init=self.debug.init
                         )
        self.bits = Var(name="bits",
                        shape=var.shape,
                        init=self.bits.init
                        )
        self._overflowed = Var(name="overflowed",
                               shape=var.shape,
                               init=self._overflowed.init
                               )

        self._post_init()

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        """Return shape of the Process."""
        return self.state.shape

    @property
    def overflowed(self) -> ty.Type[int]:
        """Return overflow Var of Process.
        1 is overflowed, 0 is not overflowed."""
        return self._overflowed.get()
