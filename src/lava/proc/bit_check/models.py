# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyRefPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel

from lava.proc.bit_check.process import BitCheck


class AbstractPyBitCheckModel(PyLoihiProcessModel):
    """Abstract implementation of BitCheckModel.

    Specific implementations inherit from here.
    """

    state: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, int)

    bits: int = LavaPyType(int, int)
    layerid: int = LavaPyType(int, int)
    debug: int = LavaPyType(int, int)


class AbstractBitCheckModel(AbstractPyBitCheckModel):
    """Abstract implementation of BitCheck process. This
    short and simple ProcessModel can be used for quick
    checking of bit-accurate process runs as to whether
    bits will overflow when running on bit limited hardware.
    """

    state: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, int)

    bits: int = LavaPyType(int, int)
    layerid: int = LavaPyType(int, int)
    debug: int = LavaPyType(int, int)
    _overflowed: int = LavaPyType(int, int)

    def post_guard(self):
        return True

    def run_post_mgmt(self):
        value = self.state.read()

        if self.debug == 1:
            print("Value is: {} at time step: {}"
                  .format(value, self.time_step))

        # If self.check_bit_overflow(value) is true,
        # the value overflowed the allowed bits from self.bits
        if self.check_bit_overflow(value):
            self._overflowed = 1
            if self.debug == 1:
                if self.layerid:
                    print("layer id number: {}".format(self.layerid))
                print(
                    "value.max: overflows {} bits {}".format(
                        self.bits, value.max()
                    )
                )
                print(
                    "max signed value {}".format(
                        self.max_signed_int_per_bits(self.bits)
                    )
                )
                print(
                    "value.min: overflows {} bits {}".format(
                        self.bits, value.min()
                    )
                )
                print(
                    "min signed value {}".format(
                        self.max_signed_int_per_bits(self.bits)
                    )
                )

    def check_bit_overflow(self, value: ty.Type[np.ndarray]):
        value = value.astype(np.int32)
        shift_amt = 32 - self.bits
        # Shift value left by shift_amt and
        # then shift value right by shift_amt,
        # the result should equal unshifted value
        # if the value did not overflow bits in self.bits
        return not np.all(
            ((value << shift_amt) >> shift_amt) == value
        )

    def max_unsigned_int_per_bits(self, bits: ty.Type[int]):
        return (1 << bits) - 1

    def min_signed_int_per_bits(self, bits: ty.Type[int]):
        return -1 << (bits - 1)

    def max_signed_int_per_bits(self, bits: ty.Type[int]):
        return (1 << (bits - 1)) - 1


@implements(proc=BitCheck, protocol=LoihiProtocol)
@requires(CPU)
class LoihiBitCheckModel(AbstractBitCheckModel):
    """Implementation of Loihi BitCheck process. This
    short and simple ProcessModel can be used for quick
    checking of Loihi bit-accurate process run as to
    whether bits will overflow when running on Loihi Hardware.
    """

    state: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, int)

    bits: int = LavaPyType(int, int)
    layerid: int = LavaPyType(int, int)
    debug: int = LavaPyType(int, int)
