# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from typing import Dict, Tuple, Union, Optional
import numpy as np
from enum import Enum, unique

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import HostCPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel


@unique
class Compression(Enum):
    """Enumeration of message compression mode.

    Attributes
    ----------
    DENSE:
        No compression. Raw 32 bit data is communicated as it is.
    SPARSE:
        Sparse 32 bit data and index is communicated.
    PACKED_4:
        Four 8 bit data packed into 32 bit message. NOTE: only works for 8 bit
        data.
    DELTA_SPARSE_8:
        8 bit data and 8 bit delta encoded index. NOTE: only works for 8 bit
        data.
    """
    DENSE = 0
    SPARSE = 1
    PACKED_4 = 2
    DELTA_SPARSE_8 = 3


class DeltaEncoder(AbstractProcess):
    """Delta encoding with threshold.

    Delta encoding looks at the difference of new input and sends only the
    difference (delta) when it is more than a positive threshold.

    Delta dynamics:
    delta   = act_new - act + residue           # delta encoding
    s_out   = delta if abs(delta) > vth else 0  # spike mechanism
    residue = delta - s_out                     # residue accumulation
    act     = act_new

    Parameters
    ----------
    shape: Tuple
        Shape of the sigma process.
    vth: int or float
        Threshold of the delta encoder.
    spike_exp: Optional[int]
        Scaling exponent with base 2 for the spike message.
        Note: This should only be used for fixed point models.
        Default is 0.
    num_bits: Optional[int]
        Precision for spike output. It is applied before spike_exp. If None,
        precision is not enforced, i.e. the spike output is unbounded.
        Default is None.
    compression : Compression
        Data compression mode, by default DENSE compression.
    """

    def __init__(self,
                 *,
                 shape: Tuple[int, ...],
                 vth: Union[int, float],
                 spike_exp: Optional[int] = 0,
                 num_bits: Optional[int] = None,
                 compression: Compression = Compression.DENSE) -> None:
        super().__init__(shape=shape, vth=vth, cum_error=False,
                         spike_exp=spike_exp, state_exp=0)

        vth = vth * (1 << (spike_exp))

        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)

        self.vth = Var(shape=(1,), init=vth)
        self.act = Var(shape=shape, init=0)
        self.residue = Var(shape=shape, init=0)
        self.spike_exp = Var(shape=(1,), init=spike_exp)
        if num_bits is not None:
            a_min = -(1 << (num_bits - 1)) << spike_exp
            a_max = ((1 << (num_bits - 1)) - 1) << spike_exp
        else:
            a_min = a_max = -1
        self.a_min = Var(shape=(1,), init=a_min)
        self.a_max = Var(shape=(1,), init=a_max)
        self.proc_params['compression'] = compression

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.proc_params['shape']


@requires(HostCPU)
class AbstractPyDeltaEncoderModel(PyLoihiProcessModel):
    """Implementation of Delta encoder."""
    a_in = None
    s_out = None

    vth: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    act: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    residue: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    spike_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)
    a_min: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    a_max: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)

    def encode_delta(self, act_new):
        delta = act_new - self.act + self.residue
        s_out = np.where(np.abs(delta) >= self.vth, delta, 0)
        if self.a_max > 0:
            s_out = np.clip(s_out, a_min=self.a_min, a_max=self.a_max)
        self.residue = delta - s_out
        self.act = act_new
        return s_out


@implements(proc=DeltaEncoder, protocol=LoihiProtocol)
@tag('dense_out')
class PyDeltaEncoderModelDense(AbstractPyDeltaEncoderModel):
    """Dense (No) compression Model of PyDeltaEncoder."""
    a_in = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    s_out = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)

    def __init__(self, proc_params: Optional[Dict] = None):
        super().__init__(proc_params)
        self.s_out_buf = np.zeros(self.proc_params['shape'])
        self.compression = self.proc_params['compression']
        if self.compression != Compression.DENSE:
            raise RuntimeError('Wrong process model selected. '
                               'Expected DENSE compression mode. '
                               f'Found {self.compression=}.')

    def run_spk(self):
        self.s_out.send(self.s_out_buf)
        a_in_data = np.left_shift(self.a_in.recv().astype(int),
                                  self.spike_exp)
        self.s_out_buf = self.encode_delta(a_in_data)


@implements(proc=DeltaEncoder, protocol=LoihiProtocol)
@tag('sparse_out')
class PyDeltaEncoderModelSparse(AbstractPyDeltaEncoderModel):
    """Sparse compression Model of PyDeltaEncoder.

    Based on compression mode, it can be
    * SPARSE: 32 bit data and 32 bit index used for messaging sparse data.
    * PACKED_4: Four 8 bit data packed into one 32 bit data for messaging.
    * DELTA_SPARSE_8: 8 bit index and 8 bit data messaging.
    """
    a_in = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    s_out = LavaPyType(PyOutPort.VEC_SPARSE, np.int32, precision=24)

    def __init__(self, proc_params: Optional[Dict] = None):
        super().__init__(proc_params)
        self.data = np.array([0])
        self.idx = np.array([0])
        self.compression = self.proc_params['compression']
        if not(
            self.compression == Compression.SPARSE
            or self.compression == Compression.PACKED_4
            or self.compression == Compression.DELTA_SPARSE_8
        ):
            raise RuntimeError('Wrong process model selected. '
                               'Expected SPARSE or PACKED_4 or DELTA_SPARSE_8 '
                               'compression mode. '
                               f'Found {self.compression=}.')

    def encode_sparse(self, s_out):
        """Basic sparse encoding."""
        idx = np.argwhere(s_out.flatten() != 0)
        data = s_out.flatten()[idx]
        if len(idx) == 0:
            idx = np.array([0])
            data = np.array([0])
        return data, idx

    def encode_packed_4(self, s_out):
        """4x 8bit data encodig into one 32 bit data."""
        padded = np.zeros(int(np.ceil(np.prod(s_out.shape) / 8) * 8))
        padded[:np.prod(s_out.shape)] = np.bitwise_and(s_out.flatten(), 0xFF)
        padded = padded.astype(np.int32)
        packed = (np.left_shift(padded[3::4], 24)
                  + np.left_shift(padded[2::4], 16)
                  + np.left_shift(padded[1::4], 8)
                  + padded[0::4])
        return packed[0::2], packed[1::2]

    def encode_delta_sparse_8(self, s_out):
        """8 bit compressed data and index encoding."""
        idx = np.argwhere(s_out.flatten() != 0)
        data = s_out.flatten()[idx]
        if len(idx) == 0:
            idx = np.array([0])
            data = np.array([0])
        max_idx = 0xFF
        if idx[0] > max_idx:
            idx = np.concatenate([np.zeros(1, dtype=idx.dtype),
                                  idx.flatten()])
            data = np.concatenate([np.zeros(1, dtype=idx.dtype),
                                   data.flatten()])

        # 8 bit index encoding
        idx[1:] = idx[1:] - idx[:-1] - 1  # default increment of 1
        delta_idx = []
        delta_data = []
        start = 0
        for i in np.argwhere(idx >= max_idx)[:, 0]:
            delta_idx.append((idx[start:i].flatten()) % max_idx)
            delta_data.append(data[start:i].flatten())
            repeat_data = idx[i] // max_idx
            num_repeats = repeat_data // max_idx
            delta_idx.append(np.array([max_idx] * (num_repeats + 1)).flatten())
            delta_data.append(np.array([max_idx] * num_repeats
                                       + [repeat_data % max_idx]).flatten())
            delta_idx.append((idx[i:i + 1].flatten()) % max_idx)
            delta_data.append(data[i:i + 1].flatten())
            start = i + 1
        delta_idx.append(idx[start:].flatten())
        delta_data.append(data[start:].flatten())

        if len(delta_idx) > 0:
            delta_idx = np.concatenate(delta_idx)
            delta_data = np.concatenate(delta_data)
        else:
            delta_idx = idx.flatten()
            delta_data = data.flatten()

        padded_idx = np.zeros(int(np.ceil(len(delta_idx) / 4) * 4))
        padded_data = np.zeros(int(np.ceil(len(delta_data) / 4) * 4))

        padded_idx[:len(delta_idx)] = np.bitwise_and(delta_idx, 0xFF)
        padded_data[:len(delta_data)] = np.bitwise_and(delta_data, 0xFF)

        padded_idx = padded_idx.astype(np.int32)
        padded_data = padded_data.astype(np.int32)

        packed_idx = (np.left_shift(padded_idx[3::4], 24)
                      + np.left_shift(padded_idx[2::4], 16)
                      + np.left_shift(padded_idx[1::4], 8)
                      + padded_idx[0::4])
        packed_data = (np.left_shift(padded_data[3::4], 24)
                       + np.left_shift(padded_data[2::4], 16)
                       + np.left_shift(padded_data[1::4], 8)
                       + padded_data[0::4])
        return packed_data, packed_idx

    def decode_encode_delta_sparse_8(self, packed_data, packed_idx):
        """Python decoding script for delta_sparse_8 encoding. It is useful for
        debug and verify the encoding."""
        data_list = []
        idx_list = []
        count = 0
        data = 0
        idx = 0
        for p_data, p_idx in zip(packed_data, packed_idx):
            for _ in range(4):
                data = p_data & 0xFF
                idx_1 = p_idx & 0xFF
                if idx_1 == 0xFF:
                    idx += data * 0xFF
                    data = 0
                else:
                    idx += idx_1 + int(count > 0)
                if data != 0:
                    data_list.append(data)
                    idx_list.append(idx)
                p_data >>= 8
                p_idx >>= 8
                count += 1
        return np.array(data_list), np.array(idx_list)

    def run_spk(self):
        self.s_out.send(self.data, self.idx)
        # Receive synaptic input
        a_in_data = np.left_shift(self.a_in.recv().astype(int),
                                  self.spike_exp)
        s_out = self.encode_delta(a_in_data)
        if self.compression == Compression.SPARSE:
            self.data, self.idx = self.encode_sparse(s_out)
        elif self.compression == Compression.PACKED_4:
            self.data, self.idx = self.encode_packed_4(s_out)
        elif self.compression == Compression.DELTA_SPARSE_8:
            self.data, self.idx = self.encode_delta_sparse_8(s_out)
