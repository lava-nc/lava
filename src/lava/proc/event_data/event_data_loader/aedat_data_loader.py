# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty

from dv import AedatFile
import numpy as np
import random
from operator import itemgetter
import os

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import OutPort

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel


class AedatDataLoader(AbstractProcess):
    def __init__(self,
                 file_path: str,
                 shape_out: tuple,
                 seed_sub_sampling: int = None,
                 **kwargs) -> None:
        super().__init__(file_path=file_path,
                         shape_out=shape_out,
                         seed_sub_sampling=seed_sub_sampling,
                         **kwargs)

        self._validate_file_path(file_path)
        self._validate_shape_out(shape_out)

        self.out_port = OutPort(shape=shape_out)

    @staticmethod
    def _validate_file_path(file_path):
        # Checking file extension
        if not file_path[-7:] == ".aedat4":
            raise ValueError(f"Given file should be an .aedat4 file. "
                             f"{file_path} given.")

        try:
            # Checking file size
            if os.stat(file_path).st_size > 0:
                return file_path
        except FileNotFoundError:
            # Checking file exists
            raise FileNotFoundError(f"File not found. {file_path} given.")

        return file_path

    @staticmethod
    def _validate_shape_out(shape_out):
        if not isinstance(shape_out[0], int):
            raise ValueError(f"Max number of events should be an integer."
                             f"{shape_out} given.")

        if shape_out[0] <= 0:
            raise ValueError(f"Max number of events should be positive. "
                             f"{shape_out} given.")

        if len(shape_out) != 1:
            raise ValueError(f"Shape of the OutPort should be 1D. "
                             f"{shape_out} given.")

        return shape_out


@implements(proc=AedatDataLoader, protocol=LoihiProtocol)
@requires(CPU)
class AedatDataLoaderPM(PyLoihiProcessModel):
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_SPARSE, int)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        self._file_path = proc_params["file_path"]
        self._shape_out = proc_params["shape_out"]

        self._init_aedat_file()
        self._frame_shape = (self._file["events"].size_x,
                             self._file["events"].size_y)

        seed_sub_sampling = proc_params["seed_sub_sampling"]
        self._random_rng = np.random.default_rng(seed_sub_sampling)

    def _init_aedat_file(self) -> None:
        self._file = AedatFile(file_name=self._file_path)
        self._stream = self._file["events"].numpy()

    def run_spk(self) -> None:
        events = self._get_next_event_batch()

        xs, ys, ps = events['x'], events['y'], events['polarity']

        data, indices = self._encode_data_and_indices(xs, ys, ps)

        if data.shape[0] > self._shape_out[0]:
            # If we have more data than our shape allows, subsample
            data, indices = self._sub_sample(data, indices)

        self.out_port.send(data, indices)

    def _get_next_event_batch(self):
        try:
            events = self._stream.__next__()
        except StopIteration:
            self._init_aedat_file()
            events = self._stream.__next__()

        return events

    def _encode_data_and_indices(self,
                                 xs: np.ndarray,
                                 ys: np.ndarray,
                                 ps: np.ndarray) \
            -> ty.Tuple[np.ndarray, np.ndarray]:
        data = ps
        indices = np.ravel_multi_index((xs, ys), self._frame_shape)

        return data, indices

    def _sub_sample(self,
                    data: np.ndarray,
                    indices: np.ndarray) \
            -> ty.Tuple[np.ndarray, np.ndarray]:
        # TODO: print a warning if subsampling, say how much data has been lost
        data_idx_array = np.arange(0, data.shape[0])
        sampled_idx = self._random_rng.choice(data_idx_array,
                                              self._shape_out[0],
                                              replace=False)

        return data[sampled_idx], indices[sampled_idx]
