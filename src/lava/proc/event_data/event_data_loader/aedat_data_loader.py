# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from dv import AedatFile
import numpy as np
import os.path
import typing as ty
import warnings

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
                 *,
                 file_path: str,
                 shape_out: ty.Tuple[int],
                 seed_sub_sampling: ty.Optional[int] = None) -> None:
        super().__init__(file_path=file_path,
                         shape_out=shape_out,
                         seed_sub_sampling=seed_sub_sampling)

        self._validate_file_path(file_path)
        self._validate_shape_out(shape_out)

        self.out_port = OutPort(shape=shape_out)

    @staticmethod
    def _validate_file_path(file_path: str) -> None:
        # Checking file extension
        if not file_path.lower().endswith('.aedat4'):
            raise ValueError(f"AedatDataLoader currently only supports aedat4 files (*.aedat4). "
                             f"{file_path} was given.")

        try:
            # Checking if file exists
            os.path.isfile(file_path)
        except FileNotFoundError as error:
            # Checking file exists
            raise FileNotFoundError(f"File not found. {file_path} given.")  # TODO: rewrite this?

    @staticmethod
    def _validate_shape_out(shape_out: ty.Tuple[int]) -> None:
        if len(shape_out) != 1:
            raise ValueError(f"Shape of the OutPort should have a shape of (n,). "
                             f"{shape_out} given.")

        if shape_out[0] <= 0:
            raise ValueError(f"Max number of events should be positive. "
                             f"{shape_out} given.")


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

        data, indices = self._encode_data_and_indices(events)

        if data.shape[0] > self._shape_out[0]:
            # If we have more data than our shape allows, subsample
            data, indices = self._sub_sample(data, indices)

        self.out_port.send(data, indices)

    def _get_next_event_batch(self):
        try:
            events = self._stream.__next__()
        except StopIteration:
            # does this reset the iterator and restart the file? add comment for clarification
            self._init_aedat_file()
            events = self._stream.__next__()

        return events

    def _encode_data_and_indices(self,
                                 events: dict) \
            -> ty.Tuple[np.ndarray, np.ndarray]:

        xs, ys, ps = events['x'], events['y'], events['polarity']
        data = ps
        indices = np.ravel_multi_index((xs, ys), self._frame_shape)

        return data, indices

    def _sub_sample(self,  # TODO: make a function, add max_events as parameter
                    data: np.ndarray,
                    indices: np.ndarray) \
            -> ty.Tuple[np.ndarray, np.ndarray]:
        data_idx_array = np.arange(0, data.shape[0])
        sampled_idx = self._random_rng.choice(data_idx_array,
                                              self._shape_out[0],
                                              replace=False)
        percent_data_lost = (1 - self._shape_out[0]/data.shape[0])*100
        warnings.warn(f"Out port shape is too small to send all the received data. "
                      f"Around {percent_data_lost}% of the data has been lost.")  # TODO: rewrite error message

        return data[sampled_idx], indices[sampled_idx]
