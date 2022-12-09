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
from lava.utils.events import sub_sample, encode_data_and_indices


class AedatStream(AbstractProcess):
    """Process that reads event-based data from an aedat4 file.

    This process outputs a sparse tensor of the event data stream, meaning
    two 1-dimensional vectors containing polarity data and indices. The
    data is sub-sampled to fit the given output shape. The process is
    implemented such that the reading from file loops back to the beginning
    of the file when it reaches the end.

    Parameters
    ----------
    file_path : str
        Path to the desired aedat4 file.

    shape_out : tuple (shape (n,))
        The shape of the OutPort. The size of this parameter sets a maximum
        number of events per time-step, and the process will subsample data
        in order to fit it into this port.

    seed_sub_sampling : int, optional
        Seed used for the random number generator that sub-samples data to
        fit the OutPort.
    """
    def __init__(self,
                 *,
                 file_path: str,
                 shape_out: ty.Tuple[int],
                 seed_sub_sampling: ty.Optional[int] = None,
                 **kwargs) -> None:
        super().__init__(file_path=file_path,
                         shape_out=shape_out,
                         seed_sub_sampling=seed_sub_sampling,
                         **kwargs)

        self._validate_file_path(file_path)
        self._validate_shape_out(shape_out)

        self.out_port = OutPort(shape=shape_out)

    @staticmethod
    def _validate_file_path(file_path: str) -> None:
        """
        Checks whether the file extension is valid and if the file can
        be found. Raises relevant exception if not.
        """
        if not file_path.lower().endswith('.aedat4'):
            raise ValueError(f"AedatDataLoader currently only supports aedat4 files (*.aedat4). "
                             f"{file_path} was given.")

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found. {file_path} given.")

    @staticmethod
    def _validate_shape_out(shape_out: ty.Tuple[int]) -> None:
        """
        Checks whether the given shape is valid and that the size given
        is not a negative number. Raises relevant exception if not
        """
        if len(shape_out) != 1:
            raise ValueError(f"Shape of the OutPort should be (n,)."
                             f"{shape_out} was given.")

        if shape_out[0] <= 0:
            raise ValueError(f"Max number of events should be positive."
                             f"{shape_out} was given.")


@implements(proc=AedatStream, protocol=LoihiProtocol)
@requires(CPU)
class AedatStreamPM(PyLoihiProcessModel):
    """
    Implementation of the Aedat Data Loader process on Loihi, with sparse
    representation of events.
    """
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_SPARSE, int)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        self._file_path = proc_params["file_path"]
        self._shape_out = proc_params["shape_out"]

        self._init_aedat_file()
        self._frame_shape = (self._file["events"].size_x,
                             self._file["events"].size_y)

        self._seed_sub_sampling = proc_params["seed_sub_sampling"]
        self._random_rng = np.random.default_rng(self._seed_sub_sampling)

    def run_spk(self) -> None:
        """
        Compiles events into a batch (roughly 10ms long). The polarity data
        and x and y values are then used to encode the sparse tensor using
        row-major (C-style) encoding. The data is sub-sampled if necessary,
        and then sent out.
        """
        events = self._get_next_event_batch()

        data, indices = encode_data_and_indices(frame_shape=self._frame_shape,
                                                events=events)

        # If we have more data than our shape allows, sub-sample
        if data.shape[0] > self._shape_out[0]:
            data, indices = sub_sample(data, indices,
                                       self._shape_out[0], self._random_rng)

            # warn the user if we need to sub-sample
            percentage_data_lost = (1 - self._shape_out[0] / data.shape[0]) * 100
            warnings.warn(f"Read {data.shape[0]} events. Maximum number of events is {self._shape_out[0]}. "
                          f"Removed {data.shape[0] - self._shape_out[0]} ({percentage_data_lost:.1f}%) "
                          f"events by subsampling.")

        self.out_port.send(data, indices)

    def _get_next_event_batch(self):
        """
        Compiles events from the event stream into batches which will be
        treated in a single timestep. Once we reach the end of the file, the
        process loops back to the start of the file.
        """
        try:
            # If end of file, raises StopIteration error.
            events = self._stream.__next__()
        except StopIteration:
            # Reset the iterator and loop back to the start of the file.
            self._init_aedat_file()
            events = self._stream.__next__()

        return events

    def _init_aedat_file(self) -> None:
        """
        Resets the event stream.
        """
        self._file = AedatFile(file_name=self._file_path)
        self._stream = self._file["events"].numpy()

    def _encode_data_and_indices(self,
                                 events: np.ndarray) \
            -> ty.Tuple[np.ndarray, np.ndarray]:
        """
        Extracts the polarity data, and x and y indices from the given
        batch of events, and encodes them using C-style encoding.
        """
        xs, ys, ps = events['x'], events['y'], events['polarity']
        data = ps
        indices = np.ravel_multi_index((xs, ys), self._frame_shape)

        return data, indices