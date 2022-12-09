# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty
import warnings
from dv import NetworkNumpyEventPacketInput

from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.utils.events import sub_sample, encode_data_and_indices


class DvStream(AbstractProcess):
    def __init__(self,
                 *,
                 address: str,
                 port: int,
                 shape_frame_in: ty.Tuple[int, int],
                 shape_out: ty.Tuple[int],
                 seed_sub_sampling: ty.Optional[int] = 0,
                 **kwargs) -> None:
        super().__init__(address=address,
                         port=port,
                         shape_out=shape_out,
                         shape_frame_in=shape_frame_in,
                         seed_sub_sampling=seed_sub_sampling,
                         **kwargs)
        self._validate_address(address)
        self._validate_port(port)
        self._validate_shape(shape_out)
        self._validate_frame_size(shape_frame_in)

        self.out_port = OutPort(shape=shape_out)

    @staticmethod
    def _validate_address(address: str) -> None:
        """Check that address is not an empty string or None."""
        if not address:
            raise ValueError("Address parameter not specified."
                             "The address must be an IP address or domain.")

    @staticmethod
    def _validate_port(port: int) -> None:
        """Check whether the given port number is valid."""
        _min = 0
        _max = 65535
        if not (_min <= port <= _max):
            raise ValueError(f"Port number must be an integer between {_min=} "
                             f"and {_max=}; got {port=}.")

    @staticmethod
    def _validate_shape(shape: ty.Tuple[int]) -> None:
        """Check that shape one-dimensional with a positive size."""
        if len(shape) != 1:
            raise ValueError(f"Shape of the OutPort should be (n,); "
                             f"got {shape=}.")
        if shape[0] <= 0:
            raise ValueError(f"Size of the shape (maximum number of events) "
                             f"must be positive; got {shape=}.")

    @staticmethod
    def _validate_frame_size(shape: ty.Tuple[int, int]) -> None:
        """Check that shape one-dimensional with a positive size."""
        if len(shape) != 2:
            raise ValueError(f"Shape of the frame should be (n,); "
                             f"got {shape=}.")
        if shape[0] <= 0 or shape[1] <= 0:
            raise ValueError(f"Size of the frame "
                             f"must be positive; got {shape=}.")


@implements(proc=DvStream, protocol=LoihiProtocol)
@requires(CPU)
class DvStreamPM(PyLoihiProcessModel):
    """Python ProcessModel of the DvStream Process"""
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_SPARSE, int)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        self._address = proc_params["address"]
        self._port = proc_params["port"]
        self._shape_out = proc_params["shape_out"]
        self._frame_shape = proc_params["shape_frame_in"]
        self._seed_sub_sampling = proc_params["seed_sub_sampling"]
        self._random_rng = np.random.default_rng(self._seed_sub_sampling)
        self._event_stream = proc_params.get("event_stream")
        if not self._event_stream:
            self._event_stream = NetworkNumpyEventPacketInput(
                address=self._address,
                port=self._port
            )

    def run_spk(self) -> None:
        """
        Compiles events into a batch (roughly 10ms long). The polarity data
        and x and y values are then used to encode the sparse tensor. The
        data is sub-sampled if necessary, and then sent out.
        """
        events = self._get_next_event_batch()
        # if we have not received a new batch
        if not events:
            data = np.empty(self._shape_out)
            indices = np.empty(self._shape_out)
            warnings.warn("no events received")
        elif not events["data"]:
            warnings.warn()
        else:
            data, indices = encode_data_and_indices(self._frame_shape,
                                                    events)
            # If we have more data than our shape allows, subsample
            if data.shape[0] > self._shape_out[0]:
               data, indices = sub_sample(data, indices,
                                          self._shape_out[0], self._random_rng)
        self.out_port.send(data, indices)

    def _get_next_event_batch(self):
        """
        Compiles events from the event stream into batches which will be
        treated in a single timestep. Once we reach the end of the file, the
        process loops back to the start of the file.
        """
        try:
            # If end of file, raises StopIteration error.
            events = self._event_stream.__next__()
        except StopIteration:
            # TODO: define expected behavior
            raise StopIteration(f"No events received. Check that everything is well connected.")
            # return None
        return events

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
