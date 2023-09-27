# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import numpy as np

from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var


class Readout(AbstractProcess):
    """ Readout process of the CLP system. It receives the output spikes from
     PrototypeLIF neurons, look up the label of the winner prototype and
     send it out to the user as the inferred label.
     Additionally, if the winner neuron does not have a label this process
     assigns a pseudo-label (a negative-valued label) for the time-being.
     When a user-provided label is available, this process refer to the most
     recent predicted label. If that is a pseudo-label, then it assigns the
     user-provided label to this neuron. On the other hand if that is a
     normal label (i.e. a positive number) then the process will check the
     correctness of the predicted label and provide feedback through another
     channel

    Parameters
        ----------
        n_protos : int
            Number of Prototype LIF neurons that this process need to read from.
        proto_labels : numpy.ndarray, optional
            Initial labels of the Prototype LIF neurons. If not provided,
            by default this array will be initialized with zeros, meaning
            they are not labelled.
    """

    def __init__(self, *,
                 n_protos: int,
                 proto_labels: ty.Optional[np.ndarray] = None) -> None:
        # If not provided by the user initialize it to the zeros
        if proto_labels is None:
            proto_labels = np.zeros(shape=(n_protos,), dtype=int)

        super().__init__(proto_labels=proto_labels, n_protos=n_protos)

        self.inference_in = InPort(shape=(n_protos,))  # To read output spikes
        self.label_in = InPort(shape=(1,))  # User-provided labels goes in here
        self.user_output = OutPort(shape=(1,))  # Output for predicted labels

        # Feedback to user about correctness of the prediction
        self.feedback = OutPort(shape=(1,))
        self.trigger_alloc = OutPort(shape=(1,))

        # The array for the labels of the prototype neurons
        self.proto_labels = Var(shape=(n_protos,), init=proto_labels)

        # The id of the most recent winner prototype
        self.last_winner_id = Var(shape=(1,), init=0)


class Allocator(AbstractProcess):
    """ Allocator process of CLP system. When triggered by other processes
    it will send a one-hot-encoded allocation signal to the prototype
    population, specifically targeting next neuron to be allocated. It holds
    the reference to the id of the next neuron to be allocated.

    Parameters
        ----------
        n_protos : int
            The number of prototypes that this Allocator process can
            target. Each time a allocation trigger input is received the
            next unallocated prototype will be targeted by the output of the
            Allocator process.
    """

    def __init__(self, *,
                 n_protos: int,
                 next_alloc_id: ty.Optional[int] = 1) -> None:

        super().__init__()

        # Input for triggering allocation
        self.trigger_in = InPort(shape=(1,))
        # One-hot-encoded output for allocating specific prototype
        self.allocate_out = OutPort(shape=(1,))

        # The id of the next prototype to be allocated
        self.next_alloc_id = Var(shape=(1,), init=next_alloc_id)
        self.n_protos = Var(shape=(1,), init=n_protos)
