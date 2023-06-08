# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel

from lava.proc.clp.novelty_detector.process import NoveltyDetector


@implements(proc=NoveltyDetector, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt", 'bit_accurate_loihi')
class PyNoveltyDetectorModel(PyLoihiProcessModel):
    """Python implementation of the NoveltyDetector process

    """
    input_aval_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32)
    output_aval_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32)

    novelty_detected_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.t_wait = proc_params['t_wait']
        self.n_protos = proc_params['n_protos']
        self.waiting = False  # A variable to know if we are waiting for output
        self.t_passed = 0  # The time passed since the injection of the input
        self.next_alloc_id = 0  # The id of the next neuron to be allocated
        self.novelty_detected = False

    def run_spk(self) -> None:

        # If input is available, we start to clock for waiting the output.
        a_in = self.input_aval_in.recv()
        if a_in != 0:
            self.waiting = True
            self.t_passed = 0
            print("in_aval")

        # If output available, that means the input is a known pattern,
        # so we turn off waiting and reset
        a_in = self.output_aval_in.recv()
        if a_in != 0:
            self.waiting = False
            self.t_passed = 0
            print("out_aval")

        # If not, then we check whether the time limit has been passed for
        # waiting. If so, we assume this is a novel pattern
        elif self.t_passed > self.t_wait:
            print("Novelty detected")
            self.novelty_detected = True
            self.waiting = False
            self.t_passed = 0

        # If we are still waiting, increment the time counter
        if self.waiting:
            self.t_passed += 1

        # If we have detected novelty, send this signal downstream, and set
        # the flag back to the False
        if self.novelty_detected:
            # Choose the specific element of the OutPort to send novelty
            # signal to allocate the next neuron. We use 7-bit fixed-point
            # numbers as this value would be written into post-synaptic trace
            # in the Loihi which is also 7-bit
            alloc_signal = np.zeros(shape=self.novelty_detected_out.shape)
            alloc_signal[self.next_alloc_id] = 127  # ~1 as 7-bit number
            self.novelty_detected_out.send(alloc_signal)

            # Increment this counter to point to the next neuron
            self.next_alloc_id += 1
            self.novelty_detected = False

        else:
            # Otherwise, just send zeros (i.e. no signal)
            self.novelty_detected_out.send(
                np.zeros(shape=self.novelty_detected_out.shape))
