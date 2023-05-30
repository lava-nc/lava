# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel

from lava.proc.clp.nsm.process import Readout


@implements(proc=Readout, protocol=LoihiProtocol)
@requires(CPU)
class PyReadoutModel(PyLoihiProcessModel):
    """Python implementation of the Readout process.
    This process will run in super host and will be the main interface
    process with the user.
    """
    inference_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32,
                                        precision=24)
    label_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32)

    user_output: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
    proto_labels: np.ndarray = LavaPyType(np.ndarray, int)
    last_winner_id: int = LavaPyType(np.ndarray, int)
    feedback: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    def run_spk(self):
        # Read the output of the prototype neurons
        output_vec = self.inference_in.recv()

        # Read the user-provided label
        user_label = self.label_in.recv()[0]

        # Feedback about the correctness of prediction. +1 if correct,
        # -1 if incorrect, 0 if no label is provided by the user at this point.
        infer_check = 0

        # If there is an active prototype neuron, this will temporarily store
        # the label of that neuron
        inferred_label = 0

        # If any prototype neuron is active, then we go here. We assume there
        # is only one neuron active in the prototype population
        if output_vec.any():

            print("Output available:", output_vec)

            # Find the id of the winner neuron and store it
            winner_proto_id = np.nonzero(output_vec)[0][0]
            self.last_winner_id = winner_proto_id

            # Get the label of this neuron from the labels' list
            inferred_label = self.proto_labels[winner_proto_id]

            # If this label is zero, that means this prototype is not labeled.
            if inferred_label == 0:
                # So, we give a pseudo label to the unknown winner.
                # These are negative temporary labels that is based on the id
                # of the prototype and generated as follows.
                self.proto_labels[winner_proto_id] = -1 * (winner_proto_id + 1)

                # So now this pseudo-label is our inferred label.
                inferred_label = self.proto_labels[winner_proto_id]

            print("Inferred label:", inferred_label)

        # Next we check if a user-provided label is available.
        if user_label != 0:

            # If so we need to access the most recent winner's label,
            # assuming the temporal causality between the prediction by the
            # system and the providence of the label;l by the user
            last_inferred_label = self.proto_labels[self.last_winner_id]
            print("Last Inferred label:", last_inferred_label)

            # If the most recently predicted label (i.e. the one for the
            # current input which is also the user-provided label refer to)
            # is an actual label (not a pseudo one), then we check the
            # correctness of the predicted label against user-provided one.

            if last_inferred_label > 0:  # "Known Known class"
                if last_inferred_label == user_label:
                    print("Correct")
                    infer_check = 1
                else:
                    print("False")
                    infer_check = -1

            # If this prototpye has a pseudo-label, then we label it with
            # the user-provided label and do not send any feedback (because
            # we did not have an actual prediction)

            elif last_inferred_label < 0:  # "Known Unknown class"
                self.proto_labels[self.last_winner_id] = user_label
                inferred_label = user_label

        # Send out the readout predicted label (if any) and the feedback
        # about the correctness of this prediction after user providing the
        # actual label
        self.user_output.send(np.array([inferred_label]))
        self.feedback.send(np.array([infer_check]))
