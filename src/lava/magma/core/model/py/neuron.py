from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
import numpy as np

class NeuronModel(PyLoihiProcessModel):

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)

        self._shape = self.proc_params["shape"]
        self._enable_learning = self.proc_params["enable_learning"]
        self._update_traces = self.proc_params['update_traces']


class NeuronModelFixed(NeuronModel):
    # Learning Ports
    s_out_bap: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    s_out_y2: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=7)
    s_out_y3: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=7)

    def __init__(self, proc_params: dict) -> None:
         super().__init__(proc_params)

    def run_spk(self):
        if self._enable_learning and self._update_traces is not None:
            y2, y3 = self._update_traces(self)
            self.s_out_y2.send(y2)
            self.s_out_y3.send(y3)


class NeuronModelFloat(NeuronModel):
    """Floating-point implementation of the Neuron Process

    This ProcessModel constitutes a behavioral implementation of Loihi synapses
    written in Python, executing on CPU, and operating in floating-point
    arithmetic.

    To summarize the behavior:

    Spiking phase:
    run_spk:

        (1) Send activations from past time step to post-synaptic
        neuron Process.
        (2) Receive spikes from pre- and post-synaptic neuron Process.
        (3) Record within-epoch spiking times. Update traces if more than
        one spike during the epoch.
        (4) Compute activations to be sent on next time step.

    Learning phase:
    run_lrn:

        (1) Compute updates for each active synaptic variable,
        according to associated learning rule,
        based on the state of Vars representing dependencies and factors.
        (2) Update traces based on within-epoch spiking times and trace
        configuration parameters (impulse, decay).
        (3) Reset within-epoch spiking times and dependency Vars

    Note: The synaptic variable tag_2 currently DOES NOT induce synaptic
    delay in this connections Process. It can be adapted according to its
    learning rule (learned), but it will not affect synaptic activity.
    """

    # Learning Ports
    s_out_bap: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool)
    s_out_y2: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    s_out_y3: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)

    def run_spk(self):
        if self._enable_learning:
            if self._update_traces is not None:
                y2, y3 = self._update_traces(self)
                self.s_out_y2.send(y2)
                self.s_out_y3.send(y3)
            else:
                self.s_out_y2.send(np.array([0]))
                self.s_out_y3.send(np.array([0]))



