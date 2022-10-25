import typing as ty

from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.process.process import AbstractProcess


class LearningNeuronProcess:
    """
    Base class for plastic neuron process.
    """
    def __init__(self,
                 shape: ty.Tuple[int, ...],
                 *args,
                 **kwargs):

        kwargs['shape'] = shape

        # Learning Ports
        # Port for backprop action potentials
        self.s_out_bap = OutPort(shape=(shape[0],))

        # Port for arbitrary trace using graded spikes
        self.s_out_y2 = OutPort(shape=(shape[0],))

        # Port for arbitrary trace using graded spikes
        self.s_out_y3 = OutPort(shape=(shape[0],))

        super().__init__(*args, **kwargs)
