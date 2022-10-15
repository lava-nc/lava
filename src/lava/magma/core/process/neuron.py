import typing as ty

from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.process.process import AbstractProcess


class PlasticNeuronProcess:
    def __init__(self,
                 shape: ty.Tuple[int, ...],
                 *args,
                 **kwargs):

        kwargs['shape'] = shape

        self.enable_learning = enable_learning

        # Learning Ports
        # Port for backprop action potentials
        self.s_out_bap = OutPort(shape=(shape[0],))

        # Port for arbitrary trace using graded spikes
        self.s_out_y2 = OutPort(shape=(shape[0],))

        # Port for arbitrary trace using graded spikes
        self.s_out_y3 = OutPort(shape=(shape[0],))

        super().__init__(*args, **kwargs)
