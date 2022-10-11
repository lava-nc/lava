import typing as ty

from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.process.process import AbstractProcess

class NeuronProcess(AbstractProcess):
    def __init__(self,
                 shape: ty.Tuple[int, ...],
                 enable_learning: bool = False,
                 update_traces = None,
                 *args,
                 **kwargs):

        kwargs['enable_learning'] = enable_learning
        kwargs['shape'] = shape
        kwargs['update_traces'] = update_traces

        self.enable_learning = enable_learning

        # Learning Ports
        self.s_out_bap = OutPort(shape=(shape)) # Port for backprop action potentials
        self.s_out_y2 = OutPort(shape=(shape)) # Port for arbitrary trace using graded spikes 
        self.s_out_y3 = OutPort(shape=(shape)) # Port for arbitrary trace using graded spikes

        super().__init__(*args, **kwargs)

