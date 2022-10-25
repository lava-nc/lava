import typing as ty
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.learning.learning_rule import LoihiLearningRule
from lava.magma.core.process.variable import Var


class LearningNeuronProcess:
    """
    Base class for plastic neuron processes.

    Parameters
    ==========

    learning_rule: Optional[LoihiLearningRule]
        Learning rule which determines the parameters for online learning.

    """
    def __init__(self,
                 shape: ty.Tuple[int, ...],
                 learning_rule: ty.Optional[LoihiLearningRule] = None,
                 *args,
                 **kwargs):

        kwargs['shape'] = shape
        kwargs["learning_rule"] = learning_rule

        self.learning_rule = learning_rule

        # Learning Ports
        # Port for backprop action potentials
        self.s_out_bap = OutPort(shape=(shape[0],))

        # Port for arbitrary trace using graded spikes
        self.s_out_y1 = OutPort(shape=(shape[0],))

        # Port for arbitrary trace using graded spikes
        self.s_out_y2 = OutPort(shape=(shape[0],))

        # Port for arbitrary trace using graded spikes
        self.s_out_y3 = OutPort(shape=(shape[0],))

        # Learning related states
        self.y1 = Var(shape=shape, init=0)
        self.y2 = Var(shape=shape, init=0)
        self.y3 = Var(shape=shape, init=0)

        super().__init__(*args, **kwargs)
