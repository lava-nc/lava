from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
import numpy as np


class PlasticNeuronModel(PyLoihiProcessModel):

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)

        self._shape = self.proc_params["shape"]


class PlasticNeuronModelFixed(PlasticNeuronModel):
    # Learning ports
    s_out_bap: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    s_out_y1: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=7)
    s_out_y2: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=7)
    s_out_y3: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=7)

    # Learning states
    y1: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=7)
    y2: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=7)
    y3: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=7)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)


class PlasticNeuronModelFloat(PlasticNeuronModel):

    # Learning Ports
    s_out_bap: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool)
    s_out_y1: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    s_out_y2: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    s_out_y3: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    # Learning states
    y1: np.ndarray = LavaPyType(np.ndarray, float)
    y2: np.ndarray = LavaPyType(np.ndarray, float)
    y3: np.ndarray = LavaPyType(np.ndarray, float)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
