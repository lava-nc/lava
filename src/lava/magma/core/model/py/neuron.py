# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort, PyInPort
from lava.magma.core.model.py.type import LavaPyType
import numpy as np


class LearningNeuronModel(PyLoihiProcessModel):
    """Base class for learning enables neuron models.

    Implements ports and vars used by learning enabled neurons. Must be
    inherited by floating point and fixed point implementations.

    Parameters
    ==========

    proc_params: dict
        Process parameters from the neuron model.

    """

    # Learning Ports
    a_third_factor_in = None
    s_out_bap = None
    s_out_y1 = None
    s_out_y2 = None
    s_out_y3 = None

    # Learning states
    y1 = None
    y2 = None
    y3 = None

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)

        self._shape = self.proc_params["shape"]
        self._learning_rule = self.proc_params["learning_rule"]


class LearningNeuronModelFixed(LearningNeuronModel):
    """Base class for learning enables neuron models.

    Implements ports and vars used by learning enabled neurons for fixed
    point implementations.

    Parameters
    ==========

    proc_params: dict
        Process parameters from the neuron model.

    """

    # Learning Ports
    a_third_factor_in: PyInPort = LavaPyType(
        PyInPort.VEC_DENSE, np.int32, precision=7
    )

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


class LearningNeuronModelFloat(LearningNeuronModel):
    """Base class for learning enables neuron models.

    Implements ports and vars used by learning enabled neurons for floating
    point implementations.

    Parameters
    ==========

    proc_params: dict
        Process parameters from the neuron model.

    """

    # Learning Ports
    a_third_factor_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

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
