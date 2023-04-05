# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from scipy.sparse import spmatrix, csr_matrix
import typing as ty

from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class Sparse(AbstractProcess):
    """Sparse connections between neurons. Realizes the following abstract
    behavior: a_out = weights * s_in. The weights are stored as 
    scipy.sparse.csr_matrix.

    Parameters
    ----------
    weights : scipy.sparse.spmatrix 
        2D connection weight matrix as sparse matrix of form 
        (num_flat_output_neurons, num_flat_input_neurons).

    weight_exp : int, optional
        Shared weight exponent of base 2 used to scale magnitude of
        weights, if needed. Mostly for fixed point implementations.
        Unnecessary for floating point implementations.
        Default value is 0.

    num_weight_bits : int, optional
        Shared weight width/precision used by weight. Mostly for fixed
        point implementations. Unnecessary for floating point
        implementations.
        Default is for weights to use full 8 bit precision.

    sign_mode : SignMode, optional
        Shared indicator whether synapse is of type SignMode.NULL,
        SignMode.MIXED, SignMode.EXCITATORY, or SignMode.INHIBITORY. If
        SignMode.MIXED, the sign of the weight is
        included in the weight bits and the fixed point weight used for
        inference is scaled by 2.
        Unnecessary for floating point implementations.

        In the fixed point implementation, weights are scaled according to
        the following equations:
        w_scale = 8 - num_weight_bits + weight_exp + isMixed()
        weights = weights * (2 ** w_scale)

    num_message_bits : int, optional
        Determines whether the Sparse Process deals with the incoming
        spikes as binary spikes (num_message_bits = 0) or as graded
        spikes (num_message_bits > 0). Default is 0.
        """
    def __init__(self,
                 *,
                 weights: spmatrix,
                 name: ty.Optional[str] = None,
                 num_message_bits: ty.Optional[int] = 0,
                 log_config: ty.Optional[LogConfig] = None,
                 **kwargs) -> None:

        super().__init__(num_message_bits=num_message_bits,
                         name=name,
                         log_config=log_config,
                         **kwargs)

        # transform weights to csr matrix
        weights = weights.tocsr() 

        self._validate_weights(weights)
        shape = weights.shape

        # Ports
        self.s_in = InPort(shape=(shape[1],))
        self.a_out = OutPort(shape=(shape[0],))

        # Variables
        self.weights = Var(shape=shape, init=weights)
        self.a_buff = Var(shape=(shape[0],), init=0)
        self.num_message_bits = Var(shape=(1,), init=num_message_bits)

    @staticmethod
    def _validate_weights(weights: spmatrix) -> None:
        if len(np.shape(weights)) != 2:
            raise ValueError("Sparse Process 'weights' expects a 2D matrix, "
                             f"got {weights}.")


