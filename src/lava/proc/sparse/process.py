# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from scipy.sparse import spmatrix, csr_matrix
import typing as ty

from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.connection import LearningConnectionProcess
from lava.magma.core.learning.constants import GradedSpikeCfg
from lava.magma.core.learning.learning_rule import LoihiLearningRule


class Sparse(AbstractProcess):
    """Sparse connections between neurons. Realizes the following abstract
    behavior: a_out = weights * s_in. The weights are stored as
    scipy.sparse.csr_matrix.

    Parameters
    ----------
    weights : scipy.sparse.spmatrix or np.ndarray
        2D connection weight matrix of form
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
                 weights: ty.Union[spmatrix, np.ndarray],
                 name: ty.Optional[str] = None,
                 num_message_bits: ty.Optional[int] = 0,
                 log_config: ty.Optional[LogConfig] = None,
                 **kwargs) -> None:

        super().__init__(num_message_bits=num_message_bits,
                         name=name,
                         log_config=log_config,
                         **kwargs)

        weights = self._create_csr_matrix(weights)
        shape = weights.shape

        # Ports
        self.s_in = InPort(shape=(shape[1],))
        self.a_out = OutPort(shape=(shape[0],))

        # Variables
        self.weights = Var(shape=shape, init=weights.copy())
        self.a_buff = Var(shape=(shape[0],), init=0)
        self.num_message_bits = Var(shape=(1,), init=num_message_bits)

    @staticmethod
    def _create_csr_matrix(matrix):
        # Transform weights to csr matrix
        if isinstance(matrix, np.ndarray):
            matrix = csr_matrix(matrix)
        else:
            matrix = matrix.tocsr()
        return matrix


class LearningSparse(LearningConnectionProcess, Sparse):
    """Sparse connections between neurons. Realizes the following abstract
    behavior: a_out = weights * s_in. The weights are stored as
    scipy.sparse.csr_matrix.

    Parameters
    ----------
    weights : scipy.sparse.spmatrix or np.ndarray
        2D connection weight matrix of form
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

    learning_rule: LoihiLearningRule
        Learning rule which determines the parameters for online learning.

    graded_spike_cfg: GradedSpikeCfg
        Indicates how to use incoming graded spike to update pre-synaptic
        traces.

        (0) GradedSpikeCfg.USE_REGULAR_IMPULSE interprets the spike as a
        binary spike, adds regular impulses to pre-synaptic traces, at the end
        of the epoch.
        (1) GradedSpikeCfg.OVERWRITE interprets the spike as a graded spike,
        overwrites the value of the pre-synaptic trace x1 by payload/2,
        upon spiking.
        (2) GradedSpikeCfg.ADD_WITH_SATURATION interprets the spike as a graded
        spike, adds payload/2 to the pre-synaptic trace x1, upon spiking,
        saturates x1 to 127 (fixed-pt/hw only).
        (3) GradedSpikeCfg.ADD_WITHOUT_SATURATION interprets the spike as a
        graded spike, adds payload/2 to the pre-synaptic trace x1, upon spiking,
        keeps only overflow above 127 in x1 (fixed-pt/hw only), adds regular
        impulse to x2 on overflow.
        In addition, only pre-synaptic graded spikes that trigger overflow in
        x1 and regular impulse addition to x2 will be considered by the
        learning rule Products conditioned on x0.
        """

    def __init__(self,
                 *,
                 weights: ty.Union[spmatrix, np.ndarray],
                 tag_2: ty.Optional[ty.Union[spmatrix, np.ndarray]] = None,
                 tag_1: ty.Optional[ty.Union[spmatrix, np.ndarray]] = None,
                 name: ty.Optional[str] = None,
                 num_message_bits: ty.Optional[int] = 0,
                 log_config: ty.Optional[LogConfig] = None,
                 learning_rule: LoihiLearningRule = None,
                 graded_spike_cfg: GradedSpikeCfg =
                 GradedSpikeCfg.USE_REGULAR_IMPULSE,
                 **kwargs) -> None:

        if graded_spike_cfg != GradedSpikeCfg.USE_REGULAR_IMPULSE:
            learning_rule.x1_impulse = 0

        super().__init__(weights=weights,
                         tag_2=tag_2,
                         tag_1=tag_1,
                         shape=weights.shape,
                         num_message_bits=num_message_bits,
                         name=name,
                         log_config=log_config,
                         learning_rule=learning_rule,
                         graded_spike_cfg=graded_spike_cfg,
                         **kwargs)

        if tag_2 is None:
            tag_2 = np.zeros(weights.shape)

        if tag_1 is None:
            tag_1 = np.zeros(weights.shape)

        tag_2 = self._create_csr_matrix(tag_2)
        tag_1 = self._create_csr_matrix(tag_1)

        self.tag_2.init = tag_2.copy()
        self.tag_1.init = tag_1.copy()

        self.proc_params["x_idx_active_syn_vars"] = {
            "weights": weights.nonzero()[1],
            "tag_2": tag_2.nonzero()[1],
            "tag_1": tag_1.nonzero()[1]
        }


class DelaySparse(Sparse):
    def __init__(self,
                 *,
                 weights: ty.Union[spmatrix, np.ndarray],
                 delays: ty.Union[spmatrix, int],
                 max_delay: ty.Optional[int] = 0,
                 name: ty.Optional[str] = None,
                 num_message_bits: ty.Optional[int] = 0,
                 log_config: ty.Optional[LogConfig] = None,
                 **kwargs) -> None:
        """Sparse, delayed connections between neurons. Realizes the following
        abstract behavior: a_out = weights * s_in

        Parameters
        ----------
        weights : scipy.sparse.spmatrix or np.ndarray
            2D connection weight matrix of form (num_flat_output_neurons,
            num_flat_input_neurons) in C-order (row major).

        delays : spmatrix, int
            2D connection delay matrix of form (num_flat_output_neurons,
            num_flat_input_neurons) in C-order (row major) or integer value if
            the same delay should be used for all synapses.

        max_delay: int, optional
            Maximum expected delay. Should be set if delays change during
            execution. Default value is 0, in this case the maximum delay
            will be determined from the values given in 'delays'.

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
        if max_delay == 0:
            max_delay = int(np.max(delays))

        super().__init__(weights=weights,
                         num_message_bits=num_message_bits,
                         name=name,
                         log_config=log_config,
                         max_delay=max_delay,
                         **kwargs)

        self._validate_delays(weights, delays)
        shape = weights.shape

        # Variables
        self.delays = Var(shape=shape, init=delays)
        self.a_buff = Var(shape=(shape[0], max_delay + 1), init=0)

    @staticmethod
    def _validate_delays(weights: spmatrix, delays: spmatrix) -> None:
        if np.min(delays) < 0:
            raise ValueError("DelaySparse Process 'delays' expects only "
                             f"positive values, got {delays}.")
        if not isinstance(delays, int):
            if weights.shape != delays.shape:
                raise ValueError("DelaySparse Process 'delays' expects same "
                                 f"shape as the weight matrix or int, got "
                                 f"{delays}.")
            if delays.dtype != int:
                raise ValueError("DelaySparse Process 'delays' expects integer "
                                 f"value(s), got {delays}.")
