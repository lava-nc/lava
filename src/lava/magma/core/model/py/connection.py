# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.core.learning.learning_rule_applier import (
    LearningRuleApplierFloat,
    LearningRuleApplierBitApprox,
)
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.model.py.type import LavaPyType

from lava.magma.core.learning.constants import *
from lava.magma.core.learning.random import TraceRandom, ConnVarRandom
import lava.magma.core.learning.string_symbols as str_symbols
from lava.utils.weightutils import SignMode
from lava.magma.core.learning.utils import saturate, stochastic_round
import numpy as np
import typing

NUM_DEPENDENCIES = len(str_symbols.DEPENDENCIES)
NUM_X_TRACES = len(str_symbols.PRE_TRACES)
NUM_Y_TRACES = len(str_symbols.POST_TRACES)


class Connection(PyLoihiProcessModel):

    # Learning Ports
    s_in_bap = None

    # Learning Vars
    x0 = None
    tx = None
    x1 = None
    x2 = None

    y0 = None
    ty = None
    y1 = None
    y2 = None
    y3 = None

    tag_2 = None
    tag_1 = None

    def _store_shapes(self) -> None:
        """Build and store several shapes that are needed in several
        computation stages of this ProcessModel."""
        num_pre_neurons = self._shape[1]
        num_post_neurons = self._shape[0]

        # Shape: (2, num_pre_neurons)
        self._shape_x_traces = (NUM_X_TRACES, num_pre_neurons)
        # Shape: (3, 2, num_post_neurons, num_pre_neurons)
        self._shape_x_traces_per_dep_broad = (
            NUM_DEPENDENCIES,
            NUM_X_TRACES,
            num_post_neurons,
            num_pre_neurons,
        )

        # Shape: (3, num_post_neurons)
        self._shape_y_traces = (NUM_Y_TRACES, num_post_neurons)
        # Shape: (3, 3, num_post_neurons, num_pre_neurons)
        self._shape_y_traces_per_dep_broad = (
            NUM_DEPENDENCIES,
            NUM_Y_TRACES,
            num_post_neurons,
            num_pre_neurons,
        )

        # Shape: (3, 5, num_post_neurons, num_pre_neurons)
        self._shape_traces_per_dep_broad = (
            NUM_DEPENDENCIES,
            NUM_X_TRACES + NUM_Y_TRACES,
            num_post_neurons,
            num_pre_neurons,
        )

    def _build_active_traces_per_dependency(self) -> None:
        """Build and store boolean numpy arrays specifying which x and y
        traces are active, per dependency.

        First dimension:
        index 0 -> x0 dependency
        index 1 -> y0 dependency
        index 2 -> u dependency

        Second dimension (for x_traces):
        index 0 -> x1 trace
        index 1 -> x2 trace

        Second dimension (for y_traces):
        index 0 -> y1 trace
        index 1 -> y2 trace
        index 2 -> y3 trace
        """
        # Shape : (3, 5)
        active_traces_per_dependency = np.zeros(
            (
                len(str_symbols.DEPENDENCIES),
                len(str_symbols.PRE_TRACES) + len(str_symbols.POST_TRACES),
            ),
            dtype=bool,
        )
        for (
            dependency,
            traces,
        ) in self._learning_rule.active_traces_per_dependency.items():
            if dependency == str_symbols.X0:
                dependency_idx = 0
            elif dependency == str_symbols.Y0:
                dependency_idx = 1
            elif dependency == str_symbols.U:
                dependency_idx = 2
            else:
                raise ValueError("Unknown Dependency in ProcessModel.")

            for trace in traces:
                if trace == str_symbols.X1:
                    trace_idx = 0
                elif trace == str_symbols.X2:
                    trace_idx = 1
                elif trace == str_symbols.Y1:
                    trace_idx = 2
                elif trace == str_symbols.Y2:
                    trace_idx = 3
                elif trace == str_symbols.Y3:
                    trace_idx = 4
                else:
                    raise ValueError("Unknown Trace in ProcessModel")

                active_traces_per_dependency[dependency_idx, trace_idx] = True

        # Shape : (3, 2)
        self._active_x_traces_per_dependency = active_traces_per_dependency[
            :, :2
        ]

        # Shape : (3, 3)
        self._active_y_traces_per_dependency = active_traces_per_dependency[
            :, 2:
        ]

    def _build_active_traces(self) -> None:
        """Build and store boolean numpy arrays specifying which x and y
        traces are active."""
        # Shape : (2, )
        self._active_x_traces = np.logical_or(
            self._active_x_traces_per_dependency[0],
            self._active_x_traces_per_dependency[1],
            self._active_x_traces_per_dependency[2],
        )

        # Shape : (3, )
        self._active_y_traces = np.logical_or(
            self._active_y_traces_per_dependency[0],
            self._active_y_traces_per_dependency[1],
            self._active_y_traces_per_dependency[2],
        )

    # Shape: (2, num_pre_neurons)
    @property
    def _x_traces(self) -> np.ndarray:
        """Get x traces.

        Returns
        ----------
        x_traces : np.ndarray
            X traces.
        """
        return np.concatenate(
            (self.x1[np.newaxis, :], self.x2[np.newaxis, :]), axis=0
        )

    def _set_x_traces(self, x_traces: np.ndarray) -> None:
        """Set x traces.

        Parameters
        ----------
        x_traces : np.ndarray
            X traces.
        """
        self.x1 = x_traces[0]
        self.x2 = x_traces[1]

    # Shape: (3, num_post_neurons)
    @property
    def _y_traces(self) -> np.ndarray:
        """Get y traces.

        Returns
        ----------
        y_traces : np.ndarray
            Y traces.
        """
        return np.concatenate(
            (
                self.y1[np.newaxis, :],
                self.y2[np.newaxis, :],
                self.y3[np.newaxis, :],
            ),
            axis=0,
        )

    def _set_y_traces(self, y_traces: np.ndarray) -> None:
        """Set y traces.

        Parameters
        ----------
        y_traces : np.ndarray
            Y traces.
        """
        self.y1 = y_traces[0]
        self.y2 = y_traces[1]
        self.y3 = y_traces[2]

    def _within_epoch_time_step(self) -> int:
        """Compute index of current time step within the epoch.

        Result ranges from 1 to t_epoch.

        Returns
        ----------
        within_epoch_ts : int
            Within-epoch time step.
        """
        within_epoch_ts = self.time_step % self._learning_rule.t_epoch

        if within_epoch_ts == 0:
            within_epoch_ts = self._learning_rule.t_epoch

        return within_epoch_ts

    def _reset_dependencies_and_spike_times(self) -> None:
        """Reset all dependencies and within-epoch spike times."""
        self.x0 = np.zeros_like(self.x0)
        self.y0 = np.zeros_like(self.y0)

        self.tx = np.zeros_like(self.tx)
        self.ty = np.zeros_like(self.ty)


class ConnectionModelBitApproximate(Connection):
    """Fixed-point, bit-approximate implementation of the Connection base
    class.

    This class implements the learning simulation with integer and fixed
    point arithmetic but does not implement the exact behavior of Loihi.
    Nevertheless, the results are comparable to those by Loihi.

    To summarize the behavior:

    Spiking phase:
    run_spk:

        (1) (Dense) Send activations from past time step to post-synaptic
        neuron Process.
        (2) (Dense) Compute activations to be sent on next time step.
        (3) (Dense) Receive spikes from pre-synaptic neuron Process.
        (4) (Dense) Record within-epoch pre-synaptic spiking time.
        Update pre-synaptic traces if more than one spike during the epoch.
        (5) Receive spikes from post-synaptic neuron Process.
        (6) Record within-epoch pre-synaptic spiking time.
        Update pre-synaptic traces if more than one spike during the epoch.
        (7) Advance trace random generators.

    Learning phase:
    run_lrn:

        (1) Advance synaptic variable random generators.
        (2) Compute updates for each active synaptic variable,
        according to associated learning rule,
        based on the state of Vars representing dependencies and factors.
        (3) Update traces based on within-epoch spiking times and trace
        configuration parameters (impulse, decay).
        (4) Reset within-epoch spiking times and dependency Vars

    Note: The synaptic variable tag_2 currently DOES NOT induce synaptic
    delay in this connections Process. It can be adapted according to its
    learning rule (learned), but it will not affect synaptic activity.

    Parameters
    ----------
    proc_params: dict
        Parameters from the ProcessModel
    """

    # Learning Ports
    s_in_bap: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)

    # Learning Vars
    x0: np.ndarray = LavaPyType(np.ndarray, bool)
    tx: np.ndarray = LavaPyType(np.ndarray, int, precision=6)
    x1: np.ndarray = LavaPyType(np.ndarray, int, precision=7)
    x2: np.ndarray = LavaPyType(np.ndarray, int, precision=7)

    y0: np.ndarray = LavaPyType(np.ndarray, bool)
    ty: np.ndarray = LavaPyType(np.ndarray, int, precision=6)
    y1: np.ndarray = LavaPyType(np.ndarray, int, precision=7)
    y2: np.ndarray = LavaPyType(np.ndarray, int, precision=7)
    y3: np.ndarray = LavaPyType(np.ndarray, int, precision=7)

    tag_2: np.ndarray = LavaPyType(np.ndarray, int, precision=6)
    tag_1: np.ndarray = LavaPyType(np.ndarray, int, precision=8)

    def __init__(self, proc_params: dict) -> None:
        # If there is a plasticity rule attached to the corresponding process,
        # add all necessary ports get access to all learning params
        self._learning_rule = proc_params["learning_rule"]
        self._shape = proc_params["shape"]

        # Flag to determine whether weights have already been scaled.
        self.weights_set = False

        if self._learning_rule is not None:
            # store shapes that useful throught the lifetime of this PM
            self._store_shapes()
            # store impulses and taus in ndarrays with the right shapes
            self._store_impulses_and_taus()

            # store active traces per dependency from learning_rule in ndarrays
            # with the right shapes
            self._build_active_traces_per_dependency()
            # store active traces from learning_rule in ndarrays
            # with the right shapes
            self._build_active_traces()
            # generate LearningRuleApplierBitApprox from ProductSeries
            self._build_learning_rule_appliers()

            # initialize TraceRandoms and ConnVarRandom
            self._init_randoms()

        super().__init__(proc_params)

    @staticmethod
    def _decompose_impulses(
        impulses: np.ndarray,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Decompose float impulse values into integer and fractional parts.

        Parameters
        ----------
        impulses : ndarray
            Impulse values.

        Returns
        ----------
        impulses_int : int
            Impulse integer values.
        impulses_frac : int
            Impulse fractional values.
        """
        impulses_int = np.floor(impulses)
        impulses_frac = np.round(
            (impulses - impulses_int) * 2**W_TRACE_FRACTIONAL_PART
        )

        return impulses_int.astype(int), impulses_frac.astype(int)

    def _store_impulses_and_taus(self) -> None:
        """Build and store integer ndarrays representing x and y
        impulses and taus."""
        x_impulses = np.array(
            [self._learning_rule.x1_impulse, self._learning_rule.x2_impulse]
        )
        self._x_impulses_int, self._x_impulses_frac = self._decompose_impulses(
            x_impulses
        )
        self._x_taus = np.array(
            [self._learning_rule.x1_tau, self._learning_rule.x2_tau]
        )

        y_impulses = np.array(
            [
                self._learning_rule.y1_impulse,
                self._learning_rule.y2_impulse,
                self._learning_rule.y3_impulse,
            ]
        )
        self._y_impulses_int, self._y_impulses_frac = self._decompose_impulses(
            y_impulses
        )
        self._y_taus = np.array(
            [
                self._learning_rule.y1_tau,
                self._learning_rule.y2_tau,
                self._learning_rule.y3_tau,
            ]
        )

    def _build_learning_rule_appliers(self) -> None:
        """Build and store LearningRuleApplierBitApprox for each active learning
        rule in a dict mapped by the learning rule's target."""
        self._learning_rule_appliers = {
            str_symbols.SYNAPTIC_VARIABLE_VAR_MAPPING[
                target[1:]
            ]: LearningRuleApplierBitApprox(ps)
            for target, ps in self._learning_rule.active_product_series.items()
        }

    def _init_randoms(self) -> None:
        """Initialize trace and synaptic variable random generators."""
        self._x_random = TraceRandom(
            seed_trace_decay=self._learning_rule._rng_seed,
            seed_impulse_addition=self._learning_rule._rng_seed + 1,
        )

        self._y_random = TraceRandom(
            seed_trace_decay=self._learning_rule._rng_seed + 2,
            seed_impulse_addition=self._learning_rule._rng_seed + 3,
        )

        self._conn_var_random = ConnVarRandom()

    def run_spk(self) -> None:
        s_in_bap = self.s_in_bap.recv().astype(bool)
        if self._learning_rule is not None:
            self._record_post_spike_times(s_in_bap)
            self._update_trace_randoms()

    def _record_pre_spike_times(self, s_in: np.ndarray) -> None:
        """Record within-epoch spiking times of pre- and post-synaptic neurons.

        If more a single pre- or post-synaptic neuron spikes more than once,
        the corresponding trace is updated by its trace impulse value.

        Parameters
        ----------
        s_in : ndarray
            Pre-synaptic spikes.
        """
        self.x0[s_in] = True
        multi_spike_x = np.logical_and(self.tx > 0, s_in)

        x_traces = self._x_traces
        x_traces[:, multi_spike_x] = self._add_impulse(
            x_traces[:, multi_spike_x],
            self._x_random.random_impulse_addition,
            self._x_impulses_int[:, np.newaxis],
            self._x_impulses_frac[:, np.newaxis],
        )
        self._set_x_traces(x_traces)

        ts_offset = self._within_epoch_time_step()
        self.tx[s_in] = ts_offset

    def _record_post_spike_times(self, s_in_bap: np.ndarray) -> None:
        """Record within-epoch spiking times of pre- and post-synaptic neurons.

        If more a single pre- or post-synaptic neuron spikes more than once,
        the corresponding trace is updated by its trace impulse value.

        Parameters
        ----------
        s_in_bap : ndarray
            Post-synaptic spikes.
        """
        self.y0[s_in_bap] = True
        multi_spike_y = np.logical_and(self.ty > 0, s_in_bap)

        y_traces = self._y_traces
        y_traces[:, multi_spike_y] = self._add_impulse(
            y_traces[:, multi_spike_y],
            self._y_random.random_impulse_addition,
            self._y_impulses_int[:, np.newaxis],
            self._y_impulses_frac[:, np.newaxis],
        )
        self._set_y_traces(y_traces)

        ts_offset = self._within_epoch_time_step()
        self.ty[s_in_bap] = ts_offset

    def lrn_guard(self) -> bool:
        if self._learning_rule is not None:
            return self.time_step % self._learning_rule.t_epoch == 0
        return 0

    def run_lrn(self) -> None:
        self._update_synaptic_variable_random()
        self._apply_learning_rules()
        self._update_traces()
        self._reset_dependencies_and_spike_times()

    def _update_trace_randoms(self) -> None:
        """Update trace random generators."""
        self._x_random.advance()
        self._y_random.advance()

    def _update_synaptic_variable_random(self) -> None:
        """Update synaptic variable random generators."""
        self._conn_var_random.advance()

    def _extract_applier_args(self) -> typing.Dict[str, np.ndarray]:
        """Extracts arguments for the LearningRuleApplierFloat.

        "shape" is a tuple, shape of this Connection Process.
        "u" is a scalar.
        "np" is a reference to numpy as it is needed for the evaluation of
        "np.sign()" types of call inside the applier string.

        Shapes of numpy array args:
        "x0": (1, num_neurons_pre)
        "y0": (num_neurons_post, 1)
        "weights":  (num_neurons_post, num_neurons_pre)
        "tag_2": (num_neurons_post, num_neurons_pre)
        "tag_1": (num_neurons_post, num_neurons_pre)
        "x_traces": (3, 2, num_neurons_post, num_neurons_pre)
        "y_traces": (3, 2, num_neurons_post, num_neurons_pre)

        "x_traces" is of shape (3, 2, num_neurons_post, num_neurons_pre) with:
        First dimension representing the within-epoch time step at which the
        trace is evaluated (tx, ty, t_epoch).
        Second dimension representing the trace that is evaluated
        (x1, x2).

        "y_traces" is of shape (3, 3, num_neurons_post, num_neurons_pre) with:
        First dimension representing the within-epoch time step at which the
        trace is evaluated (tx, ty, t_epoch).
        Second dimension representing the trace that is evaluated
        (y1, y2, y3).
        """
        # Shape: (3, 2, num_post_neurons, num_pre_neurons)
        # Shape of active_x_traces_per_dependency : (3, 2) ->
        # (3, 2, 1, 1)
        active_x_traces_per_dep_broad = np.broadcast_to(
            self._active_x_traces_per_dependency[:, :, np.newaxis, np.newaxis],
            self._shape_x_traces_per_dep_broad,
        )

        # Shape: (3, 3, num_post_neurons, num_pre_neurons)
        # Shape of active_y_traces_per_dependency : (3, 3) ->
        # (3, 3, 1, 1)
        active_y_traces_per_dep_broad = np.broadcast_to(
            self._active_y_traces_per_dependency[:, :, np.newaxis, np.newaxis],
            self._shape_y_traces_per_dep_broad,
        )

        # Shape x0: (num_pre_neurons, ) -> (1, num_pre_neurons)
        # Shape y0: (num_post_neurons, ) -> (num_post_neurons, 1)
        # Shape weights: (num_post_neurons, num_pre_neurons)
        # Shape tag_2: (num_post_neurons, num_pre_neurons)
        # Shape tag_1: (num_post_neurons, num_pre_neurons)
        applier_args = {
            "shape": self._shape,
            "x0": np.broadcast_to(self.x0[np.newaxis, :], self._shape),
            "y0": np.broadcast_to(self.y0[:, np.newaxis], self._shape),
            "weights": self.weights,
            "tag_2": self.tag_2,
            "tag_1": self.tag_1,
            "u": 0,
        }

        if self._learning_rule.decimate_exponent is not None:
            k = self._learning_rule.decimate_exponent
            u = (
                1
                if int(self.time_step / self._learning_rule.t_epoch) % 2 ^ k
                == 0
                else 0
            )

            # Shape: (0, )
            applier_args["u"] = u

        # Shape: (3, 2, num_post_neurons, num_pre_neurons)
        # Shape of tx : (num_pre_neurons, ) ->
        # (1, 1, 1, num_pre_neurons)
        t_spikes_x = np.where(
            active_x_traces_per_dep_broad,
            self.tx[np.newaxis, np.newaxis, np.newaxis, :],
            0,
        )
        # Shape: (3, 3, num_post_neurons, num_pre_neurons)
        # Shape of ty : (num_post_neurons, ) ->
        # (1, 1, num_post_neurons, 1)
        t_spikes_y = np.where(
            active_y_traces_per_dep_broad,
            self.ty[np.newaxis, np.newaxis, :, np.newaxis],
            0,
        )

        # Shape: (3, 5, num_post_neurons, num_pre_neurons)
        # Shape of t_eval[0, :, :, :] : (5, num_post_neurons, num_pre_neurons)
        # Shape of tx : (num_pre_neurons, ) ->
        # (1, 1, num_pre_neurons)
        # Shape of ty : (num_post_neurons, ) ->
        # (1, num_post_neurons, 1)
        t_eval = np.zeros(self._shape_traces_per_dep_broad, dtype=int)
        t_eval[0, :, :, :] = self.tx[np.newaxis, np.newaxis, :]
        t_eval[1, :, :, :] = self.ty[np.newaxis, :, np.newaxis]
        t_eval[2, :, :, :] = self._learning_rule.t_epoch

        # Shape: (3, 2, num_post_neurons, num_pre_neurons)
        # Shape of _x_traces: (2, num_pre_neurons) ->
        # (1, 2, 1, num_pre_neurons)
        x_traces = np.where(
            active_x_traces_per_dep_broad,
            self._x_traces[np.newaxis, :, np.newaxis, :],
            0,
        )
        # Shape: (3, 3, num_post_neurons, num_pre_neurons)
        # Shape of _y_traces: (3, num_post_neurons) ->
        # (1, 3, 1, num_post_neurons)
        y_traces = np.where(
            active_y_traces_per_dep_broad,
            self._y_traces[np.newaxis, :, :, np.newaxis],
            0,
        )

        # Shape: (3, 2, num_post_neurons, num_pre_neurons)
        # Shape of t_spikes_x: (3, 2, num_post_neurons, num_pre_neurons)
        # Shape of t_eval: (3, 2, num_post_neurons, num_pre_neurons)
        # Shape of x_impulses_int: (1, 2, 1, 1)
        # Shape of x_impulses_frac: (1, 2, 1, 1)
        # Shape of x_taus: (1, 2, 1, 1)
        evaluated_x_traces = self._evaluate_trace(
            x_traces,
            t_spikes_x,
            t_eval[:, :2, :, :],
            self._x_impulses_int[np.newaxis, :, np.newaxis, np.newaxis],
            self._x_impulses_frac[np.newaxis, :, np.newaxis, np.newaxis],
            self._x_taus[np.newaxis, :, np.newaxis, np.newaxis],
            self._x_random,
        )
        # Shape: (3, 3, num_post_neurons, num_pre_neurons)
        # Shape of t_spikes_y: (3, 3, num_post_neurons, num_pre_neurons)
        # Shape of t_eval: (3, 3, num_post_neurons, num_pre_neurons)
        # Shape of y_impulses_int: (1, 3, 1, 1)
        # Shape of y_impulses_frac: (1, 3, 1, 1)
        # Shape of y_taus: (1, 3, 1, 1)
        evaluated_y_traces = self._evaluate_trace(
            y_traces,
            t_spikes_y,
            t_eval[:, 2:, :, :],
            self._y_impulses_int[np.newaxis, :, np.newaxis, np.newaxis],
            self._y_impulses_frac[np.newaxis, :, np.newaxis, np.newaxis],
            self._y_taus[np.newaxis, :, np.newaxis, np.newaxis],
            self._y_random,
        )

        # Shape: (3, 2, num_post_neurons, num_pre_neurons)
        applier_args["x_traces"] = evaluated_x_traces
        # Shape: (3, 3, num_post_neurons, num_pre_neurons)
        applier_args["y_traces"] = evaluated_y_traces

        return applier_args

    @staticmethod
    def _stochastic_round_synaptic_variable(
        synaptic_variable_name: str,
        synaptic_variable_values: np.ndarray,
        random: float,
    ) -> np.ndarray:
        """Stochastically round synaptic variable after learning rule
        application.

        Parameters
        ----------
        synaptic_variable_name: str
            Synaptic variable name.
        synaptic_variable_values: ndarray
            Synaptic variable values to stochastically round.

        Returns
        ----------
        result : ndarray
            Stochastically rounded synaptic variable values.
        """
        exp_mant = 2 ** (W_ACCUMULATOR_U - W_SYN_VAR_U[synaptic_variable_name])

        integer_part = synaptic_variable_values / exp_mant
        fractional_part = integer_part % 1

        integer_part = np.floor(integer_part)
        integer_part = stochastic_round(integer_part, random, fractional_part)
        result = (integer_part * exp_mant).astype(
            synaptic_variable_values.dtype
        )

        return result

    def _saturate_synaptic_variable_accumulator(
        self, synaptic_variable_name: str, synaptic_variable_values: np.ndarray
    ) -> np.ndarray:
        """Saturate synaptic variable accumulator.

        Checks that sign is valid.

        Parameters
        ----------
        synaptic_variable_name: str
            Synaptic variable name.
        synaptic_variable_values: ndarray
            Synaptic variable values to saturate.

        Returns
        ----------
        result : ndarray
            Saturated synaptic variable values.
        """
        # Weights
        if synaptic_variable_name == "weights":
            if np.equal(self.sign_mode, SignMode.MIXED.value):
                return synaptic_variable_values
            elif np.equal(self.sign_mode, SignMode.EXCITATORY.value):
                return np.maximum(0, synaptic_variable_values)
            elif np.equal(self.sign_mode, SignMode.INHIBITORY.value):
                return np.minimum(0, synaptic_variable_values)
        # Delays
        elif synaptic_variable_name == "tag_2":
            return np.maximum(0, synaptic_variable_values)
        # Tags
        elif synaptic_variable_name == "tag_1":
            return synaptic_variable_values
        else:
            raise ValueError(
                "Invalid synaptic_variable_name in "
                "saturate_synaptic_variable"
            )

    def _saturate_synaptic_variable(
        self, synaptic_variable_name: str, synaptic_variable_values: np.ndarray
    ) -> np.ndarray:
        """Saturate synaptic variable.

        Checks that synaptic variable values is between bounds set by
        the hardware.

        Parameters
        ----------
        synaptic_variable_name: str
            Synaptic variable name.
        synaptic_variable_values: ndarray
            Synaptic variable values to saturate.

        Returns
        ----------
        result : ndarray
            Saturated synaptic variable values.
        """
        # Weights
        if synaptic_variable_name == "weights":
            if np.equal(self.sign_mode, SignMode.MIXED.value):
                return saturate(
                    -(2**W_WEIGHTS_U) - 1,
                    synaptic_variable_values,
                    2**W_WEIGHTS_U - 1,
                )
            elif np.equal(self.sign_mode, SignMode.EXCITATORY.value):
                return saturate(
                    0, synaptic_variable_values, 2**W_WEIGHTS_U - 1
                )
            elif np.equal(self.sign_mode, SignMode.INHIBITORY.value):
                return saturate(
                    -(2**W_WEIGHTS_U) - 1, synaptic_variable_values, 0
                )
        # Delays
        elif synaptic_variable_name == "tag_2":
            return saturate(0, synaptic_variable_values, 2**W_TAG_2_U - 1)
        # Tags
        elif synaptic_variable_name == "tag_1":
            return saturate(
                -(2**W_TAG_1_U) - 1,
                synaptic_variable_values,
                2**W_TAG_1_U - 1,
            )
        else:
            raise ValueError(
                "Invalid synaptic_variable_name in "
                "saturate_synaptic_variable"
            )

    def _apply_learning_rules(self) -> None:
        """Update all synaptic variables according to the
        LearningRuleApplier representation of their corresponding
        learning rule."""
        applier_args = self._extract_applier_args()

        for syn_var_name, lr_applier in self._learning_rule_appliers.items():
            syn_var = getattr(self, syn_var_name).copy()
            syn_var = np.left_shift(
                syn_var, W_ACCUMULATOR_S - W_SYN_VAR_S[syn_var_name]
            )
            syn_var = lr_applier.apply(syn_var, **applier_args)
            syn_var = self._saturate_synaptic_variable_accumulator(
                syn_var_name, syn_var
            )
            syn_var = self._stochastic_round_synaptic_variable(
                syn_var_name,
                syn_var,
                self._conn_var_random.random_stochastic_round,
            )
            syn_var = np.right_shift(
                syn_var, W_ACCUMULATOR_S - W_SYN_VAR_S[syn_var_name]
            )

            syn_var = self._saturate_synaptic_variable(syn_var_name, syn_var)
            setattr(self, syn_var_name, syn_var)

    @staticmethod
    def _add_impulse(
        trace_values: np.ndarray,
        random: int,
        impulses_int: np.ndarray,
        impulses_frac: np.ndarray,
    ) -> np.ndarray:
        """Add trace impulse impulse value and stochastically round
        the result.

        Parameters
        ----------
        trace_values : np.ndarray
            Trace values before impulse addition.
        random : int
            Randomly generated number.
        impulses_int: np.ndarray
            Trace impulses integer part.
        impulses_frac: np.ndarray
            Trace impulses fractional part.

        Returns
        ----------
        trace_new : np.ndarray
            Trace values before impulse addition and stochastic rounding.
        """
        trace_new = trace_values + impulses_int
        trace_new = stochastic_round(trace_new, random, impulses_frac)
        trace_new = saturate(0, trace_new, 2**W_TRACE - 1)

        return trace_new

    @staticmethod
    def _decay_trace(
        trace_values: np.ndarray, t: np.ndarray, taus: np.ndarray, random: float
    ) -> np.ndarray:
        """Stochastically decay trace to a given within-epoch time step.

        Parameters
        ----------
        trace_values : ndarray
            Trace values to decay.
        t : np.ndarray
            Time steps to advance.
        taus : int
            Trace decay time constant
        random: float
            Randomly generated number.

        Returns
        ----------
        result : ndarray
            Decayed trace values.
        """
        integer_part = np.exp(-t / taus) * trace_values
        fractional_part = integer_part % 1

        integer_part = np.floor(integer_part)
        result = stochastic_round(integer_part, random, fractional_part)

        return result

    def _evaluate_trace(
        self,
        trace_values: np.ndarray,
        t_spikes: np.ndarray,
        t_eval: np.ndarray,
        trace_impulses_int: np.ndarray,
        trace_impulses_frac: np.ndarray,
        trace_taus: np.ndarray,
        trace_random: TraceRandom,
    ) -> np.ndarray:
        """Evaluate a trace at given within-epoch time steps, given
        within-epoch spike timings.

        (1) If t_spikes > 0, stochastic decay to t_spikes,
        stochastic addition of trace impulse value, stochastic decay to t_eval.

        (2) If t_spikes == 0, stochastic decay to t_eval.

        Parameters
        ----------
        trace_values : ndarray
            Trace values at the beginning of the epoch.
        t_eval: ndarray
            Within-epoch evaluation time steps.
        t_spikes : ndarray
            Within-epoch spike timings.
        trace_impulses_int: ndarray
            Trace impulse values, integer part.
        trace_impulses_frac: ndarray
            Trace impulse values, fractional part.
        trace_taus: ndarray
            Trace decay time constants.
        trace_random: TraceRandom
            Trace random generator.

        Returns
        ----------
        result : ndarray
            Evaluated trace values.
        """
        broad_impulses_int = np.broadcast_to(
            trace_impulses_int, trace_values.shape
        )
        broad_impulses_frac = np.broadcast_to(
            trace_impulses_frac, trace_values.shape
        )
        broad_taus = np.broadcast_to(trace_taus, trace_values.shape)

        t_diff = t_eval - t_spikes

        decay_only = np.logical_and(
            np.logical_or(t_spikes == 0, t_diff < 0), broad_taus > 0
        )
        decay_spike_decay = np.logical_and(
            t_spikes != 0, t_diff >= 0, broad_taus > 0
        )

        result = trace_values.copy()

        result[decay_only] = self._decay_trace(
            trace_values[decay_only],
            t_eval[decay_only],
            broad_taus[decay_only],
            trace_random.random_trace_decay,
        )

        result[decay_spike_decay] = self._decay_trace(
            result[decay_spike_decay],
            t_spikes[decay_spike_decay],
            broad_taus[decay_spike_decay],
            trace_random.random_trace_decay,
        )

        result[decay_spike_decay] = self._add_impulse(
            result[decay_spike_decay],
            trace_random.random_impulse_addition,
            broad_impulses_int[decay_spike_decay],
            broad_impulses_frac[decay_spike_decay],
        )

        result[decay_spike_decay] = self._decay_trace(
            result[decay_spike_decay],
            t_diff[decay_spike_decay],
            broad_taus[decay_spike_decay],
            trace_random.random_trace_decay,
        )

        return result

    def _update_traces(self) -> None:
        """Update all traces at the end of the learning epoch."""
        # Shape: (2, num_pre_neurons)
        active_x_traces_broad = np.broadcast_to(
            self._active_x_traces[:, np.newaxis], self._shape_x_traces
        )
        # Shape: (3, num_post_neurons)
        active_y_traces_broad = np.broadcast_to(
            self._active_y_traces[:, np.newaxis], self._shape_y_traces
        )

        # Shape: (2, num_pre_neurons)
        t_spikes_x = np.where(active_x_traces_broad, self.tx, 0)
        # Shape: (3, num_post_neurons)
        t_spikes_y = np.where(active_y_traces_broad, self.ty, 0)

        # Shape: (2, num_pre_neurons)
        t_eval_x = np.where(
            active_x_traces_broad, self._learning_rule.t_epoch, 0
        )
        # Shape: (3, num_post_neurons)
        t_eval_y = np.where(
            active_y_traces_broad, self._learning_rule.t_epoch, 0
        )

        # Shape: (2, num_pre_neurons)
        # Shape of _x_impulses and _x_taus: (2, ) -> (2, 1)
        self._set_x_traces(
            self._evaluate_trace(
                self._x_traces,
                t_spikes_x,
                t_eval_x,
                self._x_impulses_int[:, np.newaxis],
                self._x_impulses_frac[:, np.newaxis],
                self._x_taus[:, np.newaxis],
                self._x_random,
            )
        )
        # Shape: (3, num_post_neurons)
        # Shape of _x_impulses and _x_taus: (3, ) -> (3, 1)
        self._set_y_traces(
            self._evaluate_trace(
                self._y_traces,
                t_spikes_y,
                t_eval_y,
                self._y_impulses_int[:, np.newaxis],
                self._y_impulses_frac[:, np.newaxis],
                self._y_taus[:, np.newaxis],
                self._y_random,
            )
        )


class ConnectionModelFloat(Connection):
    """Floating-point implementation of the Connection Process

    This ProcessModel constitutes a behavioral implementation of Loihi synapses
    written in Python, executing on CPU, and operating in floating-point
    arithmetic.

    To summarize the behavior:

    Spiking phase:
    run_spk:

        (1) (Dense) Send activations from past time step to post-synaptic
        neuron Process.
        (2) (Dense) Compute activations to be sent on next time step.
        (3) (Dense) Receive spikes from pre-synaptic neuron Process.
        (4) (Dense) Record within-epoch pre-synaptic spiking time.
        Update pre-synaptic traces if more than one spike during the epoch.
        (5) Receive spikes from post-synaptic neuron Process.
        (6) Record within-epoch pre-synaptic spiking time.
        Update pre-synaptic traces if more than one spike during the epoch.

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

    Parameters
    ----------
    proc_params: dict
        Parameters from the ProcessModel
    """

    # Learning Ports
    s_in_bap: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool)

    # Learning Vars
    x0: np.ndarray = LavaPyType(np.ndarray, bool)
    tx: np.ndarray = LavaPyType(np.ndarray, int, precision=6)
    x1: np.ndarray = LavaPyType(np.ndarray, float)
    x2: np.ndarray = LavaPyType(np.ndarray, float)

    y0: np.ndarray = LavaPyType(np.ndarray, bool)
    ty: np.ndarray = LavaPyType(np.ndarray, int, precision=6)
    y1: np.ndarray = LavaPyType(np.ndarray, float)
    y2: np.ndarray = LavaPyType(np.ndarray, float)
    y3: np.ndarray = LavaPyType(np.ndarray, float)

    tag_2: np.ndarray = LavaPyType(np.ndarray, float)
    tag_1: np.ndarray = LavaPyType(np.ndarray, float)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)

        self._shape = self.proc_params["shape"]
        self._learning_rule = self.proc_params["learning_rule"]

        if self._learning_rule is not None:
            # store shapes that useful throught the lifetime of this PM
            self._store_shapes()
            # store impulses and taus in ndarrays with the right shapes
            self._store_impulses_and_taus()

            # store active traces per dependency from learning_rule in ndarrays
            # with the right shapes
            self._build_active_traces_per_dependency()
            # store active traces from learning_rule in ndarrays
            # with the right shapes
            self._build_active_traces()
            # generate LearningRuleApplierFloat from ProductSeries
            self._build_learning_rule_appliers()

    def _store_impulses_and_taus(self) -> None:
        """Build and store integer ndarrays representing x and y
        impulses and taus."""
        self._x_impulses = np.array(
            [self._learning_rule.x1_impulse, self._learning_rule.x2_impulse]
        )
        self._x_taus = np.array(
            [self._learning_rule.x1_tau, self._learning_rule.x2_tau]
        )

        self._y_impulses = np.array(
            [
                self._learning_rule.y1_impulse,
                self._learning_rule.y2_impulse,
                self._learning_rule.y3_impulse,
            ]
        )
        self._y_taus = np.array(
            [
                self._learning_rule.y1_tau,
                self._learning_rule.y2_tau,
                self._learning_rule.y3_tau,
            ]
        )

    def _build_learning_rule_appliers(self) -> None:
        """Build and store LearningRuleApplierFloat for each active learning
        rule in a dict mapped by the learning rule's target."""
        self._learning_rule_appliers = {
            str_symbols.SYNAPTIC_VARIABLE_VAR_MAPPING[
                target[1:]
            ]: LearningRuleApplierFloat(ps)
            for target, ps in self._learning_rule.active_product_series.items()
        }

    def run_spk(self) -> None:
        s_in_bap = self.s_in_bap.recv().astype(bool)
        if self._learning_rule is not None:
            self._record_post_spike_times(s_in_bap)

    def _record_pre_spike_times(self, s_in: np.ndarray) -> None:
        """Record within-epoch spiking times of pre-synaptic neurons.

        If more a single pre-synaptic neuron spikes more than once,
        the corresponding trace is updated by its trace impulse value.

        Parameters
        ----------
        s_in : ndarray
            Pre-synaptic spikes.
        """

        self.x0[s_in] = True
        multi_spike_x = np.logical_and(self.tx > 0, s_in)

        x_traces = self._x_traces
        x_traces[:, multi_spike_x] += self._x_impulses[:, np.newaxis]
        self._set_x_traces(x_traces)

        ts_offset = self._within_epoch_time_step()
        self.tx[s_in] = ts_offset

    def _record_post_spike_times(self, s_in_bap: np.ndarray) -> None:
        """Record within-epoch spiking times of post-synaptic neurons.

        If more a single post-synaptic neuron spikes more than once,
        the corresponding trace is updated by its trace impulse value.

        Parameters
        ----------
        s_in_bap : ndarray
            Post-synaptic spikes.
        """

        self.y0[s_in_bap] = True
        multi_spike_y = np.logical_and(self.ty > 0, s_in_bap)

        y_traces = self._y_traces
        y_traces[:, multi_spike_y] += self._y_impulses[:, np.newaxis]
        self._set_y_traces(y_traces)

        ts_offset = self._within_epoch_time_step()
        self.ty[s_in_bap] = ts_offset

    def lrn_guard(self) -> bool:
        if self._learning_rule is not None:
            return self.time_step % self._learning_rule.t_epoch == 0
        return 0

    def run_lrn(self) -> None:
        self._apply_learning_rules()

        self._update_traces()

        self._reset_dependencies_and_spike_times()

    def _apply_learning_rules(self) -> None:
        """Update all synaptic variables according to the
        LearningRuleApplier representation of their corresponding
        learning rule."""
        applier_args = self._extract_applier_args()

        for syn_var_name, lr_applier in self._learning_rule_appliers.items():
            syn_var = getattr(self, syn_var_name).copy()
            syn_var = lr_applier.apply(syn_var, **applier_args)
            syn_var = self._saturate_synaptic_variable(syn_var_name, syn_var)
            setattr(self, syn_var_name, syn_var)

    def _extract_applier_args(self) -> dict:
        """Extracts arguments for the LearningRuleApplierFloat.

        "u" is a scalar.
        "np" is a reference to numpy as it is needed for the evaluation of
        "np.sign()" types of call inside the applier string.

        Shapes of numpy array args:
        "x0": (1, num_neurons_pre)
        "y0": (num_neurons_post, 1)
        "weights":  (num_neurons_post, num_neurons_pre)
        "tag_2": (num_neurons_post, num_neurons_pre)
        "tag_1": (num_neurons_post, num_neurons_pre)
        "traces": (3, 5, num_neurons_post, num_neurons_pre)

        "traces" is of shape (3, 5, num_neurons_post, num_neurons_pre) with:
        First dimension representing the within-epoch time step at which the
        trace is evaluated (tx, ty, t_epoch).
        Second dimension representing the trace that is evaluated
        (x1, x2, y1, y2, y3).
        """
        # Shape: (3, 2, num_post_neurons, num_pre_neurons)
        # Shape of active_x_traces_per_dependency : (3, 2) ->
        # (3, 2, 1, 1)
        active_x_traces_per_dep_broad = np.broadcast_to(
            self._active_x_traces_per_dependency[:, :, np.newaxis, np.newaxis],
            self._shape_x_traces_per_dep_broad,
        )

        # Shape: (3, 3, num_post_neurons, num_pre_neurons)
        # Shape of active_y_traces_per_dependency : (3, 3) ->
        # (3, 3, 1, 1)
        active_y_traces_per_dep_broad = np.broadcast_to(
            self._active_y_traces_per_dependency[:, :, np.newaxis, np.newaxis],
            self._shape_y_traces_per_dep_broad,
        )

        # Shape x0: (num_pre_neurons, ) -> (1, num_pre_neurons)
        # Shape y0: (num_post_neurons, ) -> (num_post_neurons, 1)
        # Shape weights: (num_post_neurons, num_pre_neurons)
        # Shape tag_2: (num_post_neurons, num_pre_neurons)
        # Shape tag_1: (num_post_neurons, num_pre_neurons)
        applier_args = {
            "x0": self.x0[np.newaxis, :],
            "y0": self.y0[:, np.newaxis],
            "weights": self.weights,
            "tag_2": self.tag_2,
            "tag_1": self.tag_1,
            "u": 0,
            # Adding numpy to applier args to be able to use it for sign method
            "np": np,
        }

        if self._learning_rule.decimate_exponent is not None:
            k = self._learning_rule.decimate_exponent
            u = (
                1
                if int(self.time_step / self._learning_rule.t_epoch) % 2 ^ k
                == 0
                else 0
            )

            # Shape: (0, )
            applier_args["u"] = u

        # Shape: (3, 2, num_post_neurons, num_pre_neurons)
        # Shape of tx : (num_pre_neurons, ) ->
        # (1, 1, 1, num_pre_neurons)
        t_spikes_x = np.where(
            active_x_traces_per_dep_broad,
            self.tx[np.newaxis, np.newaxis, np.newaxis, :],
            0,
        )
        # Shape: (3, 3, num_post_neurons, num_pre_neurons)
        # Shape of ty : (num_post_neurons, ) ->
        # (1, 1, num_post_neurons, 1)
        t_spikes_y = np.where(
            active_y_traces_per_dep_broad,
            self.ty[np.newaxis, np.newaxis, :, np.newaxis],
            0,
        )

        # Shape: (3, 5, num_post_neurons, num_pre_neurons)
        # Shape of t_eval[0, :, :, :] : (5, num_post_neurons, num_pre_neurons)
        # Shape of tx : (num_pre_neurons, ) ->
        # (1, 1, num_pre_neurons)
        # Shape of ty : (num_post_neurons, ) ->
        # (1, num_post_neurons, 1)
        t_eval = np.zeros(self._shape_traces_per_dep_broad, dtype=int)
        t_eval[0, :, :, :] = self.tx[np.newaxis, np.newaxis, :]
        t_eval[1, :, :, :] = self.ty[np.newaxis, :, np.newaxis]
        t_eval[2, :, :, :] = self._learning_rule.t_epoch

        # Shape: (3, 2, num_post_neurons, num_pre_neurons)
        # Shape of _x_traces: (2, num_pre_neurons) ->
        # (1, 2, 1, num_pre_neurons)
        x_traces = np.where(
            active_x_traces_per_dep_broad,
            self._x_traces[np.newaxis, :, np.newaxis, :],
            0.0,
        )
        # Shape: (3, 3, num_post_neurons, num_pre_neurons)
        # Shape of _y_traces: (3, num_post_neurons) ->
        # (1, 3, 1, num_post_neurons)
        y_traces = np.where(
            active_y_traces_per_dep_broad,
            self._y_traces[np.newaxis, :, :, np.newaxis],
            0.0,
        )

        # Shape: (3, 5, num_post_neurons, num_pre_neurons)
        # Shape of concat(x_traces, y_traces):
        # (3, 5, num_post_neurons, num_pre_neurons)
        # Shape of t_eval: (3, 5, num_post_neurons, num_pre_neurons)
        # Shape of concat(t_spikes_x, t_spikes_y):
        # (3, 5, num_post_neurons, num_pre_neurons)
        # Shape of concat(_x_impulses, _y_impulses) and _taus:
        # (5, ) -> (1, 5, 1, 1)
        evaluated_traces = self._evaluate_trace(
            np.concatenate((x_traces, y_traces), axis=1),
            np.concatenate((t_spikes_x, t_spikes_y), axis=1),
            t_eval,
            np.concatenate((self._x_impulses, self._y_impulses), axis=0)[
                np.newaxis, :, np.newaxis, np.newaxis
            ],
            np.concatenate((self._x_taus, self._y_taus), axis=0)[
                np.newaxis, :, np.newaxis, np.newaxis
            ],
        )

        # Shape: (3, 5, num_post_neurons, num_pre_neurons)
        applier_args["traces"] = evaluated_traces

        return applier_args

    def _saturate_synaptic_variable(
        self, synaptic_variable_name: str, synaptic_variable_values: np.ndarray
    ) -> np.ndarray:
        """Saturate synaptic variable.

        Parameters
        ----------
        synaptic_variable_name: str
            Synaptic variable name.
        synaptic_variable_values: ndarray
            Synaptic variable values to saturate.

        Returns
        ----------
        result : ndarray
            Saturated synaptic variable values.
        """
        # Weights
        if synaptic_variable_name == "weights":
            if np.equal(self.sign_mode, SignMode.MIXED.value):
                return synaptic_variable_values
            elif np.equal(self.sign_mode, SignMode.EXCITATORY.value):
                return np.maximum(0, synaptic_variable_values)
            elif np.equal(self.sign_mode, SignMode.INHIBITORY.value):
                return np.minimum(0, synaptic_variable_values)
        # Delays
        elif synaptic_variable_name == "tag_2":
            return np.maximum(0, synaptic_variable_values)
        # Tags
        elif synaptic_variable_name == "tag_1":
            return synaptic_variable_values
        else:
            raise ValueError(
                "Invalid synaptic_variable_name in "
                "saturate_synaptic_variable"
            )

    @staticmethod
    def _decay_trace(
        trace_values: np.ndarray, t: np.ndarray, taus: np.ndarray
    ) -> np.ndarray:
        """Decay trace to a given within-epoch time step.

        Parameters
        ----------
        trace_values : ndarray
            Trace values to decay.
        t : np.ndarray
            Time steps to advance.
        taus : int
            Trace decay time constant

        Returns
        ----------
        result : ndarray
            Decayed trace values.

        """
        return np.exp(-t / taus) * trace_values

    def _evaluate_trace(
        self,
        trace_values: np.ndarray,
        t_spikes: np.ndarray,
        t_eval: np.ndarray,
        trace_impulses: np.ndarray,
        trace_taus: np.ndarray,
    ) -> np.ndarray:
        """Evaluate a trace at given within-epoch time steps, given
        within-epoch spike timings.

        (1) If t_spikes > 0, decay to t_spikes,
        addition of trace impulse value, decay to t_eval.

        (2) If t_spikes == 0, decay to t_eval.

        Parameters
        ----------
        trace_values : ndarray
            Trace values at the beginning of the epoch.
        t_spikes : ndarray
            Within-epoch spike timings.
        t_eval: ndarray
            Within-epoch evaluation time steps.
        trace_impulses: ndarray
            Trace impulse values.
        trace_taus: ndarray
            Trace decay time constants.

        Returns
        ----------
        result : ndarray
            Evaluated trace values.
        """
        broad_impulses = np.broadcast_to(trace_impulses, trace_values.shape)
        broad_taus = np.broadcast_to(trace_taus, trace_values.shape)

        t_diff = t_eval - t_spikes

        decay_only = np.logical_and(
            np.logical_or(t_spikes == 0, t_diff < 0), broad_taus > 0
        )
        decay_spike_decay = np.logical_and(
            t_spikes != 0, t_diff >= 0, broad_taus > 0
        )

        result = trace_values.copy()

        result[decay_only] = self._decay_trace(
            trace_values[decay_only], t_eval[decay_only], broad_taus[decay_only]
        )

        result[decay_spike_decay] = self._decay_trace(
            result[decay_spike_decay],
            t_spikes[decay_spike_decay],
            broad_taus[decay_spike_decay],
        )

        result[decay_spike_decay] += broad_impulses[decay_spike_decay]

        result[decay_spike_decay] = self._decay_trace(
            result[decay_spike_decay],
            t_diff[decay_spike_decay],
            broad_taus[decay_spike_decay],
        )

        return result

    def _update_traces(self) -> None:
        """Update all traces at the end of the learning epoch."""
        # Shape: (2, num_pre_neurons)
        active_x_traces_broad = np.broadcast_to(
            self._active_x_traces[:, np.newaxis], self._shape_x_traces
        )
        # Shape: (3, num_post_neurons)
        active_y_traces_broad = np.broadcast_to(
            self._active_y_traces[:, np.newaxis], self._shape_y_traces
        )

        # Shape: (2, num_pre_neurons)
        t_spikes_x = np.where(active_x_traces_broad, self.tx, 0)
        # Shape: (3, num_post_neurons)
        t_spikes_y = np.where(active_y_traces_broad, self.ty, 0)

        # Shape: (2, num_pre_neurons)
        t_eval_x = np.where(
            active_x_traces_broad, self._learning_rule.t_epoch, 0
        )
        # Shape: (3, num_post_neurons)
        t_eval_y = np.where(
            active_y_traces_broad, self._learning_rule.t_epoch, 0
        )

        # Shape: (2, num_pre_neurons)
        # Shape of _x_impulses and _x_taus: (2, ) -> (2, 1)
        self._set_x_traces(
            self._evaluate_trace(
                self._x_traces,
                t_spikes_x,
                t_eval_x,
                self._x_impulses[:, np.newaxis],
                self._x_taus[:, np.newaxis],
            )
        )
        # Shape: (3, num_post_neurons)
        # Shape of _x_impulses and _x_taus: (3, ) -> (3, 1)
        self._set_y_traces(
            self._evaluate_trace(
                self._y_traces,
                t_spikes_y,
                t_eval_y,
                self._y_impulses[:, np.newaxis],
                self._y_taus[:, np.newaxis],
            )
        )
