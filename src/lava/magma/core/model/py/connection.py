# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from abc import abstractmethod
from lava.utils.sparse import find
import numpy as np
import typing
from scipy.sparse import csr_matrix

from lava.magma.core.learning.learning_rule import (
    LoihiLearningRule,
    Loihi2FLearningRule,
    Loihi3FLearningRule,
)
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.model.py.type import LavaPyType

from lava.magma.core.learning.constants import (
    GradedSpikeCfg,
    W_TRACE,
    W_TRACE_FRACTIONAL_PART,
    W_SYN_VAR_U,
    W_SYN_VAR_S,
    W_WEIGHTS_U,
    W_TAG_1_U,
    W_TAG_2_U,
    W_ACCUMULATOR_U,
    W_ACCUMULATOR_S,
)
from lava.magma.core.learning.random import TraceRandom, ConnVarRandom
from lava.magma.core.learning.product_series import ProductSeries
from lava.magma.core.learning.learning_rule_applier import (
    AbstractLearningRuleApplier,
    LearningRuleApplierFloat,
    LearningRuleApplierBitApprox,
)
import lava.magma.core.learning.string_symbols as str_symbols
from lava.utils.weightutils import SignMode, clip_weights
from lava.magma.core.learning.utils import stochastic_round
import logging

NUM_DEPENDENCIES = len(str_symbols.DEPENDENCIES)
NUM_X_TRACES = len(str_symbols.PRE_TRACES)
NUM_Y_TRACES = len(str_symbols.POST_TRACES)


class AbstractLearningConnection:
    """Base class for learning connection ProcessModels."""

    # Learning Ports
    s_in_bap = None
    s_in_y1 = None
    s_in_y2 = None
    s_in_y3 = None

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

    dw = None
    dt = None
    dd = None

    x1_tau = None
    x1_impulse = None
    x2_tau = None
    x2_impulse = None

    y1_tau = None
    y1_impulse = None
    y2_tau = None
    y2_impulse = None
    y3_tau = None
    y3_impulse = None


class PyLearningConnection(AbstractLearningConnection):
    """Base class for learning connection ProcessModels in Python / CPU.

    This class provides commonly used functions for simulating the Loihi
    learning engine. It is subclasses for floating and fixed point
    simulations.

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

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)

        # If there is a plasticity rule attached to the corresponding process,
        # add all necessary ports get access to all learning params
        self._learning_rule: LoihiLearningRule = proc_params["learning_rule"]
        self._shape: typing.Tuple[int, ...] = proc_params["shape"]
        self._graded_spike_cfg: GradedSpikeCfg = proc_params["graded_spike_cfg"]

        self.sign_mode = proc_params.get("sign_mode", SignMode.MIXED)

        self._store_shapes()
        self._store_impulses_and_taus()

        self._build_active_traces_per_dependency()
        self._build_active_traces()
        self._build_learning_rule_appliers()
        self._init_randoms()

    def on_var_update(self):
        """ Update the learning rule parameters when on single Var is
        updated. """

        self._learning_rule.x1_tau = self.x1_tau[0]
        self._learning_rule.x1_impulse = self.x1_impulse[0]
        self._learning_rule.x2_tau = self.x2_tau[0]
        self._learning_rule.x2_impulse = self.x2_impulse[0]

        self._learning_rule.y1_tau = self.y1_tau[0]
        self._learning_rule.y1_impulse = self.y1_impulse[0]
        self._learning_rule.y2_tau = self.y2_tau[0]
        self._learning_rule.y2_impulse = self.y2_impulse[0]
        self._learning_rule.y3_tau = self.y3_tau[0]
        self._learning_rule.y3_impulse = self.y3_impulse[0]

        self._learning_rule.dw_str = self.dw
        self._learning_rule.dd_str = self.dd
        self._learning_rule.dt_str = self.dt

        self._store_impulses_and_taus()
        self._build_active_traces_per_dependency()
        self._build_active_traces()
        self._build_learning_rule_appliers()

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

    @abstractmethod
    def _store_impulses_and_taus(self):
        pass

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
                    raise ValueError("Unknown Trace in ProcessModel.")

                active_traces_per_dependency[dependency_idx, trace_idx] = True

        # Shape : (3, 2)
        self._active_x_traces_per_dependency = \
            active_traces_per_dependency[:, :2]

        # Shape : (3, 3)
        self._active_y_traces_per_dependency = \
            active_traces_per_dependency[:, 2:]

    def _build_active_traces(self) -> None:
        """Build and store boolean numpy arrays specifying which x and y
        traces are active."""
        # Shape : (2, )
        self._active_x_traces = \
            (self._active_x_traces_per_dependency[0]
             | self._active_x_traces_per_dependency[1]
             | self._active_x_traces_per_dependency[2])

        # Shape : (3, )
        self._active_y_traces = \
            (self._active_y_traces_per_dependency[0]
             | self._active_y_traces_per_dependency[1]
             | self._active_y_traces_per_dependency[2])

    def _build_learning_rule_appliers(self) -> None:
        """Build and store LearningRuleApplier for each active learning
        rule in a dict mapped by the learning rule's target."""
        self._learning_rule_appliers = {
            str_symbols.SYNAPTIC_VARIABLE_VAR_MAPPING[
                target[1:]
            ]: self._create_learning_rule_applier(ps)
            for target, ps in self._learning_rule.active_product_series.items()
        }

    @abstractmethod
    def _create_learning_rule_applier(
            self, product_series: ProductSeries
    ) -> AbstractLearningRuleApplier:
        pass

    @abstractmethod
    def _init_randoms(self) -> None:
        pass

    @property
    def _x_traces(self) -> np.ndarray:
        """Get x traces.

        Returns
        ----------
        x_traces : np.ndarray
            X traces (shape: (2, num_pre_neurons)).
        """
        return np.concatenate(
            (self.x1[np.newaxis, :], self.x2[np.newaxis, :]), axis=0
        ).copy()

    def _set_x_traces(self, x_traces: np.ndarray) -> None:
        """Set x traces.

        Parameters
        ----------
        x_traces : np.ndarray
            X traces.
        """
        self.x1 = x_traces[0]
        self.x2 = x_traces[1]

    @property
    def _y_traces(self) -> np.ndarray:
        """Get y traces.

        Returns
        ----------
        y_traces : np.ndarray
            Y traces (shape: (3, num_post_neurons)).
        """
        return np.concatenate(
            (
                self.y1[np.newaxis, :],
                self.y2[np.newaxis, :],
                self.y3[np.newaxis, :],
            ),
            axis=0,
        ).copy()

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

    def recv_traces(self, s_in) -> None:
        """
        Function to receive and update y1, y2 and y3 traces
        from the post-synaptic neuron.

        Parameters
        ----------
        s_in : np.adarray
            Synaptic spike input
        """
        self._process_pre_spikes(s_in)
        self._update_trace_randoms()

        if isinstance(self._learning_rule, Loihi2FLearningRule):
            s_in_bap = self.s_in_bap.recv().astype(bool)
            self._process_post_spikes(s_in_bap)
        elif isinstance(self._learning_rule, Loihi3FLearningRule):
            s_in_bap = self.s_in_bap.recv().astype(bool)
            y1 = self.s_in_y1.recv()
            y2 = self.s_in_y2.recv()
            y3 = self.s_in_y3.recv()

            self._process_post_spikes(s_in_bap)

            y_traces = self._y_traces
            y_traces[0, :] = y1
            y_traces[1, :] = y2
            y_traces[2, :] = y3
            self._set_y_traces(y_traces)

        self._update_trace_randoms()

    @abstractmethod
    def _process_pre_spikes(self, s_in: np.ndarray) -> None:
        pass

    @abstractmethod
    def _process_post_spikes(self, s_in_bap: np.ndarray) -> None:
        pass

    @abstractmethod
    def _update_trace_randoms(self) -> None:
        pass

    def lrn_guard(self) -> bool:
        return self.time_step % self._learning_rule.t_epoch == 0

    def run_lrn(self) -> None:
        self._update_synaptic_variable_random()
        self._update_dependencies()
        x_traces_history, y_traces_history = self._compute_trace_histories()
        self._update_traces(x_traces_history, y_traces_history)
        self._apply_learning_rules(x_traces_history, y_traces_history)
        self._reset_dependencies_and_spike_times()

    @abstractmethod
    def _update_synaptic_variable_random(self) -> None:
        pass

    def _update_dependencies(self) -> None:
        self.x0 = self.tx > 0
        self.y0 = self.ty > 0

    @abstractmethod
    def _compute_trace_histories(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        pass

    def _update_traces(
        self, x_traces_history: np.ndarray, y_traces_history: np.ndarray
    ) -> None:
        """Update x and y traces to last values in the epoch history.

        Parameters
        ----------
        x_traces_history : ndarray
            History of x trace values within the epoch.
        y_traces_history : np.ndarray
            History of y trace values within the epoch.
        """
        # set traces to last value
        self._set_x_traces(x_traces_history[-1])
        if isinstance(self._learning_rule, Loihi2FLearningRule):
            self._set_y_traces(y_traces_history[-1])

    @abstractmethod
    def _apply_learning_rules(
        self, x_traces_history: np.ndarray, y_traces_history: np.ndarray
    ) -> None:
        pass

    def _extract_applier_evaluated_traces(
        self, x_traces_history: np.ndarray, y_traces_history: np.ndarray
    ) -> typing.Dict[str, np.ndarray]:
        """Extract x and y trace values on time steps derived from each of
        allowed dependencies.

        Parameters
        ----------
        x_traces_history : ndarray
            History of x trace values within the epoch.
        y_traces_history : np.ndarray
            History of y trace values within the epoch.

        Returns
        ----------
        evaluated_traces : dict
            x and y traces evaluated on time steps derived from dependencies
        """
        evaluated_traces = {
            # Shape : (1, num_pre_neurons)
            "x1_x0": x_traces_history[self.tx, 0].diagonal()[np.newaxis, :],
            # Shape : (1, num_pre_neurons)
            "x2_x0": x_traces_history[self.tx, 1].diagonal()[np.newaxis, :],
            # Shape : (num_post_neurons, num_pre_neurons)
            "y1_x0": y_traces_history[self.tx, 0].T,
            # Shape : (num_post_neurons, num_pre_neurons)
            "y2_x0": y_traces_history[self.tx, 1].T,
            # Shape : (num_post_neurons, num_pre_neurons)
            "y3_x0": y_traces_history[self.tx, 2].T,
            # Shape : (num_post_neurons, num_pre_neurons)
            "x1_y0": x_traces_history[self.ty, 0],
            # Shape : (num_post_neurons, num_pre_neurons)
            "x2_y0": x_traces_history[self.ty, 1],
            # Shape : (num_post_neurons, 1)
            "y1_y0": y_traces_history[self.ty, 0].diagonal()[:, np.newaxis],
            # Shape : (num_post_neurons, 1)
            "y2_y0": y_traces_history[self.ty, 1].diagonal()[:, np.newaxis],
            # Shape : (num_post_neurons, 1)
            "y3_y0": y_traces_history[self.ty, 2].diagonal()[:, np.newaxis],
            # Shape : (1, num_pre_neurons)
            "x1_u": x_traces_history[-1, 0][np.newaxis, :],
            # Shape : (1, num_pre_neurons)
            "x2_u": x_traces_history[-1, 1][np.newaxis, :],
            # Shape : (num_post_neurons, 1)
            "y1_u": y_traces_history[-1, 0][:, np.newaxis],
            # Shape : (num_post_neurons, 1)
            "y2_u": y_traces_history[-1, 1][:, np.newaxis],
            # Shape : (num_post_neurons, 1)
            "y3_u": y_traces_history[-1, 2][:, np.newaxis],
        }

        return evaluated_traces

    def _reset_dependencies_and_spike_times(self) -> None:
        """Reset all dependencies and within-epoch spike times."""
        self.x0 = np.zeros_like(self.x0)
        self.y0 = np.zeros_like(self.y0)

        self.tx = np.zeros_like(self.tx)
        self.ty = np.zeros_like(self.ty)


class LearningConnectionModelBitApproximate(PyLearningConnection):
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
    s_in_y1: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=7)
    s_in_y2: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=7)
    s_in_y3: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=7)

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

    dw: str = LavaPyType(str, str)
    dd: str = LavaPyType(str, str)
    dt: str = LavaPyType(str, str)

    x1_tau: np.ndarray = LavaPyType(np.ndarray, int, precision=8)
    x1_impulse: np.ndarray = LavaPyType(np.ndarray, int, precision=8)
    x2_tau: np.ndarray = LavaPyType(np.ndarray, int, precision=8)
    x2_impulse: np.ndarray = LavaPyType(np.ndarray, int, precision=8)

    y1_tau: np.ndarray = LavaPyType(np.ndarray, int, precision=8)
    y1_impulse: np.ndarray = LavaPyType(np.ndarray, int, precision=8)
    y2_tau: np.ndarray = LavaPyType(np.ndarray, int, precision=8)
    y2_impulse: np.ndarray = LavaPyType(np.ndarray, int, precision=8)
    y3_tau: np.ndarray = LavaPyType(np.ndarray, int, precision=8)
    y3_impulse: np.ndarray = LavaPyType(np.ndarray, int, precision=8)

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

    def _create_learning_rule_applier(
        self, product_series: ProductSeries
    ) -> AbstractLearningRuleApplier:
        """Create a LearningRuleApplierBitApprox."""
        return LearningRuleApplierBitApprox(product_series)

    def _init_randoms(self) -> None:
        """Initialize trace and synaptic variable random generators."""
        self._x_random = TraceRandom(
            seed_trace_decay=self._learning_rule.rng_seed,
            seed_impulse_addition=self._learning_rule.rng_seed + 1,
        )

        self._y_random = TraceRandom(
            seed_trace_decay=self._learning_rule.rng_seed + 2,
            seed_impulse_addition=self._learning_rule.rng_seed + 3,
        )

        self._conn_var_random = ConnVarRandom()

    def _process_pre_spikes(self, s_in: np.ndarray) -> None:
        """Process pre-synaptic spikes.

        Four different modes of operation, based on GradedSpikeCfg value:
        (0) GradedSpikeCfg.USE_REGULAR_IMPULSE if a single pre-synaptic neuron
        spikes more than once, pre-traces are updated by their regular impulses.
        (1) GradedSpikeCfg.OVERWRITE overwrites the value of the pre-synaptic
        trace x1 by payload/2, upon spiking.
        (2) GradedSpikeCfg.ADD_WITH_SATURATION adds payload/2 to the
        pre-synaptic trace x1, upon spiking, saturates x1 to 127.
        (3) GradedSpikeCfg.ADD_WITHOUT_SATURATION adds payload/2 to the
        pre-synaptic trace x1, upon spiking, keeps only overflow from 127 in x1,
        adds regular impulse to x2 on overflow.

        Within-epoch spike times are recorded.
        With GradedSpikeCfg.ADD_WITHOUT_SATURATION, only spike times of spikes
        triggering x1 overflow and x2 impulse addition are recorded.

        Parameters
        ----------
        s_in : ndarray
            Pre-synaptic spikes.
        """
        spiked = s_in.astype(bool)

        multi_spike_x = (self.tx > 0) & spiked

        activations = s_in
        scaled_activations = np.round(activations / 2).astype(np.uint8)

        update_t_spike = spiked
        x2_update_idx = multi_spike_x

        if self._graded_spike_cfg == GradedSpikeCfg.USE_REGULAR_IMPULSE:
            self.x1[multi_spike_x] = self._add_impulse(
                self.x1[multi_spike_x],
                0,
                self._x_impulses_int[0],
                self._x_impulses_frac[0],
            )

        elif self._graded_spike_cfg == GradedSpikeCfg.OVERWRITE:
            self.x1[spiked] = scaled_activations[spiked]

        elif self._graded_spike_cfg == GradedSpikeCfg.ADD_WITH_SATURATION or \
                self._graded_spike_cfg == GradedSpikeCfg.ADD_WITHOUT_SATURATION:
            sums = (self.x1 + scaled_activations).astype(np.uint8)

            if self._graded_spike_cfg == GradedSpikeCfg.ADD_WITH_SATURATION:
                self.x1 = np.clip(sums, 0, 127)

            if self._graded_spike_cfg == GradedSpikeCfg.ADD_WITHOUT_SATURATION:
                overflow_idx = sums > 127
                update_t_spike = update_t_spike & overflow_idx
                x2_update_idx = x2_update_idx & overflow_idx

                self.x1 = sums
                self.x1[overflow_idx] -= 127

        self.x2[x2_update_idx] = self._add_impulse(
            self.x2[x2_update_idx],
            0,
            self._x_impulses_int[1],
            self._x_impulses_frac[1],
        )

        ts_offset = self._within_epoch_time_step()
        self.tx[update_t_spike] = ts_offset

    def _process_post_spikes(self, s_in_bap: np.ndarray) -> None:
        """Process post-synaptic spikes.

        Records within-epoch spiking times of post-synaptic neurons.
        If a single post-synaptic neuron spikes more than once,
        the corresponding trace is updated by its trace impulse value.

        Parameters
        ----------
        s_in_bap : ndarray
            Post-synaptic spikes.
        """
        self.y0[s_in_bap] = True
        multi_spike_y = (self.ty > 0) & s_in_bap

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

    def _update_trace_randoms(self) -> None:
        """Update trace random generators."""
        self._x_random.advance()
        self._y_random.advance()

    def _update_synaptic_variable_random(self) -> None:
        """Update synaptic variable random generators."""
        self._conn_var_random.advance()

    def _compute_trace_histories(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Compute history of x and y trace values within the past epoch.

        Returns
        ----------
        x_traces_history : ndarray
            History of x trace values within the epoch.
        y_traces_history : np.ndarray
            History of y trace values within the epoch.
        """
        # Gather all necessary information to decay traces
        x_traces = self._x_traces
        y_traces = self._y_traces

        t_epoch = self._learning_rule.t_epoch

        x_random = self._x_random
        y_random = self._y_random

        x_impulses_int = self._x_impulses_int[:, np.newaxis]
        y_impulses_int = self._y_impulses_int[:, np.newaxis]

        x_impulses_frac = self._x_impulses_frac[:, np.newaxis]
        y_impulses_frac = self._y_impulses_frac[:, np.newaxis]

        x_taus = self._x_taus
        y_taus = self._y_taus

        # get spike times
        t_spike_x = self.tx
        t_spike_y = self.ty

        # most naive algorithm to decay traces
        x_traces_history = np.full((t_epoch + 1,) + x_traces.shape, 0,
                                   dtype=int)
        x_traces_history[0] = x_traces
        y_traces_history = np.full((t_epoch + 1,) + y_traces.shape, 0,
                                   dtype=int)

        y_traces_history[0] = y_traces

        for t in range(1, t_epoch + 1):
            x_traces_history[t][x_taus != 0] = self._decay_trace(
                x_traces_history[t - 1][x_taus != 0],
                1,
                x_taus[x_taus != 0][:, np.newaxis],
                x_random.random_trace_decay,
            )
            y_traces_history[t][y_taus != 0] = self._decay_trace(
                y_traces_history[t - 1][y_taus != 0],
                1,
                y_taus[y_taus != 0][:, np.newaxis],
                y_random.random_trace_decay,
            )

            # add impulses if spike happens in this timestep
            x_spike_ids = np.where(t_spike_x == t)[0]
            x_traces_history[t][:, x_spike_ids] = self._add_impulse(
                x_traces_history[t][:, x_spike_ids],
                x_random.random_impulse_addition,
                x_impulses_int,
                x_impulses_frac,
            )

            y_spike_ids = np.where(t_spike_y == t)[0]
            y_traces_history[t][:, y_spike_ids] = self._add_impulse(
                y_traces_history[t][:, y_spike_ids],
                y_random.random_impulse_addition,
                y_impulses_int,
                y_impulses_frac,
            )

        return x_traces_history, y_traces_history

    @staticmethod
    def _decay_trace(
            trace_values: np.ndarray, t: np.ndarray, taus: np.ndarray,
            random: float
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
        trace_new = np.clip(trace_new, a_min=0, a_max=2**W_TRACE - 1)

        return trace_new

    def _apply_learning_rules(
        self, x_traces_history: np.ndarray, y_traces_history: np.ndarray
    ) -> None:
        """Update all synaptic variables according to the
        LearningRuleApplier representation of their corresponding
        learning rule."""
        applier_args = self._extract_applier_args(
            x_traces_history, y_traces_history
        )

        for syn_var_name, lr_applier in self._learning_rule_appliers.items():
            syn_var = getattr(self, syn_var_name).copy()
            shift = W_ACCUMULATOR_S - W_SYN_VAR_S[syn_var_name]
            if isinstance(syn_var, csr_matrix):
                syn_var.data = syn_var.data << shift
                dst, src, _ = find(syn_var, explicit_zeros=True)
                syn_var[dst, src] = lr_applier.apply(syn_var,
                                                     **applier_args)[dst, src]
            else:
                syn_var = syn_var << shift
                syn_var = lr_applier.apply(syn_var, **applier_args)

            syn_var = self._saturate_synaptic_variable_accumulator(
                syn_var_name, syn_var
            )
            syn_var = self._stochastic_round_synaptic_variable(
                syn_var_name,
                syn_var,
                self._conn_var_random.random_stochastic_round,
            )

            if isinstance(syn_var, csr_matrix):
                syn_var.data = syn_var.data >> shift
            else:
                syn_var = syn_var >> shift

            syn_var = self._saturate_synaptic_variable(syn_var_name, syn_var)
            setattr(self, syn_var_name, syn_var)

    def _extract_applier_args(
        self, x_traces_history: np.ndarray, y_traces_history: np.ndarray
    ) -> dict:
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
        "evaluated_traces": see _extract_applier_evaluated_traces method
        for details.
        """

        # Shape x0: (num_pre_neurons, ) -> (1, num_pre_neurons)
        # Shape y0: (num_post_neurons, ) -> (num_post_neurons, 1)
        # Shape weights: (num_post_neurons, num_pre_neurons)
        # Shape tag_2: (num_post_neurons, num_pre_neurons)
        # Shape tag_1: (num_post_neurons, num_pre_neurons)
        applier_args = {
            "shape": self._shape,
            "x0": self.x0[np.newaxis, :],
            "y0": self.y0[:, np.newaxis],
            "weights": self.weights,
            "tag_2": self.tag_2,
            "tag_1": self.tag_1,
            "u": 0,
        }

        if self._learning_rule.decimate_exponent is not None:
            k = self._learning_rule.decimate_exponent
            u = (
                1
                if int(self.time_step / self._learning_rule.t_epoch) % 2 ** k
                == 0
                else 0
            )

            # Shape: (0, )
            applier_args["u"] = u

        evaluated_traces = self._extract_applier_evaluated_traces(
            x_traces_history, y_traces_history
        )
        applier_args.update(evaluated_traces)

        return applier_args

    def _saturate_synaptic_variable_accumulator(
            self, syn_var_name: str,
            syn_var_values: typing.Union[np.ndarray, csr_matrix]
    ) -> typing.Union[np.ndarray, csr_matrix]:
        """Saturate synaptic variable accumulator.

        Checks that sign is valid.

        Parameters
        ----------
        syn_var_name: str
            Synaptic variable name.
        syn_var_values: ndarray, csr_matrix
            Synaptic variable values to saturate.

        Returns
        ----------
        result : ndarray, csr_matrix
            Saturated synaptic variable values.
        """
        # Weights
        if syn_var_name == "weights":
            if self.sign_mode == SignMode.MIXED:
                return syn_var_values
            elif self.sign_mode == SignMode.EXCITATORY:
                return np.maximum(0, syn_var_values)
            elif self.sign_mode == SignMode.INHIBITORY:
                return np.minimum(0, syn_var_values)
        # Delays
        elif syn_var_name == "tag_2":
            if isinstance(syn_var_values, csr_matrix):
                syn_var_values.data[syn_var_values.data < 0] = 0
                return syn_var_values
            return np.maximum(0, syn_var_values)
        # Tags
        elif syn_var_name == "tag_1":
            return syn_var_values
        else:
            raise ValueError(
                f"syn_var_name can be 'weights', "
                f"'tag_1', or 'tag_2'."
                f"Got {syn_var_name=}."
            )

    @staticmethod
    def _stochastic_round_synaptic_variable(
        syn_var_name: str,
        syn_var_values: typing.Union[np.ndarray, csr_matrix],
        random: float,
    ) -> typing.Union[np.ndarray, csr_matrix]:
        """Stochastically round synaptic variable after learning rule
        application.

        Parameters
        ----------
        syn_var_name: str
            Synaptic variable name.
        syn_var_values: ndarray, csr_matrix
            Synaptic variable values to stochastically round.

        Returns
        ----------
        result : ndarray, csr_matrix
            Stochastically rounded synaptic variable values.
        """
        exp_mant = 2 ** (W_ACCUMULATOR_U - W_SYN_VAR_U[syn_var_name])

        if isinstance(syn_var_values, csr_matrix):
            integer_part = syn_var_values.data / exp_mant
        else:
            integer_part = syn_var_values / exp_mant
        fractional_part = integer_part % 1

        integer_part = np.floor(integer_part)
        integer_part = stochastic_round(integer_part, random, fractional_part)

        if isinstance(syn_var_values, csr_matrix):
            syn_var_values.data = (integer_part
                                   * exp_mant).astype(syn_var_values.dtype)
            return syn_var_values
        else:
            return (integer_part * exp_mant).astype(
                syn_var_values.dtype
            )

    def _saturate_synaptic_variable(
            self, syn_var_name: str,
            syn_var_val: typing.Union[np.ndarray, csr_matrix]
    ) -> np.ndarray:
        """Saturate synaptic variable.

        Checks that synaptic variable values is between bounds set by
        the hardware.

        Parameters
        ----------
        syn_var_name: str
            Synaptic variable name.
        syn_var_val: ndarray, csr_matrix
            Synaptic variable values to saturate.

        Returns
        ----------
        result : ndarray, csr_matrix
            Saturated synaptic variable values.
        """

        # Weights
        if syn_var_name == "weights":
            return clip_weights(
                syn_var_val,
                sign_mode=self.sign_mode,
                num_bits=W_WEIGHTS_U,
            )
        # Delays
        elif syn_var_name == "tag_2":
            if isinstance(syn_var_val, csr_matrix):
                _min = -(2 ** W_TAG_2_U) - 1
                _max = (2 ** W_TAG_2_U) - 1
                syn_var_val.data[syn_var_val.data < _min] = _min
                syn_var_val.data[syn_var_val.data > _max] = _max
                return syn_var_val

            return np.clip(
                syn_var_val, a_min=0, a_max=2 ** W_TAG_2_U - 1
            )
        # Tags
        elif syn_var_name == "tag_1":
            if isinstance(syn_var_val, csr_matrix):
                _min = -(2 ** W_TAG_1_U) - 1
                _max = (2 ** W_TAG_1_U) - 1
                syn_var_val.data[syn_var_val.data < _min] = _min
                syn_var_val.data[syn_var_val.data > _max] = _max
                return syn_var_val
            return np.clip(
                syn_var_val,
                a_min=-(2 ** W_TAG_1_U) - 1,
                a_max=2 ** W_TAG_1_U - 1,
            )
        else:
            raise ValueError(
                f"syn_var_name can be 'weights', "
                f"'tag_1', or 'tag_2'."
                f"Got {syn_var_name=}."
            )


class LearningConnectionModelFloat(PyLearningConnection):
    """Floating-point implementation of the Connection Process.

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
    s_in_y1: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_in_y2: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_in_y3: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

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

    dw: str = LavaPyType(str, str)
    dd: str = LavaPyType(str, str)
    dt: str = LavaPyType(str, str)

    x1_tau: np.ndarray = LavaPyType(np.ndarray, float)
    x1_impulse: np.ndarray = LavaPyType(np.ndarray, float)
    x2_tau: np.ndarray = LavaPyType(np.ndarray, float)
    x2_impulse: np.ndarray = LavaPyType(np.ndarray, float)

    y1_tau: np.ndarray = LavaPyType(np.ndarray, float)
    y1_impulse: np.ndarray = LavaPyType(np.ndarray, float)
    y2_tau: np.ndarray = LavaPyType(np.ndarray, float)
    y2_impulse: np.ndarray = LavaPyType(np.ndarray, float)
    y3_tau: np.ndarray = LavaPyType(np.ndarray, float)
    y3_impulse: np.ndarray = LavaPyType(np.ndarray, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)

        if self._graded_spike_cfg == GradedSpikeCfg.ADD_WITH_SATURATION or \
                self._graded_spike_cfg == GradedSpikeCfg.ADD_WITHOUT_SATURATION:
            logging.warning(
                'The floating-pt PyProcessModel has been selected for the '
                'LearningDense Process and '
                f'graded_spike_cfg={self._graded_spike_cfg}.')
            logging.warning(
                'Incoming graded spike payloads (divided by 2) will be added '
                'to the pre-trace x1, with no saturation. This will not '
                'effect the pre-trace x2.')
            logging.warning(
                'All incoming spikes will be considered by learning rule '
                'Products conditioned on x0.')

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

    def _create_learning_rule_applier(
            self, product_series: ProductSeries
    ) -> AbstractLearningRuleApplier:
        """Create a LearningRuleApplierFloat."""
        return LearningRuleApplierFloat(product_series)

    def _process_pre_spikes(self, s_in: np.ndarray) -> None:
        """Process pre-synaptic spikes.

        Four different modes of operation, based on GradedSpikeCfg value:
        (0) GradedSpikeCfg.USE_REGULAR_IMPULSE if a single pre-synaptic neuron
        spikes more than once, pre-traces are updated by their regular impulses.
        (1) GradedSpikeCfg.OVERWRITE overwrites the value of the pre-synaptic
        trace x1 by payload/2, upon spiking.

        Only in floating-pt:
            (2) & (3) GradedSpikeCfg.ADD_WITH_SATURATION &
            GradedSpikeCfg.ADD_WITHOUT_SATURATION have the same behavior:
            adds payload/2 to the pre-synaptic trace x1, upon spiking.
            x1 is not saturated.

        Within-epoch spike times are recorded.

        Parameters
        ----------
        s_in : ndarray
            Pre-synaptic spikes.
        """
        spiked = s_in.astype(bool)

        multi_spike_x = (self.tx > 0) & spiked

        scaled_activations = s_in / 2

        if self._graded_spike_cfg == GradedSpikeCfg.USE_REGULAR_IMPULSE:
            self.x1[multi_spike_x] += self._x_impulses[0]

        elif self._graded_spike_cfg == GradedSpikeCfg.OVERWRITE:
            self.x1[spiked] = scaled_activations[spiked]

        elif self._graded_spike_cfg == GradedSpikeCfg.ADD_WITH_SATURATION or \
                self._graded_spike_cfg == GradedSpikeCfg.ADD_WITHOUT_SATURATION:
            sums = self.x1 + scaled_activations

            self.x1 = sums

        self.x2[multi_spike_x] += self._x_impulses[1]

        ts_offset = self._within_epoch_time_step()
        self.tx[spiked] = ts_offset

    def _process_post_spikes(self, s_in_bap: np.ndarray) -> None:
        """Process post-synaptic spikes.

        Records within-epoch spiking times of post-synaptic neurons.
        If a single post-synaptic neuron spikes more than once,
        the corresponding trace is updated by its trace impulse value.

        Parameters
        ----------
        s_in_bap : ndarray
            Post-synaptic spikes.
        """

        self.y0[s_in_bap] = True
        multi_spike_y = (self.ty > 0) & s_in_bap

        y_traces = self._y_traces
        y_traces[:, multi_spike_y] += self._y_impulses[:, np.newaxis]
        self._set_y_traces(y_traces)

        ts_offset = self._within_epoch_time_step()
        self.ty[s_in_bap] = ts_offset

    def _compute_trace_histories(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Compute history of x and y trace values within the past epoch.

        Returns
        ----------
        x_traces_history : ndarray
            History of x trace values within the epoch.
        y_traces_history : np.ndarray
            History of y trace values within the epoch.
        """
        # Gather all necessary information to decay traces
        x_traces = self._x_traces
        y_traces = self._y_traces

        t_epoch = self._learning_rule.t_epoch

        x_impulses = self._x_impulses[:, np.newaxis]
        y_impulses = self._y_impulses[:, np.newaxis]

        x_taus = self._x_taus
        y_taus = self._y_taus

        # get spike times
        t_spike_x = self.tx
        t_spike_y = self.ty

        # most naive algorithm to decay traces
        x_traces_history = np.full((t_epoch + 1,) + x_traces.shape, np.nan,
                                   dtype=float)
        x_traces_history[0] = x_traces
        y_traces_history = np.full(
            (t_epoch + 1,) + y_traces.shape, np.nan, dtype=float
        )
        y_traces_history[0] = y_traces

        for t in range(1, t_epoch + 1):
            x_traces_history[t][x_taus != 0] = self._decay_trace(
                x_traces_history[t - 1][x_taus != 0],
                1,
                x_taus[x_taus != 0][:, np.newaxis],
            )
            y_traces_history[t][y_taus != 0] = self._decay_trace(
                y_traces_history[t - 1][y_taus != 0],
                1,
                y_taus[y_taus != 0][:, np.newaxis],
            )

            # add impulses if spike happens in this timestep
            x_spike_ids = np.where(t_spike_x == t)[0]
            x_traces_history[t][:, x_spike_ids] += x_impulses

            y_spike_ids = np.where(t_spike_y == t)[0]
            y_traces_history[t][:, y_spike_ids] += y_impulses

        return x_traces_history, y_traces_history

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

    def _apply_learning_rules(
        self, x_traces_history: np.ndarray, y_traces_history: np.ndarray
    ) -> None:
        """Update all synaptic variables according to the
        LearningRuleApplier representation of their corresponding
        learning rule."""
        applier_args = self._extract_applier_args(
            x_traces_history, y_traces_history
        )

        for syn_var_name, lr_applier in self._learning_rule_appliers.items():
            syn_var = getattr(self, syn_var_name).copy()
            if (isinstance(syn_var, csr_matrix)):
                dst, src, _ = find(syn_var, explicit_zeros=True)
                syn_var[dst, src] = lr_applier.apply(syn_var,
                                                     **applier_args)[dst, src]
            else:
                syn_var = lr_applier.apply(syn_var, **applier_args)
            syn_var = self._saturate_synaptic_variable(syn_var_name, syn_var)
            setattr(self, syn_var_name, syn_var)

    def _extract_applier_args(
        self, x_traces_history: np.ndarray, y_traces_history: np.ndarray
    ) -> dict:
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
        "evaluated_traces": see _extract_applier_evaluated_traces method
        for details.
        """

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
                if int(self.time_step / self._learning_rule.t_epoch) % 2 ** k
                == 0
                else 0
            )

            # Shape: (0, )
            applier_args["u"] = u

        evaluated_traces = self._extract_applier_evaluated_traces(
            x_traces_history, y_traces_history
        )
        applier_args.update(evaluated_traces)

        return applier_args

    def _saturate_synaptic_variable(
            self, synaptic_variable_name: str,
            synaptic_variable_values: np.ndarray
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
        # Weights and Tags
        if synaptic_variable_name == "weights":
            if self.sign_mode == SignMode.MIXED:
                return synaptic_variable_values
            elif self.sign_mode == SignMode.EXCITATORY:
                return np.maximum(0, synaptic_variable_values)
            elif self.sign_mode == SignMode.INHIBITORY:
                return np.minimum(0, synaptic_variable_values)
        # Delays
        elif synaptic_variable_name == "tag_1":
            return synaptic_variable_values
        elif synaptic_variable_name == "tag_2":
            if isinstance(synaptic_variable_values, csr_matrix):
                synaptic_variable_values[synaptic_variable_values < 0] = 0
                return synaptic_variable_values
            return np.maximum(0, synaptic_variable_values)
        else:
            raise ValueError(
                f"synaptic_variable_name can be 'weights', "
                f"'tag_1', or 'tag_2'."
                f"Got {synaptic_variable_name=}."
            )

    def _init_randoms(self) -> None:
        pass
