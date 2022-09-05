# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
#
# Copyright © 2021-2022 Intel Corporation.
#
# This software and the related documents are Intel copyrighted
# materials, and your use of them is governed by the express
# license under which they were provided to you (License). Unless
# the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit  this software or the
# related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with
# no express or implied warranties, other than those that are
# expressly stated in the License.

import typing as ty
import re

import lava.magma.core.learning.string_symbols as str_symbols
from lava.magma.core.learning.symbolic_equation import SymbolicEquation
from lava.magma.core.learning.product_series import ProductSeries


class LearningRule:
    pass


class LoihiLearningRule(LearningRule):
    """Encapsulation of learning-related information according to: Loihi

    A LearningRule object has the following main objectives:
    (1) Given string representations of learning rules (equations) describing
    dynamics of the three synaptic variables (weight, delay, tag),
    generate adequate ProductSeries representations and store them.

    (2) Store other learning-related information such as:
    impulse values by which to update traces upon spikes,
    time constants by which to decay traces over time,
    the length of the learning epoch,
    the set of dependencies used by all specified learning rules and
    the set of traces used by all specified learning rules.

    From the user's perspective, a LearningRule object is to be used as follows:
    (1) Instantiate a LearningRule object with learning rules given in string
    format for all three synaptic variables (dw, dd, dt), as well as trace
    configuration parameters (impulse, decay) for all available traces
    (x1, x2, y1), and the learning epoch length.

    (2) The LearningRule object encapsulating learning-related information is
    then passed to the LearningConn Process as instantiation argument.

    (3) It will internally be used by ProcessModels to derive the operations
    to be executed in the learning phase (Py and Nc).

    Attributes
    ----------
    decimate_exponent: int
        Decimate exponent used in uk dependencies, if any.
    x1_impulse: float
        Impulse by which x1 increases upon each pre-synaptic spike.
    x2_impulse: float
        Impulse by which x2 increases upon each pre-synaptic spike.
    y1_impulse: float
        Impulse by which y1 increases upon each post-synaptic spike.
    x1_tau: int
        Time constant by which x1 trace decays exponentially over time.
    x2_tau: int
        Time constant by which x2 trace decays exponentially over time.
    y1_tau: int
        Time constant by which y1 trace decays exponentially over time.
    t_epoch: int
        Duration of learning epoch.
    dw: ProductSeries
        ProductSeries representation of synaptic weight learning rule.
    dd: ProductSeries
        ProductSeries representation of synaptic delay learning rule.
    dt: ProductSeries
        ProductSeries representation of synaptic tag learning rule.
    active_product_series: dict
        Dict containing ProductSeries for active learning rules.
    active_traces_per_dependency: dict
        Dict mapping active traces to the set of dependencies they appear with.
    active_traces : set
        Set of all active traces.
    """

    def __init__(
        self,
        dw: ty.Optional[str] = None,
        dd: ty.Optional[str] = None,
        dt: ty.Optional[str] = None,
        x1_impulse: float = 0.0,
        x1_tau: int = 0,
        x2_impulse: float = 0.0,
        x2_tau: int = 0,
        y1_impulse: float = 0.0,
        y1_tau: int = 0,
        t_epoch: int = 1,
    ) -> None:
        str_learning_rules = {
            str_symbols.DW: dw,
            str_symbols.DD: dd,
            str_symbols.DT: dt,
        }

        active_str_learning_rules = {
            key: str_learning_rule
            for key, str_learning_rule in str_learning_rules.items()
            if str_learning_rule is not None
        }

        self._validate_at_least_one_learning_rule(active_str_learning_rules)
        self._decimate_exponent = self._validate_uk(active_str_learning_rules)

        self._x1_impulse = self._validate_impulse(x1_impulse)
        self._x2_impulse = self._validate_impulse(x2_impulse)
        self._x1_tau = self._validate_tau(x1_tau)
        self._x2_tau = self._validate_tau(x2_tau)

        self._y1_impulse = self._validate_impulse(y1_impulse)
        self._y1_tau = self._validate_tau(y1_tau)

        self._t_epoch = self._validate_t_epoch(t_epoch)

        self._active_product_series = {
            key: self._generate_product_series_from_string(
                key, str_learning_rule
            )
            for key, str_learning_rule in active_str_learning_rules.items()
            if str_learning_rule is not None
        }

        self._dw = self._dd = self._dt = None
        for key, product_series in self._active_product_series.items():
            setattr(self, f"_{key}", product_series)

        (
            self._active_traces,
            self._active_traces_per_dependency,
        ) = self._get_active_traces_from_active_product_series()

    @property
    def x1_impulse(self) -> float:
        """Get the impulse value for x1 trace.

        Returns
        ----------
        x1_impulse : float
            Impulse value for x1 trace.
        """
        return self._x1_impulse

    @property
    def x2_impulse(self) -> float:
        """Get the impulse value for x2 trace.

        Returns
        ----------
        x2_impulse : float
            Impulse value for x2 trace.
        """
        return self._x2_impulse

    @property
    def y1_impulse(self) -> float:
        """Get the impulse value for y1 trace.

        Returns
        ----------
        y1_impulse : float
            Impulse value for y1 trace.
        """
        return self._y1_impulse

    @property
    def x1_tau(self) -> int:
        """Get the tau value for x1 trace.

        Returns
        ----------
        x1_tau : int
            Tau value for x1 trace.
        """
        return self._x1_tau

    @property
    def x2_tau(self) -> int:
        """Get the tau value for x2 trace.

        Returns
        ----------
        x2_tau : int
            Tau value for x2 trace.
        """
        return self._x2_tau

    @property
    def y1_tau(self) -> int:
        """Get the tau value for y1 trace.

        Returns
        ----------
        y1_tau : int
            Tau value for y1 trace.
        """
        return self._y1_tau

    @property
    def t_epoch(self) -> int:
        """Get the epoch length.

        Returns
        ----------
        t_epoch : int
            Epoch length.
        """
        return self._t_epoch

    @property
    def dw(self) -> ty.Optional[ProductSeries]:
        """Get the ProductSeries associated with the "dw" target.

        Returns
        ----------
        dw : ProductSeries, optional
            ProductSeries associated with the "dw" target.
        """
        return self._dw

    @property
    def dd(self) -> ty.Optional[ProductSeries]:
        """Get the ProductSeries associated with the "dd" target.

        Returns
        ----------
        dd : ProductSeries, optional
            ProductSeries associated with the "dd" target.
        """
        return self._dd

    @property
    def dt(self) -> ty.Optional[ProductSeries]:
        """Get the ProductSeries associated with the "dt" target.

        Returns
        ----------
        dt : ProductSeries, optional
            ProductSeries associated with the "dt" target.
        """
        return self._dt

    @property
    def decimate_exponent(self) -> ty.Optional[int]:
        """Get the decimate exponent of this LearningRule.

        Returns
        ----------
        decimate_exponent : int, optional
            Decimate exponent of this LearningRule.
        """
        return self._decimate_exponent

    @property
    def active_product_series(self) -> ty.Dict[str, ProductSeries]:
        """Get the active ProductSeries dict, containing ProductSeries
        associated to string learning rules that were not None.

        Mapped by target name: either one of (dw, dd, dt)

        Returns
        ----------
        active_product_series : dict
            Active ProductSeries dict.
        """
        return self._active_product_series

    @property
    def active_traces_per_dependency(self) -> ty.Dict[str, ty.Set[str]]:
        """Get the dict of active traces per dependency associated with
        this LearningRule.

        Returns
        ----------
        active_traces_per_dependency : dict
            Set of active traces per dependency in the list of ProductSeries.
        """
        return self._active_traces_per_dependency

    @property
    def active_traces(self) -> ty.Set[str]:
        """Get the set of all active traces in this LearningRule.

        Returns
        ----------
        active_traces : set
            Set of all active traces.
        """
        return self._active_traces

    @staticmethod
    def _validate_at_least_one_learning_rule(
        active_str_learning_rules: ty.Dict[str, str]
    ) -> None:
        """Validate that the dictionary of active learning rules contains at
        least one element.

        Parameters
        ----------
        active_str_learning_rules : dict
            dictionary containing active learning rules
        """
        if len(active_str_learning_rules) == 0:
            raise ValueError(
                "No learning rule was specified. "
                "At least one learning rule should be specified."
            )

    @staticmethod
    def _validate_uk(
        active_str_learning_rules: ty.Dict[str, str]
    ) -> ty.Optional[int]:
        """Validate that the list of learning rules contains at most one
        decimation factor if any uk terms are used.

        Return the decimate exponent of the found uk.

        Parameters
        ----------
        active_str_learning_rules : dict
            dictionary containing active learning rules

        Returns
        -------
        decimate_exponent : int
            Decimate exponent of the found uk term
        """
        uk_list = []
        p = re.compile("u[0-9]+")

        for rule in active_str_learning_rules.values():
            uk_list.extend(p.findall(rule))

        uk_set = set(uk_list)

        decimate_exponent = None

        if len(uk_set) == 1:
            decimate_exponent = int(uk_list[0])
        elif len(uk_set) > 1:
            raise ValueError(
                "Learning rules (dw, dd, dt) cannot contain uk terms "
                "with different decimation factors k."
            )

        return decimate_exponent

    @staticmethod
    def _validate_impulse(impulse: float) -> float:
        """Validate that an impulse value is allowed.

        Parameters
        ----------
        impulse : float
            Impulse by which trace increases upon each pre-synaptic spike.

        Returns
        ----------
        impulse : float
            Impulse by which trace increases upon each pre-synaptic spike.
        """
        if impulse < 0:
            raise ValueError(
                f"Impulse value cannot be negative for trace impulse value."
            )

        return impulse

    @staticmethod
    def _validate_tau(tau: int) -> int:
        """Validate that a decay time constant is allowed.

        Parameters
        ----------
        tau : int
            Time constant by which a trace decays exponentially over time.

        Returns
        ----------
        tau : int
            Time constant by which a trace decays exponentially over time.
        """
        if tau < 0:
            raise ValueError(f"Decay time constant cannot be negative.")

        return tau

    @staticmethod
    def _validate_t_epoch(t_epoch: int) -> int:
        """Validate that a learning epoch length is allowed.

        Parameters
        ----------
        t_epoch : int
            Duration of learning epoch.

        Returns
        ----------
        t_epoch : int
            Duration of learning epoch.
        """
        if t_epoch <= 0:
            raise ValueError(f"Learning epoch length cannot be negative.")

        return t_epoch

    def _get_active_traces_from_active_product_series(
        self,
    ) -> ty.Tuple[ty.Set[str], ty.Dict[str, ty.Set[str]]]:
        """Find set of active traces associated with all dependencies in a
        list of ProductSeries.

        Return a tuple of two different representation for the set of
        active traces, active_traces and active_traces_per_dependency

        Example:

        For
        list_product_series = [
        ProductSeries(y0 * x1 * y2 + x0 * y1 * y2),
        ProductSeries(u * x2 * y2 * y3)
        ]

        active_traces = {"x1", "x2", "y1", "y2", "y3"}

        active_traces_per_dependency = {
            "x0": {"y1", "y2"},
            "y0": {"x1", "y2"},
            "u": {"x2", "y2", "y3"}
        }

        Returns
        ----------
        active_traces : set
            Set of all active traces.
        active_traces_per_dependency : dict
            Set of active traces per dependency in the set of active
            ProductSeries.
        """
        active_traces = set()
        active_traces_per_dependency = {}

        for product_series in self._active_product_series.values():
            for (
                dependency,
                traces,
            ) in product_series.active_traces_per_dependency.items():
                active_traces = active_traces.union(traces)

                if dependency not in active_traces_per_dependency.keys():
                    active_traces_per_dependency[dependency] = traces
                else:
                    active_traces_per_dependency[
                        dependency
                    ] = active_traces_per_dependency[dependency].union(traces)

        return active_traces, active_traces_per_dependency

    @staticmethod
    def _generate_product_series_from_string(
        target: str, str_learning_rule: str
    ) -> ProductSeries:
        """Generate a ProductSeries representation from a string representation.

        Parameters
        ----------
        target : str
            Left-hand side of learning rule equation.
            Either one of (dw, dd, dt).
        str_learning_rule : str
            Learning rule in string representation.

        Returns
        ----------
        product_series : ProductSeries
            Learning rule in ProductSeries representation.
        """
        symbolic_equation = SymbolicEquation(target, str_learning_rule)

        product_series = ProductSeries(symbolic_equation)

        return product_series


class Loihi1LearningRule(LoihiLearningRule):
    def __init__(
        self,
        dw: ty.Optional[str] = None,
        dd: ty.Optional[str] = None,
        dt: ty.Optional[str] = None,
        x1_impulse: float = 0.0,
        x1_tau: int = 0,
        x2_impulse: float = 0.0,
        x2_tau: int = 0,
        y1_impulse: float = 0.0,
        y1_tau: int = 0,
        y2_impulse: float = 0.0,
        y2_tau: int = 0,
        y3_impulse: float = 0.0,
        y3_tau: int = 0,
        t_epoch: int = 1,
    ) -> None:
        """Encapsulation of learning-related information according to: Loihi1

        A LearningRule object has the following main objectives:
        (1) Given string representations of learning rules (equations)
            describing dynamics of the three synaptic variables
            (weight, delay, tag), generate adequate ProductSeries
            representations and store them.

        (2) Store other learning-related information such as:
            impulse values by which to update traces upon spikes,
            time constants by which to decay traces over time,
            the length of the learning epoch,
            the set of dependencies used by all specified learning rules and
            the set of traces used by all specified learning rules.

        From the user's perspective, a LearningRule object is to be used as
        follows:

        (1) Instantiate a LearningRule object with learning rules given in
            string format for all three synaptic variables (dw, dd, dt), as
            well as trace configuration parameters (impulse, decay) for all
            available traces (x1, x2, y1, y2, y3), and the learning epoch
            length.

        (2) The LearningRule object encapsulating learning-related information
            is then passed to the LearningConn Process as instantiation
            argument.

        (3) It will internally be used by ProcessModels to derive the
            operations to be executed in the learning phase (Py and Nc).

        Attributes
        ----------
        decimate_exponent: int
            Decimate exponent used in uk dependencies, if any.
        x1_impulse: float
            Impulse by which x1 increases upon each pre-synaptic spike.
        x2_impulse: float
            Impulse by which x2 increases upon each pre-synaptic spike.
        y1_impulse: float
            Impulse by which y1 increases upon each post-synaptic spike.
        y2_impulse: float
            Impulse by which y2 increases upon each post-synaptic spike.
        y3_impulse: float
            Impulse by which y3 increases upon each post-synaptic spike.
        x1_tau: int
            Time constant by which x1 trace decays exponentially over time.
        x2_tau: int
            Time constant by which x2 trace decays exponentially over time.
        y1_tau: int
            Time constant by which y1 trace decays exponentially over time.
        y2_tau: int
            Time constant by which y2 trace decays exponentially over time.
        y3_tau: int
            Time constant by which y3 trace decays exponentially over time.
        t_epoch: int
            Duration of learning epoch.
        dw: ProductSeries
            ProductSeries representation of synaptic weight learning rule.
        dd: ProductSeries
            ProductSeries representation of synaptic delay learning rule.
        dt: ProductSeries
            ProductSeries representation of synaptic tag learning rule.
        active_product_series: dict
            Dict containing ProductSeries for active learning rules.
        active_traces_per_dependency: dict
            Dict mapping active traces to the set of dependencies they appear
            with.
        active_traces : set
            Set of all active traces.
        """

        self._y2_impulse = self._validate_impulse(y2_impulse)
        self._y3_impulse = self._validate_impulse(y3_impulse)

        self._y2_tau = self._validate_tau(y2_tau)
        self._y3_tau = self._validate_tau(y3_tau)
        super().__init__(
            dw=dw,
            dd=dd,
            dt=dt,
            x1_impulse=x1_impulse,
            x1_tau=x1_tau,
            x2_impulse=x2_impulse,
            x2_tau=x2_tau,
            y1_impulse=y1_impulse,
            y1_tau=y1_tau,
            t_epoch=t_epoch,
        )

    @property
    def y2_tau(self) -> int:
        """Get the tau value for y2 trace.

        Returns
        ----------
        y2_tau : int
            Tau value for y2 trace.
        """
        return self._y2_tau

    @property
    def y3_tau(self) -> int:
        """Get the tau value for y3 trace.

        Returns
        ----------
        y3_tau : int
            Tau value for y3 trace.
        """
        return self._y3_tau

    @property
    def y2_impulse(self) -> float:
        """Get the impulse value for y2 trace.

        Returns
        ----------
        y2_impulse : float
            Impulse value for y2 trace.
        """
        return self._y2_impulse

    @property
    def y3_impulse(self) -> float:
        """Get the impulse value for y3 trace.

        Returns
        ----------
        y3_impulse : float
            Impulse value for y3 trace.
        """
        return self._y3_impulse


class Loihi2LearningRule(LoihiLearningRule):
    pass
