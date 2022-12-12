# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from abc import abstractmethod
import asteval

from lava.magma.core.learning.product_series import ProductSeries, Factor
from lava.magma.core.learning.constants import *


class AbstractLearningRuleApplier:
    """The LearningRuleApplier is a Python-specific representation of learning
    rules. It is associated with a ProductSeries.

    LearningRuleApplier implementations have to define an apply method, which
    tells how the learning rule represented by the associated ProductSeries,
    given states of Dependencies and Factors passed as arguments, is to be
    evaluated.

    Parameters
    ----------
    product_series: ProductSeries
        ProductSeries associated with this LearningRuleApplier.
    """
    def __init__(self, product_series: ProductSeries) -> None:
        self._product_series = product_series

    @abstractmethod
    def apply(self, init_accumulator: np.ndarray, **applier_args) -> np.ndarray:
        pass


class LearningRuleApplierFloat(AbstractLearningRuleApplier):
    """The LearningRuleFloatApplier is an implementation of
    AbstractLearningRuleApplier to be used with the
    PyFloatLearningDenseProcessModel.

    At initialization, it goes through the associated ProductSeries and derives
    a string representation of the learning rule where Dependencies and Factors
    are written in a way that is coherent with the names of the state variables
    they are associated to in the arguments of the apply method.

    This string is compiled at initialization and evaluated at every call to
    apply.

    Example:
    dw = "-2 * x0 * y1 + 4 * y0 * x1 + u0 * w"

    applier_str = "-2 * x0 * traces[0][2] + 4 * y0 * traces[1][0] + u * weights"
    """

    def __init__(self, product_series: ProductSeries) -> None:
        super().__init__(product_series)

        self._applier_str = self._build_applier_str()
        self._applier_compiled = compile(self._applier_str, "<string>", "eval")
        self.eval_func = asteval.Interpreter()

    def _build_applier_str(self) -> str:
        """Build the string representation of the LearningRuleFloatApplier
        out of the ProductSeries.

        Returns
        ----------
        applier_str: str
            String representation associated to this LearningRuleFloatApplier.
        """
        # initialize empty applier_str for the ProductSeries
        applier_str = ""

        # for each Product in the ProductSeries
        for product in self._product_series.products:
            # initialize empty applier_sub_str for the Product
            applier_sub_str = ""

            # derive scaling factor string representation
            sf_str = f"({product.s_mantissa}) * 2. ** (({product.s_exp}) - 7)"
            # derive dependency string representation
            dep_str = f"{product.dependency}"

            # append sf_str and dep_str to applier_sub_str
            applier_sub_str = applier_sub_str + sf_str + " * " + dep_str + " * "

            # for each Factor in the Product
            for factor in product.factors:
                # derive factor string representation
                factor_str = self._extract_factor_str(
                    factor, product.dependency
                )

                # append factor_str to applier_sub_str
                applier_sub_str = applier_sub_str + factor_str + " * "

            # remove dangling " * " at the end of applier_sub_str
            applier_sub_str = applier_sub_str[:-3]

            # append applier_sub_str to applier_str
            applier_str = applier_str + applier_sub_str + " + "

        # remove dangling " * " at the end of applier_str
        applier_str = applier_str[:-3]

        return applier_str

    @staticmethod
    def _extract_factor_str(factor: Factor, dep: str) -> str:
        """Get the LearningRuleFloatApplier-string representation
        of a single Factor, given the dependency it appears with.

        Returns
        ----------
        factor_str: str
            LearningRuleFloatApplier-string representation of the Factor.
        """
        # if factor is a C Factor, return "{factor.const}"
        if factor.state_var == "C":
            factor_str = f"({factor.const})"
            return factor_str

        if factor.state_var in str_symbols.TRACES:
            factor_str = factor.state_var + "_" + dep

        # if factor is a Synaptic Variable Factor
        elif factor.state_var in str_symbols.SYNAPTIC_VARIABLES:
            factor_str = str_symbols.SYNAPTIC_VARIABLE_VAR_MAPPING[
                factor.state_var
            ]

        else:
            raise Exception("Unknown Factor in LearningRuleFloatApplier.")

        # if factor has a constant, add " + {factor.const}" to factor_str
        if factor.has_const():
            factor_str = f"({factor_str} + ({factor.const}))"

        # if factor is a sign Factor, wrap factor_str in np.sign() call
        if factor.is_sgn:
            factor_str = f"np.sign({factor_str})"

        return factor_str

    def apply(self, init_accumulator: np.ndarray, **applier_args) -> np.ndarray:
        """Apply the learning rule represented by this LearningRuleFloatApplier.

        When called from the PyFloatLearningDenseProcessModel, applier_args
        contains variables with the following names :
        {"x0", "y0", "u", "weights", "tag_2", "tag_1", "np", "traces"}

        All variables apart from "u", "np" are numpy arrays.

        "u" is a scalar.
        "np" is a reference to numpy as it is needed for the evaluation of
        "np.sign()" types of call inside the applier string.

        Parameters
        ----------
        init_accumulator: np.ndarray
            Values of the synaptic variable before learning rule application.

        Returns
        ----------
        result: np.ndarray
            Values of the synaptic variable after learning rule application.
        """
        self.eval_func.symtable = applier_args

        # e.g: result = w + evaluation(dw)
        result = init_accumulator + self.eval_func(self._applier_str)

        return result


class LearningRuleApplierBitApprox(AbstractLearningRuleApplier):
    """The LearningRuleFixedApplier is an implementation of
    AbstractLearningRuleApplier to be used with the
    PyFixedLearningDenseProcessModel.

    Contrary to LearningRuleFloatApplier, there is no applier string constructed
    at initialization for LearningRuleFixedApplier.
    The apply method has to loop through all Products/Factors of the associated
    ProductSeries and accumulate results of synaptic variable update computation
    along the way.

    This is due to the fact that it is not straightforward to construct such a
    string, in the fixed-point case, as there are intermediary stateful
    bit-shifts happening between steps of the computation, which can't be
    translated to string operations.
    """

    @staticmethod
    def _compute_factor(dependency: str, factor: Factor, applier_args: dict):
        """Evaluate a factor based on a dependency, a Factor and the state
        variables involved in it.

        Parameters
        ----------
        dependency : str
            Dependency gating factor evaluation.
        factor : Factor
            Factor representation of the factor to be evaluated.

        Returns
        ----------
        factor_val : ndarray
            Computed value for the given dependency and Factor.
        """
        # get constant of factor if it is not None, otherwise set it to 0
        const = factor.const if factor.has_const() else 0

        # handle factor of type Dependency
        if factor.state_var in str_symbols.DEPENDENCIES:
            return applier_args[factor.state_var] + const

        # handle factor of type Trace
        if factor.state_var in str_symbols.TRACES:
            return applier_args[f"{factor.state_var}_{dependency}"] + const

        # handle factor of type Synaptic variable
        if (
            factor.state_var in str_symbols.SYNAPTIC_VARIABLES
            and not factor.is_sgn
        ):
            attr_name = str_symbols.SYNAPTIC_VARIABLE_VAR_MAPPING[
                factor.state_var
            ]

            return applier_args[attr_name] + const

        # handle factor of type Sign of Synaptic Variable
        if factor.state_var in str_symbols.SYNAPTIC_VARIABLES and factor.is_sgn:
            attr_name = str_symbols.SYNAPTIC_VARIABLE_VAR_MAPPING[
                factor.state_var
            ]

            return np.sign(applier_args[attr_name] + const)

        # handle factor of type Constant
        if factor.state_var == str_symbols.C:
            return const

    def apply(self, init_accumulator: np.ndarray, **applier_args) -> np.ndarray:
        """Apply the learning rule represented by this LearningRuleFixedApplier.

        When called from the PyFixedLearningDenseProcessModel, applier_args
        contains variables with the following names :
        {"shape", "x0", "y0", "u", "weights", "tag_2", "tag_1",
        "x_traces", "y_traces"}

        All variables apart from "shape", "u" are numpy arrays.
        "shape" is a tuple.
        "u" is a scalar.

        Parameters
        ----------
        init_accumulator: np.ndarray
            Shifted values of the synaptic variable before learning rule
            application.

        Returns
        ----------
        result: np.ndarray
            Shifted values of the synaptic variable after learning rule
            application.
        """
        # result: addition accumulator, initialize to left-shifted values of
        # synaptic variable before learning rule application
        # bit width: 15b
        result = init_accumulator

        # for each Product in ProductSeries
        for product in self._product_series.products:
            # initialize multiplication accumulator (MAC)
            current_result = np.ones(applier_args["shape"], dtype=np.int32)

            # multiply MAC with dependency state variable from applier_args
            current_result *= applier_args[product.dependency]

            # sum of factor bit widths
            factor_width_sum = 0

            # for each Factor in Product
            for factor in product.factors:
                # add factor bit width to sum of factor bit widths
                factor_width = FACTOR_TO_WIDTH_DICT[factor.state_var]  # BITS
                factor_width_sum += factor_width

                # get value of factor from applier_args
                factor_val = self._compute_factor(
                    product.dependency, factor, applier_args
                )

                # saturate value of factor (after constant addition)
                factor_val = np.clip(factor_val,
                                     a_min=-(2 ** factor_width),
                                     a_max=2 ** factor_width - 1)

                # multiply MAC with value of factor
                current_result *= factor_val.astype(np.int32)

            # add bit width of mantissa of scaling factor to
            # sum of factor bit widths
            factor_width_sum += W_S_MANT

            # multiply MAC with mantissa of scaling factor
            current_result *= product.s_mantissa

            # right-shift MAC back to 15b, in order to avoid overflows from
            # all the multiplications
            shift = np.maximum(0, factor_width_sum - W_ACCUMULATOR_U)
            current_result = np.right_shift(current_result, shift)

            # left-shift by the exponent of scaling factor
            current_result = np.left_shift(current_result, product.s_exp)

            # saturate current_result
            current_result = np.clip(current_result,
                                     a_min=-(2 ** W_ACCUMULATOR_U) - 1,
                                     a_max=2 ** W_ACCUMULATOR_U - 1)

            # accumulate and saturate result
            result = result + current_result
            result = np.clip(result,
                             a_min=-(2 ** W_ACCUMULATOR_U) - 1,
                             a_max=2 ** W_ACCUMULATOR_U - 1)

        return result
