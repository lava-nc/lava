# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

import lava.magma.core.learning.string_symbols as str_symbols
from lava.magma.core.learning.symbolic_equation import SymbolicEquation
from lava.magma.core.learning.product_series import Factor, Product, \
    ProductSeries


class TestFactor(unittest.TestCase):
    def _sub_test_regular_factor(self, state_var: str) -> None:
        """Atomic test for creating simple Factor objects with "regular"
        factor state vars."""
        factor_type = state_var + "+C"

        factor = Factor(state_var=state_var)

        self.assertIsInstance(factor, Factor)
        self.assertEqual(factor.state_var, state_var)
        self.assertEqual(factor.const, None)
        self.assertEqual(factor.is_sgn, False)
        self.assertEqual(factor.factor_type, factor_type)

    def _sub_test_w_t_factor(self, state_var: str) -> None:
        """Atomic test for creating simple Factor objects with "exception"
        factor state vars."""
        factor_type = state_var + "+2C"

        factor = Factor(state_var=state_var)

        self.assertIsInstance(factor, Factor)
        self.assertEqual(factor.state_var, state_var)
        self.assertEqual(factor.const, None)
        self.assertEqual(factor.is_sgn, False)
        self.assertEqual(factor.factor_type, factor_type)

    def test_regular_factors(self) -> None:
        """Tests creating simple Factor objects for each of the "regular"
        factor state vars."""
        excluded_factors = {'w', 't', 'C'}

        for state_var in str_symbols.FACTOR_STATE_VARS:
            if state_var not in excluded_factors:
                self._sub_test_regular_factor(state_var)

    def test_exception_factors(self) -> None:
        """Tests creating simple Factor objects for each of the "exception"
        factor state vars."""
        self._sub_test_w_t_factor("w")
        self._sub_test_w_t_factor("t")

    def test_factor_sgn(self) -> None:
        """Tests creating sgn Factor objects for each of the synaptic variable
        factor state vars."""
        state_var = "w"
        factor_type = f"sgn({state_var}+2C)"

        factor = Factor(state_var=state_var, is_sgn=True)

        self.assertIsInstance(factor, Factor)
        self.assertEqual(factor.state_var, state_var)
        self.assertEqual(factor.const, None)
        self.assertEqual(factor.is_sgn, True)
        self.assertEqual(factor.factor_type, factor_type)

        state_var = "d"
        factor_type = f"sgn({state_var}+C)"

        factor = Factor(state_var=state_var, is_sgn=True)

        self.assertIsInstance(factor, Factor)
        self.assertEqual(factor.state_var, state_var)
        self.assertEqual(factor.const, None)
        self.assertEqual(factor.is_sgn, True)
        self.assertEqual(factor.factor_type, factor_type)

        state_var = "t"
        factor_type = f"sgn({state_var}+2C)"

        factor = Factor(state_var=state_var, is_sgn=True)

        self.assertIsInstance(factor, Factor)
        self.assertEqual(factor.state_var, state_var)
        self.assertEqual(factor.const, None)
        self.assertEqual(factor.is_sgn, True)
        self.assertEqual(factor.factor_type, factor_type)

    def test_unknown_factor(self) -> None:
        """Tests that instantiating Factor objects for a factor state var
        that unknown throws an error."""
        state_var = "n"

        with self.assertRaises(ValueError):
            Factor(state_var=state_var)

    def test_invalid_factor_sgn(self) -> None:
        """Tests that instantiating sgn Factor objects for a factor state var
        that is not a synaptic variable throws an error."""
        state_var = "x1"

        with self.assertRaises(ValueError):
            Factor(state_var=state_var, is_sgn=True)

    def test_factor_plus_const(self) -> None:
        """Tests creating Factor object with a constant added."""
        state_var = "x1"
        const = 5
        factor_type = state_var + "+C"

        factor = Factor(state_var=state_var, const=const)

        self.assertIsInstance(factor, Factor)
        self.assertEqual(factor.state_var, state_var)
        self.assertEqual(factor.const, const)
        self.assertEqual(factor.is_sgn, False)
        self.assertEqual(factor.factor_type, factor_type)

    def test_factor_const(self) -> None:
        """Tests creating Factor object with C as factor state var."""
        state_var = "C"
        const = 5
        factor_type = state_var

        factor = Factor(state_var=state_var, const=const)

        self.assertIsInstance(factor, Factor)
        self.assertEqual(factor.state_var, state_var)
        self.assertEqual(factor.const, const)
        self.assertEqual(factor.is_sgn, False)
        self.assertEqual(factor.factor_type, factor_type)


class TestProduct(unittest.TestCase):
    def test_product(self) -> None:
        """Tests creating a valid Product object."""
        target = "dw"
        dependency = "y0"
        s_mantissa = 1
        s_exp = 0

        factor_1 = Factor(state_var="x1", const=2)
        factor_2 = Factor(state_var="w", is_sgn=True)
        factor_list = [factor_1, factor_2]

        product = Product(target, dependency, s_mantissa, s_exp,
                          factor_list)

        self.assertIsInstance(product, Product)
        self.assertEqual(product.target, target)
        self.assertEqual(product.dependency, dependency)
        self.assertEqual(product.s_mantissa, s_mantissa)
        self.assertEqual(product.s_exp, s_exp)
        self.assertEqual(product.decimate_exponent, None)

    def test_product_u_dep_and_decimate_exponent(self) -> None:
        """Tests creating a valid Product object with u dependency and
        a decimate exponent."""
        target = "dw"
        dependency = "u"
        s_mantissa = 1
        s_exp = 0
        decimate_exponent = 2

        factor_1 = Factor(state_var="x1", const=2)
        factor_2 = Factor(state_var="w", is_sgn=True)
        factor_list = [factor_1, factor_2]

        product = Product(target, dependency, s_mantissa, s_exp,
                          factor_list, decimate_exponent)

        self.assertIsInstance(product, Product)
        self.assertEqual(product.target, target)
        self.assertEqual(product.dependency, dependency)
        self.assertEqual(product.s_mantissa, s_mantissa)
        self.assertEqual(product.s_exp, s_exp)
        self.assertEqual(product.decimate_exponent, decimate_exponent)

    def test_product_u_dep_and_no_decimate_exponent(self) -> None:
        """Tests that creating a Product object with u dependency without
        a decimate exponent throws an error."""
        target = "dw"
        dependency = "u"
        s_mantissa = 1
        s_exp = 0

        factor_1 = Factor(state_var="x1", const=2)
        factor_2 = Factor(state_var="w", is_sgn=True)
        factor_list = [factor_1, factor_2]

        with self.assertRaises(ValueError):
            Product(target, dependency, s_mantissa, s_exp, factor_list)

    def test_product_no_u_dep_and_decimate_exponent(self) -> None:
        """Tests that creating a Product object without u dependency
        but with a decimate exponent throws an error."""
        target = "dw"
        dependency = "x0"
        s_mantissa = 1
        s_exp = 0
        decimate_exponent = 2

        factor_1 = Factor(state_var="x1", const=2)
        factor_2 = Factor(state_var="w", is_sgn=True)
        factor_list = [factor_1, factor_2]

        with self.assertRaises(ValueError):
            Product(target, dependency, s_mantissa, s_exp, factor_list,
                    decimate_exponent)

    def test_product_unknown_target(self) -> None:
        """Tests that creating a Product object with an unknown target
        throws an error."""
        target = "n"
        dependency = "x0"
        s_mantissa = 1
        s_exp = 0

        factor_1 = Factor(state_var="x1", const=2)
        factor_2 = Factor(state_var="w", is_sgn=True)
        factor_list = [factor_1, factor_2]

        with self.assertRaises(ValueError):
            Product(target, dependency, s_mantissa, s_exp, factor_list)

    def test_product_unknown_dependency(self) -> None:
        """Tests that creating a Product object with an unknown dependency
        throws an error."""
        target = "dw"
        dependency = "n"
        s_mantissa = 1
        s_exp = 0

        factor_1 = Factor(state_var="x1", const=2)
        factor_2 = Factor(state_var="w", is_sgn=True)
        factor_list = [factor_1, factor_2]

        with self.assertRaises(ValueError):
            Product(target, dependency, s_mantissa, s_exp, factor_list)


class TestProductSeries(unittest.TestCase):
    def test_product_series(self) -> None:
        """Tests creating a valid ProductSeries object."""
        target = "dw"
        str_learning_rule = 'x0*(-1)*2^-1*y1 + y0*1*2^1*x1'

        se = SymbolicEquation(target, str_learning_rule)
        product_series = ProductSeries(se)

        self.assertIsInstance(product_series, ProductSeries)
        self.assertEqual(product_series.target, target)
        self.assertEqual(product_series.decimate_exponent, None)
        self.assertEqual(len(product_series.products), 2)
        self.assertDictEqual(product_series.active_traces_per_dependency, {
            "x0": {"y1"},
            "y0": {"x1"}
        })

        product_1 = product_series.products[0]
        product_2 = product_series.products[1]

        self.assertIsInstance(product_1, Product)
        self.assertEqual(product_1.target, target)
        self.assertEqual(product_1.dependency, "x0")
        self.assertEqual(product_1.s_mantissa, -1)
        self.assertEqual(product_1.s_exp, 6)
        self.assertEqual(product_1.decimate_exponent, None)
        self.assertEqual(product_1.factors[0].state_var, "y1")
        self.assertEqual(product_1.factors[0].const, None)
        self.assertEqual(product_1.factors[0].is_sgn, False)

        self.assertIsInstance(product_2, Product)
        self.assertEqual(product_2.target, target)
        self.assertEqual(product_2.dependency, "y0")
        self.assertEqual(product_2.s_mantissa, 1)
        self.assertEqual(product_2.s_exp, 8)
        self.assertEqual(product_2.decimate_exponent, None)
        self.assertEqual(product_2.factors[0].state_var, "x1")
        self.assertEqual(product_2.factors[0].const, None)
        self.assertEqual(product_2.factors[0].is_sgn, False)

    def test_product_series_u_dep(self) -> None:
        """Tests creating a valid ProductSeries object with u dependency."""
        target = "dw"
        str_learning_rule = 'u4*(-1)*2^-1*y1 + y0*1*2^1*x1'

        se = SymbolicEquation(target, str_learning_rule)
        product_series = ProductSeries(se)

        self.assertIsInstance(product_series, ProductSeries)
        self.assertEqual(product_series.target, target)
        self.assertEqual(product_series.decimate_exponent, 4)
        self.assertEqual(len(product_series.products), 2)
        self.assertDictEqual(product_series.active_traces_per_dependency, {
            "u": {"y1"},
            "y0": {"x1"}
        })

        product_1 = product_series.products[0]
        product_2 = product_series.products[1]

        self.assertIsInstance(product_1, Product)
        self.assertEqual(product_1.target, target)
        self.assertEqual(product_1.dependency, "u")
        self.assertEqual(product_1.s_mantissa, -1)
        self.assertEqual(product_1.s_exp, 6)
        self.assertEqual(product_1.decimate_exponent, 4)
        self.assertEqual(product_1.factors[0].state_var, "y1")
        self.assertEqual(product_1.factors[0].const, None)
        self.assertEqual(product_1.factors[0].is_sgn, False)

        self.assertIsInstance(product_2, Product)
        self.assertEqual(product_2.target, target)
        self.assertEqual(product_2.dependency, "y0")
        self.assertEqual(product_2.s_mantissa, 1)
        self.assertEqual(product_2.s_exp, 8)
        self.assertEqual(product_2.decimate_exponent, None)
        self.assertEqual(product_2.factors[0].state_var, "x1")
        self.assertEqual(product_2.factors[0].const, None)
        self.assertEqual(product_2.factors[0].is_sgn, False)

    def test_product_series_u_dep_multi_decimate_exponent(self) -> None:
        """Tests that creating a ProductSeries object with u dependency with
        multiple decimate exponent throws an error."""
        target = "dw"
        str_learning_rule = 'u4*(-1)*2^-1*y1 + u5*1*2^1*x1'

        se = SymbolicEquation(target, str_learning_rule)

        with self.assertRaises(ValueError):
            ProductSeries(se)


if __name__ == "__main__":
    unittest.main()
