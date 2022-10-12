# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.precision import Precision
import numpy as np


class TestLavaPyType(unittest.TestCase):

    def test_lavapytype_constructor(self):
        """Check if LavaPyType is correctly instantiated."""
        lava_py_type = LavaPyType(cls=None, d_type=None)

        # Check if correct instance is created.
        self.assertIsInstance(lava_py_type, LavaPyType)

    def test_validate_precision_input_warning(self):
        """Check if Warning is raised when precision is validated but value is
        None."""

        with self.assertWarns(Warning):
            LavaPyType._validate_precision(precision=None)

    def test_validate_precision_wrong_input(self):
        """Check if TypeError is raised when precision is neither None nor
        Precision."""

        with self.assertRaises(TypeError):
            LavaPyType._validate_precision(precision=24)

    def test_validate_precision_wrong_first_literal(self):
        """Check if ValueError is raised when precision has wrong parameter for
        is_signed."""

        # Incorrect setting for first literal.
        with self.assertRaises(ValueError):
            LavaPyType._validate_precision(precision=Precision(
                is_signed=1,
                num_bits=24,
                implicit_shift=1))

    def test_validate_precision_wrong_second_literal(self):
        """Check if ValueError is raised when precision has wrong parameter for
        num_bits."""

        # Incorrect setting for second literal.
        with self.assertRaises(ValueError):
            LavaPyType._validate_precision(precision=Precision(
                is_signed=False,
                num_bits=24.1,
                implicit_shift=10))

    def test_validate_precision_wrong_third_literal(self):
        """Check if ValueError is raised when precision has wrong parameter for
        implicit_shift."""

        # Incorrect setting for third literal.
        with self.assertRaises(ValueError):
            LavaPyType._validate_precision(precision=Precision(
                is_signed=False,
                num_bits=24,
                implicit_shift=10.1))

    def test_validate_exp_data_no_exp_var(self):
        """Check if ValueError is raised when num_bits_exp is passed but no
        exp_var."""

        # Instantiate LavaPyType with 'num_bits_exp' but no 'exp_var'.
        lava_py_type = LavaPyType(cls=None, d_type=None, num_bits_exp=8)

        with self.assertRaises(ValueError):
            lava_py_type._validate_exp_data()

    def test_validate_exp_data_no_exp_var(self):
        """Check if ValueError is raised when exp_var is passed but no
        num_bits_exp."""

        # Instantiate LavaPyType with 'exp_var' but no 'num_bits_exp'.
        lava_py_type = LavaPyType(cls=None, d_type=None, exp_var='var')

        with self.assertRaises(ValueError):
            lava_py_type._validate_exp_data()

    def test_conversion_data(self):
        """Check conversion_data function of LavaPyType returns correct
        dictionary."""

        # Instantiate LavaPyType with information needed for float- to
        # fixed-point conversion.
        lava_py_type = LavaPyType(cls=None, d_type=np.ndarray,
                                  precision=Precision(is_signed=False,
                                                      num_bits=24,
                                                      implicit_shift=0),
                                  domain=np.array([0, 1]), constant=True,
                                  num_bits_exp=8, exp_var="var",
                                  scale_domain=1, meta_parameter=False)

        true_conv_data = {"is_signed": False,
                          "num_bits": 24,
                          "implicit_shift": 0,
                          "scale_domain" : 1,
                          "domain": np.array([0, 1]),
                          "constant" : True,
                          "num_bits_exp": 8,
                          "exp_var": "var"}

        conv_data = lava_py_type.conversion_data()

        # Check if dicts of precision are equal.
        # Test array with entries separately and remove it from dicts.
        np.testing.assert_array_equal(conv_data.pop('domain'),
                                      true_conv_data.pop('domain'))

        self.assertDictEqual(conv_data, true_conv_data)


if __name__ == "__main__":
    unittest.main()
