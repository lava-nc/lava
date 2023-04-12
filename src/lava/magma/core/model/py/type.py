# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
from dataclasses import dataclass
from lava.magma.core.model.precision import Precision
from lava.magma.core.model.py.ports import PyInPort, PyOutPort, PyRefPort
import numpy as np
import warnings


@dataclass
class LavaPyType:
    cls: ty.Union[
        type, ty.Type[PyInPort], ty.Type[PyOutPort], ty.Type[PyRefPort]
    ]
    d_type: type
    precision: str = None  # If None, infinite precision is assumed.
    num_bits_exp: int = None  # If None, fixed exponent in fixed-pt model.
    exp_var: str = None  # Name of Var in which exponent is stored, if needed.
    domain: np.ndarray = None  # If None, no a-priori knowledge of Var domain.
    constant: bool = False  # If True, indicates that Var is constant.
    meta_parameter: bool = False  # If True, indicates that Var is config var
    # Indicate to which scale domain Var belongs, i.e., with which
    # Vars scaling needs to be consistent. Scale domains are identified by
    # integers: '0' indicates the global scale domain shared between all
    # Processes. Other integers are reserved for scale domains local to one
    # Process.
    # By default all Vars are assumed to be in same scale domain.
    scale_domain: int = 0

    # Members of LavaPyType needed for float- to fixed-point conversion.
    conv_vars: tuple = ('num_bits_exp', 'domain', 'constant', 'scale_domain',
                        'exp_var', 'is_signed', 'num_bits', 'implicit_shift')

    @staticmethod
    def _validate_precision(precision):
        """Validates the format of the precision passed to the LavaPyType.

        Raises
        ------
        Warning
            If precision is None, might cause error in Float2Fixed Point
            Conversion
        TypeError
            If precision is not of type str or None
        ValueError
            If precision passed in wrong format
        """

        if precision is None:
            warnings.warn("'precision' is None: This might cause an error in"
                          + " the Float2FixedPoint conversion.")
        elif type(precision) is not Precision:
            raise TypeError("'precision' must be of type Precision or None"
                            + f" but has type {type(precision)}.")
        else:
            # Check if is_signed was correctly assigned".
            if not type(precision.is_signed) is bool:
                raise ValueError("'is_signed' has type"
                                 + f"{type(precision.is_signed)}, but must"
                                 + " be bool.")

            # Check if number of bits and implicit shift was set correctly.
            if not type(precision.num_bits) is int:
                raise ValueError("'num_bits' has type"
                                 + f"{type(precision.num_bits)}, but must"
                                 + " be int.")

            if not type(precision.implicit_shift) is int:
                raise ValueError("'num_bits' has type"
                                 + f"{type(precision.implicit_shift)}, but"
                                 + " must be int.")

    def _validate_exp_data(self):
        """Validates if data considered in a split of the variable in mantissa
        and exponent is complete.

        Raises
        ------
        ValueError
            If exponent data is not complete
        """

        if self.num_bits_exp and not self.exp_var:
            raise ValueError("Provided number of bits for exponent but no"
                             + " name for exponent variable.")
        if (self.exp_var and not self.num_bits_exp) and self.num_bits_exp != 0:
            # Evaluates to true if exp_var given and num_bits_exp None.
            # If num_bits_exp is greater or equalt than 0, evaluates to False.
            raise ValueError("Provided name for exponent variable but not"
                             + " number of bits for exponent..")

    def conversion_data(self):
        """Get data for variables needed for float- to fixed-point conversion
        defined in conv_vars.

        Returns
        -------
        conv_data : dict
            Conversion data dictionary, keys defined in conv_vars
        """
        LavaPyType._validate_precision(self.precision)
        self._validate_exp_data()

        conv_data = {}

        for key in self.conv_vars:
            if key in ['is_signed', 'num_bits', 'implicit_shift']:
                conv_data[key] = self.precision.__getattribute__(key)
            else:
                conv_data[key] = self.__getattribute__(key)

        return conv_data
