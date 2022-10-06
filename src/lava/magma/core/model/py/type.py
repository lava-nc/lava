# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from dataclasses import dataclass
from lava.magma.core.model.py.ports import PyInPort, PyOutPort, PyRefPort
import numpy as np
import warnings

# Members of LavaPyType needed for float- to fixed-point conversion.
CONST_CONV_VARS = ('num_bits_exp', 'domain', 'constant', 'scale_domain',
                   'exp_var', 'signedness', 'precision_bits', 'implicit_shift')


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
    # Indicate to which scale domain Var belongs, i.e. with which
    # Vars scaling needs to be consistent. Scale domains are identified by
    # integers: '0' indicates the global scale domain shared between all
    # Processes. Other integers are reserved for scale domains local to one
    # Process.
    # By default all Vars are assumend to be in same scale domain.
    scale_domain: int = 0

    @staticmethod
    def _validate_precision(precision):
        """Validates the format of the precision passed to the LavaPyType.

        Parameters
        ----------
        precision : string
            Specific precision in format '{u, s}:x:y', x, y ints

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
                          + " the Float2FixedPoint conversion")
        elif type(precision) is not str:
            raise TypeError("'precision' must be of type str or None")
        else:
            split_precision = precision.split(sep=":")

            # Check if signed or unsigned was correctly assigned".
            if not split_precision[0] in ['u', 's']:
                raise ValueError("First string literal of precision string"
                                 + " must be 'u' (unsigned) or 's' (signed)")

            # Check if number of bits and implicit shift was set correctly.
            try:
                int(split_precision[1])
            except ValueError:
                raise ValueError("Second string literal of precision (number"
                                 + " of  bits) must be interpretable as int")
            try:
                int(split_precision[2])
            except ValueError:
                raise ValueError("Thirds string literal of precision (implicit"
                                 + " shift) must be interpretable as int")

    def _validate_exp_data(self):
        """Validates if data regarding in a split of the variable in mantissa
        and exponent is complete.

        Raises
        ------
        ValueError
            If exponent data is not complete
        """

        if self.num_bits_exp and not self.exp_var:
            raise ValueError("Provided number of bits for exponent but no"
                             + " name for exponent variable.")
        if self.exp_var and not self.num_bits_exp:
            raise ValueError("Provided name for exponent variable but not"
                             + " number of bits for exponent..")

    def conversion_data(self):
        """Get data for variables needed for float- to fixed-point conversion
        defined in CONST_CONV_VARS.

        Returns
        -------
        conv_data : dict
            Conversion data dictionary, keys defined in conv_vars
        """
        LavaPyType._validate_precision(self.precision)
        self._validate_exp_data()

        conv_data = {}

        for key in CONST_CONV_VARS:
            if key == 'signedness':
                conv_data[key] = self.precision.split(":")[0]
            elif key == 'precision_bits':
                conv_data[key] = int(self.precision.split(":")[1])
            elif key == 'implicit_shift':
                conv_data[key] = int(self.precision.split(":")[2])
            else:
                conv_data[key] = self.__getattribute__(key)

        return conv_data
