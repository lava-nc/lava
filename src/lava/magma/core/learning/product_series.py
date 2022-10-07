# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty

import lava.magma.core.learning.string_symbols as str_symbols
from lava.magma.core.learning.symbolic_equation import (
    SymbolicEquation,
    Symbol,
    SymbolList,
    Operator,
    Addition,
    Subtraction,
    Multiplication,
    Dependency,
    Uk,
    Variable,
    Expression,
    BracketExpression,
    SgnExpression,
    Literal,
)


class Factor:
    """Factor representation of a single factor found in a Product.

    A Factor is a custom data structure holding information on:
    (1) State variable used by this Factor.
    (2) An optional constant added to the state variable.
    (3) Flag specifying if this Factor is a sgn() factor.

    Parameters
    ----------
    state_var: str
        State variable involved in this Factor.
    const: int, optional
        Constant involved in this Factor.
    is_sgn: bool
        Flag specifying if this Factor involves the sgn() function.
    """

    def __init__(
        self,
        state_var: str,
        const: ty.Optional[int] = None,
        is_sgn: bool = False,
    ) -> None:
        self._state_var = self._validate_state_var(state_var)
        self._const = const
        self._is_sgn = self._validate_sgn(is_sgn)

        if self._state_var in ["w", "t"]:
            const_str = "2C"
        else:
            const_str = "C"

        if self._state_var == "C":
            self._factor_type = self._state_var
        elif self._is_sgn:
            self._factor_type = f"sgn({self._state_var}+{const_str})"
        else:
            self._factor_type = f"{self._state_var}+{const_str}"

    @property
    def state_var(self) -> str:
        """Get the state variable involved in this Factor.

        Returns
        ----------
        state_var : str
            State variable involved in this Factor.
        """
        return self._state_var

    @property
    def const(self) -> ty.Optional[int]:
        """Get the constant involved in this Factor.

        Returns
        ----------
        const : int, optional
            Constant involved in this Factor.
        """
        return self._const

    @property
    def is_sgn(self) -> bool:
        """Get the is_sgn flag involved in this Factor, specifying if
        this Factor is an sgn Factor.

        Returns
        ----------
        is_sgn : bool
            Flag specifying if this Factor is an sgn Factor.
        """
        return self._is_sgn

    @property
    def factor_type(self) -> str:
        """Get factor type string of this Factor.

        Returns
        ----------
        factor_type : str
            Factor type string.
        """
        return self._factor_type

    @staticmethod
    def _validate_state_var(state_var: str) -> str:
        """Validate that a state variable is allowed.

        Parameters
        ----------
        state_var : str
            State variable.

        Returns
        ----------
        state_var : str
            State variable.
        """
        if state_var not in str_symbols.FACTOR_STATE_VARS:
            raise ValueError("Unknown state_var in Factor.")

        return state_var

    def _validate_sgn(self, is_sgn: bool) -> bool:
        """Validate that the state variable is allowed for an sgn Factor,
        if is_sgn True.

        Parameters
        ----------
        is_sgn : bool
            Flag specifying if this Factor is an sgn Factor.

        Returns
        ----------
        is_sgn : bool
            Flag specifying if this Factor is an sgn Factor.
        """
        if is_sgn and self._state_var not in str_symbols.SYNAPTIC_VARIABLES:
            raise ValueError(
                f"Cannot have sgn factor with state_var not in "
                f"{str_symbols.SYNAPTIC_VARIABLES}."
            )

        return is_sgn

    def has_const(self) -> bool:
        """Check if this Factor has a constant.

        Returns
        ----------
        has_constant : bool
            Flag specifying if this Factor has a constant or not.
        """
        return self._const is not None

    def __str__(self) -> str:
        return f"({self._factor_type}, C = {self._const})"


class Product:
    """Product representation of a single product found in a ProductSeries.

    A Product is a custom data structure holding information on:
    (1) Synaptic variable affected by the learning rule (target).
    (2) Dependency of the Product.
    (3) Mantissa of the scaling factor associated with the Product.
    (4) Exponent of the scaling factor associated with the Product.
    (5) List of Factors.
    (6) Decimate exponent used if the Dependency is uk.

    Parameters
    ----------
    target: str
        Left-hand side of learning rule equation in which the product appears.
        Either one of (dw, dd, dt).
    dependency: str
        Dependency used for this Product.
    s_mantissa: int
        Mantissa of the scaling constant for this Product.
    s_exp: int
        Exponent of the scaling constant for this Product.
    factors: list
        List of Factor objects for this Product.
    decimate_exponent: int, optional
        Decimate exponent used, if dependency is uk.
    """

    def __init__(
        self,
        target: str,
        dependency: str,
        s_mantissa: int,
        s_exp: int,
        factors: ty.List[Factor],
        decimate_exponent: ty.Optional[int] = None,
    ) -> None:
        self._target = self._validate_target(target)
        self._dependency = self._validate_dependency(dependency)
        self._s_mantissa = s_mantissa
        self._s_exp = s_exp
        self._factors = factors
        self._decimate_exponent = self._validate_decimate_exponent(
            decimate_exponent
        )

    @property
    def target(self) -> str:
        """Get the target of this Product.

        Returns
        ----------
        target : str
            Target of this Product.
        """
        return self._target

    @property
    def dependency(self) -> str:
        """Get the dependency of this Product.

        Returns
        ----------
        dependency : str
            Dependency of this Product.
        """
        return self._dependency

    @property
    def decimate_exponent(self) -> ty.Optional[int]:
        """Get the decimate exponent of this Product.

        Will be None if the dependency is not "u".

        Returns
        ----------
        decimate_exponent : int, optional
            Decimate exponent of this Product.
        """
        return self._decimate_exponent

    @property
    def s_mantissa(self) -> int:
        """Get the mantissa of the scaling factor of this Product.

        Returns
        ----------
        s_mantissa : str
            Mantissa of the scaling factor of this Product.
        """
        return self._s_mantissa

    @property
    def s_exp(self) -> int:
        """Get the exponent of the scaling factor of this Product.

        Returns
        ----------
        s_exp : str
            Exponent of the scaling factor of this Product.
        """
        return self._s_exp

    @property
    def factors(self) -> ty.List[Factor]:
        """Get the list of Factors involved in this Product.

        Returns
        ----------
        factors : list
            List of Factors involved in this Product.
        """
        return self._factors

    @staticmethod
    def _validate_target(target: str) -> str:
        """Validate that the target of this Product is allowed.

        Parameters
        ----------
        target : str
            Target of this Product.

        Returns
        ----------
        target : str
            Target of this Product.
        """
        if target not in str_symbols.LEARNING_RULE_TARGETS:
            raise ValueError("Unknown target in Product.")

        return target

    @staticmethod
    def _validate_dependency(dependency: str) -> str:
        """Validate that the dependency of this Product is allowed.

        Parameters
        ----------
        dependency : str
            Dependency of this Product.

        Returns
        ----------
        dependency : str
            Dependency of this Product.
        """
        if dependency not in str_symbols.DEPENDENCIES:
            raise ValueError("Unknown dependency in Product.")

        return dependency

    def _validate_decimate_exponent(
        self, decimate_exponent: ty.Optional[int]
    ) -> ty.Optional[int]:
        """Validate that the decimate exponent is None when
        the dependency is not "u" and that it is not None when
        the dependency is "u".

        Parameters
        ----------
        decimate_exponent : int, optional
            Decimate exponent of this Product.

        Returns
        ----------
        decimate_exponent : int, optional
            Decimate exponent of this Product.
        """
        if decimate_exponent is not None and self._dependency != str_symbols.U:
            raise ValueError(
                f"Can't have a non-null decimate exponent with a "
                f"dependency different from {str_symbols.U} in "
                f"Product. Found dependency is {self._dependency}"
            )

        if decimate_exponent is None and self._dependency == str_symbols.U:
            raise ValueError(
                f"Can't have a {str_symbols.U} dependency with a "
                f"null decimate exponent in Product."
            )

        return decimate_exponent

    def __str__(self) -> str:
        res = (
            f"{4 * ' '}Product: dependency={self._dependency}, "
            f"scaling={self._s_mantissa}*2^({self._s_exp}-7), "
            f"target={self._target}, "
            f"num_factors={len(self._factors)}\n"
        )

        res = res + f"{8 * ' '}Factors:\n"

        for factor in self._factors:
            res = res + f"{12 * ' '}{factor.__str__()}\n"

        return res


class ProductSeries:
    """ProductSeries representation of a single learning rule.

    A ProductSeries is a custom data structure holding information on:
    (1) Synaptic variable affected by the learning rule (target).
    (2) Decimate exponent used in uk dependencies, if any.
    (3) List of Products.
    (4) Dict with dependencies as keys and the set of all traces appearing
    with them in this ProductSeries.

    Parameters
    ----------
    target: str
        Left-hand side of learning rule equation. Either one of (dw, dd, dt).
    decimate_exponent: int, optional
        Decimate exponent used in uk dependencies, if any.
    products: list
        List of Products.

    Attributes
    ----------
    active_traces_per_dependency: dict
        Dict mapping active traces to the set of dependencies they appear with.
    """

    def __init__(self, symbolic_equation: SymbolicEquation) -> None:
        self._target = symbolic_equation.target
        self._decimate_exponent = None

        self._products = self._generate_product_list_from_symbol_list(
            symbolic_equation.symbol_list
        )

        self._active_traces_per_dependency = (
            self._get_active_traces_per_dependency_from_products()
        )

    @property
    def target(self) -> str:
        """Get the target of this ProductSeries.

        Returns
        ----------
        target : str
            Target of this ProductSeries.
        """
        return self._target

    @property
    def decimate_exponent(self) -> ty.Optional[int]:
        """Get the decimate exponent of this ProductSeries.

        Returns
        ----------
        decimate_exponent : int, optional
            Decimate exponent of this ProductSeries.
        """
        return self._decimate_exponent

    @property
    def products(self) -> ty.List[Product]:
        """Get the list of Products involved in this ProductSeries.

        Returns
        ----------
        products : list
            List of Products involved in this ProductSeries.
        """
        return self._products

    @property
    def active_traces_per_dependency(self) -> ty.Dict[str, ty.Set[str]]:
        """Get the dict of active traces per dependency associated with
        this ProductSeries.

        Returns
        ----------
        active_traces_per_dependency : dict
            Set of active traces per dependency in the list of ProductSeries.
        """
        return self._active_traces_per_dependency

    def _get_active_traces_per_dependency_from_products(
        self,
    ) -> ty.Dict[str, ty.Set[str]]:
        """Find set of traces associated with all dependencies in a
        ProductSeries.

        Example:

        For
        product_list = ProductSeries(y0 * x1 * y2 + x0 * y1 * y2)

        active_traces_per_dependency = {
            "y0": {"x1", "y2"},
            "x0": {"y1", "y2"}
        }

        Returns
        ----------
        active_traces_per_dependency : dict
            Set of active traces per dependency in the list of ProductSeries.
        """
        active_traces_per_dependency = {}

        for product in self._products:
            if product.dependency not in active_traces_per_dependency.keys():
                active_traces_per_dependency[product.dependency] = set()

            for factor in product.factors:
                if factor.state_var in str_symbols.TRACES:
                    active_traces_per_dependency[product.dependency].add(
                        factor.state_var
                    )

        return active_traces_per_dependency

    def _generate_product_list_from_symbol_list(
        self, symbol_list: SymbolList
    ) -> ty.List[Product]:
        """Generate a list of Products representation from a SymbolList
        representation.

        Parameters
        ----------
        symbol_list : SymbolList
            Learning rule in SymbolList representation.

        Returns
        ----------
        product_list : list
            List of Product objects.
        """
        separated_products = self._separate_products(symbol_list)

        product_list = []
        for symbols in separated_products:
            product_list.append(self._generate_product(symbols))

        return product_list

    @staticmethod
    def _separate_products(symbol_list: SymbolList) -> ty.List[ty.List[Symbol]]:
        """Separate a SymbolList representation into a list of lists of Symbols
        as an intermediary representation of a learning rule between SymbolList
        and ProductSeries.

        Separation is done by Addition or Subtraction operators into sub lists
        for individual products. Multiplication operators are excluded since
        they do not convey further information.

        Parameters
        ----------
        symbol_list : SymbolList
            Learning rule in SymbolList representation.

        Returns
        ----------
        intermediary_products : list
            Learning rule in intermediary product-series representation.
        """
        product_list = []
        current_product = []

        for symbol in symbol_list.list:
            if isinstance(symbol, Multiplication):
                continue

            if (
                isinstance(symbol, Addition) or isinstance(symbol, Subtraction)
            ) and len(current_product) > 0:
                product_list.append(current_product)

                current_product = [symbol]
            else:
                current_product.append(symbol)

        product_list.append(current_product)

        return product_list

    def _generate_product(self, symbols: ty.List[Symbol]) -> Product:
        """Generate a Product object from a list of Symbol objects

        Parameters
        ----------
        symbols : list
            Intermediary representation of a Product as a list of Symbols.

        Returns
        ----------
        product : Product
            Product object representing a product.
        """
        dependency = None
        decimate_exponent = None
        s_mantissa = 1
        s_exp = 0
        factors = []

        combine_literals = True
        scaling_factor_is_set = False

        for symbol in symbols:
            # Addition
            if isinstance(symbol, Addition):
                continue
            # Subtraction
            if isinstance(symbol, Subtraction):
                s_mantissa = s_mantissa * -1
            # Dependency
            elif isinstance(symbol, Dependency):
                if isinstance(symbol, Uk):
                    decimate_exponent = symbol.decimate_exponent
                    if (
                        self._decimate_exponent is not None
                        and self._decimate_exponent != decimate_exponent
                    ):
                        raise ValueError(
                            "Can't have multiple decimate "
                            "exponents in the same ProductSeries."
                        )
                    self._decimate_exponent = decimate_exponent

                    if dependency is None:
                        dependency = str_symbols.U
                else:
                    if dependency is None:
                        dependency = symbol.expr
                    else:
                        factors.append(Factor(symbol.expr))

            # Literal
            elif isinstance(symbol, Literal):
                if not scaling_factor_is_set or combine_literals:
                    s_mantissa = s_mantissa * symbol.mantissa
                    s_exp = s_exp + symbol.exponent
                    scaling_factor_is_set = True
                else:
                    const = symbol.mantissa * symbol.base**symbol.exponent
                    factors.append(Factor("C", const))
            # Variable
            elif isinstance(symbol, Variable):
                factors.append(Factor(symbol.expr))
            # Expression
            elif isinstance(symbol, Expression):
                sub_symbol_list = symbol.symbol_list.list

                if not isinstance(sub_symbol_list[0], Operator):
                    sub_symbol_list.insert(0, Addition())

                var_sgn_idx = None
                var = const = None

                if len(sub_symbol_list) <= 2:
                    operator = sub_symbol_list[0]
                    sub_symbol = sub_symbol_list[1]
                    sign = 1

                    if isinstance(operator, Subtraction):
                        sign = -1

                    if isinstance(sub_symbol, Variable):
                        var = sub_symbol.expr
                    elif isinstance(sub_symbol, Literal):
                        if not scaling_factor_is_set or combine_literals:
                            s_mantissa = s_mantissa * sub_symbol.mantissa
                            s_mantissa = s_mantissa * sign
                            s_exp = s_exp + sub_symbol.exponent
                            scaling_factor_is_set = True
                        else:
                            var = "C"
                            const = (
                                sign
                                * sub_symbol.mantissa
                                * sub_symbol.base**sub_symbol.exponent
                            )

                else:
                    if (
                        isinstance(sub_symbol_list[1], Dependency)
                        or isinstance(sub_symbol_list[1], Variable)
                    ) and isinstance(sub_symbol_list[3], Literal):
                        var_sgn_idx = 0
                        var_idx = 1
                        const_sgn_idx = 2
                        const_idx = 3
                    elif (
                        isinstance(sub_symbol_list[3], Dependency)
                        or isinstance(sub_symbol_list[3], Variable)
                    ) and isinstance(sub_symbol_list[1], Literal):
                        var_sgn_idx = 2
                        var_idx = 3
                        const_sgn_idx = 0
                        const_idx = 1
                    else:
                        raise ValueError(
                            f"Sub-expression {symbol.sub_expr} cannot "
                            f"be interpreted to be of type [+,-]var[+,-]const"
                        )

                    sign_var = (
                        -1
                        if isinstance(sub_symbol_list[var_sgn_idx], Subtraction)
                        else 1
                    )
                    var = sub_symbol_list[var_idx].expr
                    sign_const = (
                        -1
                        if isinstance(
                            sub_symbol_list[const_sgn_idx], Subtraction
                        )
                        else 1
                    )
                    const = (
                        sub_symbol_list[const_idx].mantissa
                        * sub_symbol_list[const_idx].base
                        ** sub_symbol_list[const_idx].exponent
                    )
                    const = const * sign_const

                    if sign_var == -1:
                        s_mantissa = s_mantissa * sign_var
                        const = const * sign_var

                if isinstance(symbol, BracketExpression):
                    if var is not None:
                        factors.append(Factor(var, const))

                elif isinstance(symbol, SgnExpression):
                    if var not in ["w", "d", "t"]:
                        raise ValueError(
                            "sgn(..) function can only be applied to "
                            "'w', 'd' or 't' variables."
                        )
                    if var_sgn_idx is not None and isinstance(
                        sub_symbol_list[var_sgn_idx], Subtraction
                    ):
                        raise ValueError(
                            "Variable in sgn(..) function "
                            "cannot be negative."
                        )

                    factors.append(Factor(var, const, is_sgn=True))

            else:
                raise ValueError(
                    f"Sub-expression {symbol} cannot be interpreted."
                )

        while abs(s_mantissa) > 7 and s_mantissa % 2 == 0 and s_exp < 8:
            s_mantissa = int(s_mantissa / 2)
            s_exp = s_exp + 1

        if dependency is None:
            raise ValueError("No dependency found.")

        if s_mantissa < -8 or s_mantissa >= 8:
            raise ValueError(
                f"Mantissa magnitude too large or not representable as "
                "'s_mantissa = x * 2**y'."
            )

        if s_exp < -7 or s_exp >= 9:
            raise ValueError(
                "Exponent of scaling constant exponent " "must within -7 and 9."
            )

        product = Product(
            self._target,
            dependency,
            s_mantissa,
            s_exp + 7,
            factors,
            decimate_exponent,
        )

        return product

    def __str__(self) -> str:
        res = f"ProductSeries: decimate_exponent={self._decimate_exponent}\n"
        for product in self._products:
            res = res + product.__str__()

        return res
