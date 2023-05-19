# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import re
from abc import ABC, abstractmethod
import typing as ty
import ast

import lava.magma.core.learning.string_symbols as str_symbols


class Symbol(ABC):
    """Super class for all possible symbols."""

    def __init__(self, expr: ty.Optional[str] = "") -> None:
        """Initialize a Symbol.

        Parameters
        ----------
        expr : str
            String expression.
        """
        self._expr = expr

    @property
    def expr(self) -> str:
        """Get expression of the Symbol.

        Returns
        ----------
        expr : str
            String expression.
        """
        return self._expr

    @expr.setter
    def expr(self, expr: str) -> None:
        """Set expression of the Symbol.

        Parameters
        ----------
        expr : str
            String expression.
        """
        self._expr = expr

    @abstractmethod
    def __str__(self):
        pass

    @staticmethod
    def find_expr(expr: str,
                  reg_expr: str,
                  symbol: "Symbol") -> ty.Tuple[ty.Optional["Symbol"], str]:
        """Factory method for creating symbols.

        Matches an expression to a regular expression and if there is a match,
        return a symbol of the matching part of the expression
        as well as the rest of the expression.

        Parameters
        ----------
        expr : str
            String expression.
        reg_expr: str
            Regular expression.
        symbol: Symbol
            Uninitialized symbol

        Returns
        ----------
        symbol : Symbol, optional
            Symbol matching regular expression.
        expr : str
            Remaining string expression after extraction of the symbol.
        """
        p = re.compile(reg_expr)
        m = p.match(expr)

        if m is None:
            return None, expr

        match_len = m.span()[1]
        symbol.expr = m.group()

        return symbol, expr[match_len:]


class SymbolList(Symbol):
    """Represents as list of Symbols."""

    def __init__(self):
        super(SymbolList, self).__init__('')

        self._list = []

    @property
    def list(self) -> ty.List[Symbol]:
        """Get list of the SymbolList.

        Returns
        ----------
        list : list
            List of Symbol objects.
        """
        return self._list

    def append(self, symbol: Symbol) -> None:
        """Append a Symbol to the SymbolList's list and the Symbol's
        expression to the SymbolList's expression.

        Parameters
        ----------
        symbol : Symbol
            Symbol object.
        """
        self._list.append(symbol)
        self._expr += symbol.expr

    def __str__(self) -> str:
        res = "["

        for idx, symbol in enumerate(self.list):
            res = res + symbol.__str__()
            if idx != len(self.list) - 1:
                res = res + ", "

        res = res + "]"

        return res


class Operator(Symbol):
    """Abstract super class for operator Symbols."""


class Addition(Operator):
    """Symbol representing the addition operator."""

    @staticmethod
    def find(expr: str) -> ty.Tuple[ty.Optional["Addition"], str]:
        r"""Factory method for creating Addition symbols.

        Matches an expression to the regular expressions
        "\+".

        Return a Addition Symbol if there is a match and the rest of the
        expression.

        Parameters
        ----------
        expr : str
            String expression.

        Returns
        ----------
        symbol : Addition, optional
            Symbol matching regular expression.
        expr : str
            Remaining string expression after extraction of the symbol.
        """
        return Symbol.find_expr(expr, r"\+", Addition())

    def __str__(self):
        return "Addition(" + self._expr + ")"


class Subtraction(Operator):
    """Symbol representing the subtraction operator."""

    @staticmethod
    def find(expr: str) -> ty.Tuple[ty.Optional["Subtraction"], str]:
        r"""Factory method for creating Subtraction symbols.

        Matches an expression to the regular expressions
        "\-".

        Return a Subtraction Symbol if there is a match and the rest of the
        expression.

        Parameters
        ----------
        expr : str
            String expression.

        Returns
        ----------
        symbol : Subtraction, optional
            Symbol matching regular expression.
        expr : str
            Remaining string expression after extraction of the symbol.
        """
        return Symbol.find_expr(expr, r'\-', Subtraction())

    def __str__(self):
        return "Subtraction(" + self._expr + ")"


class Multiplication(Operator):
    """Symbol representing the multiplication operator."""

    @staticmethod
    def find(expr: str) -> ty.Tuple[ty.Optional["Multiplication"], str]:
        r"""Factory method for creating Multiplication symbols.

        Matches an expression to the regular expressions
        "\*".

        Return a Multiplication Symbol if there is a match and the rest of the
        expression.

        Parameters
        ----------
        expr : str
            String expression.

        Returns
        ----------
        symbol : Multiplication, optional
            Symbol matching regular expression.
        expr : str
            Remaining string expression after extraction of the symbol.
        """
        return Symbol.find_expr(expr, r'\*', Multiplication())

    def __str__(self):
        return "Multiplication(" + self._expr + ")"


class FactorSym(Symbol):
    """Abstract super class for factor Symbols."""


class Dependency(FactorSym):
    """Abstract super class for dependency Symbols."""


class X0(Dependency):
    """Symbol representing the x0 dependency."""

    @staticmethod
    def find(expr: str) -> ty.Tuple[ty.Optional["X0"], str]:
        """Factory method for creating X0 symbols.

        Matches an expression to the regular expressions
        "x0".

        Return a X0 Symbol if there is a match and the rest of the
        expression.

        Parameters
        ----------
        expr : str
            String expression.

        Returns
        ----------
        symbol : X0, optional
            Symbol matching regular expression.
        expr : str
            Remaining string expression after extraction of the symbol.
        """
        return Symbol.find_expr(expr, str_symbols.X0, X0())

    def __str__(self):
        return "X0(" + self._expr + ")"


class Y0(Dependency):
    """Symbol representing the y0 dependency."""

    @staticmethod
    def find(expr: str) -> ty.Tuple[ty.Optional["Y0"], str]:
        """Factory method for creating Y0 symbols.

        Matches an expression to the regular expressions
        "y0".

        Return a Y0 Symbol if there is a match and the rest of the
        expression.

        Parameters
        ----------
        expr : str
            String expression.

        Returns
        ----------
        symbol : Y0, optional
            Symbol matching regular expression.
        expr : str
            Remaining string expression after extraction of the symbol.
        """
        return Symbol.find_expr(expr, str_symbols.Y0, Y0())

    def __str__(self):
        return "Y0(" + self._expr + ")"


class Uk(Dependency):
    """Symbol representing the uk dependency."""

    def __init__(self):
        super().__init__()
        self._decimate_exponent = None

    @property
    def decimate_exponent(self) -> ty.Optional[int]:
        """Get decimate exponent of this Uk.

        Returns
        ----------
        decimate_exponent : int
            Decimate exponent.
        """
        return self._decimate_exponent

    @decimate_exponent.setter
    def decimate_exponent(self, decimate_exponent: int) -> None:
        """Set decimate exponent of this Uk.

        Parameters
        ----------
        decimate_exponent : int
            Decimate exponent.
        """
        self._decimate_exponent = decimate_exponent

    @staticmethod
    def find(expr: str) -> ty.Tuple[ty.Optional["Uk"], str]:
        r"""Factory method for creating Uk symbols.

        Matches an expression to the regular expressions
        "u\d".

        Return a Uk Symbol if there is a match and the rest of the
        expression.

        Parameters
        ----------
        expr : str
            String expression.

        Returns
        ----------
        symbol : Uk, optional
            Symbol matching regular expression.
        expr : str
            Remaining string expression after extraction of the symbol.
        """
        symbol, expr = Symbol.find_expr(expr, r'u\d', Uk())

        if symbol is not None:
            symbol.decimate_exponent = int(symbol.expr[1:])

        return symbol, expr

    def __str__(self):
        return "Uk(" + self._expr + ")"


class Variable(FactorSym):
    """Symbol representing traces and synaptic variable factors."""

    @staticmethod
    def find(expr: str) -> ty.Tuple[ty.Optional["Variable"], str]:
        """Factory method for creating Variable symbols.

        Matches an expression to the regular expressions
        "x[12]", "y[123]", "w", "d", "t".

        Return a Variable Symbol if there is a match and the rest of the
        expression.

        Parameters
        ----------
        expr : str
            String expression.

        Returns
        ----------
        symbol : Variable, optional
            Symbol matching regular expression.
        expr : str
            Remaining string expression after extraction of the symbol.
        """
        v = Variable()

        symbol, expr = Symbol.find_expr(expr, 'x[12]', v)
        if not isinstance(symbol, Variable):
            symbol, expr = Symbol.find_expr(expr, 'y[123]', v)
        if not isinstance(symbol, Variable):
            symbol, expr = Symbol.find_expr(expr, str_symbols.W, v)
        if not isinstance(symbol, Variable):
            symbol, expr = Symbol.find_expr(expr, str_symbols.D, v)
        if not isinstance(symbol, Variable):
            symbol, expr = Symbol.find_expr(expr, str_symbols.T, v)

        return symbol, expr

    def __str__(self):
        return "Variable(" + self._expr + ")"


class Expression(FactorSym):
    """Abstract super class for multi-symbol Symbols."""

    def __init__(self):
        super().__init__()
        self._sub_expr = None
        self._symbol_list = None

    @property
    def sub_expr(self) -> ty.Optional[str]:
        """Get sub-expression of this Expression.

        Returns
        ----------
        sub_expr : str
            String sub-expression.
        """
        return self._sub_expr

    @sub_expr.setter
    def sub_expr(self, sub_expr: str) -> None:
        """Set sub-expression of this Expression.

        Parameters
        ----------
        sub_expr : str
            String sub-expression.
        """
        self._sub_expr = sub_expr

    @property
    def symbol_list(self) -> ty.Optional[SymbolList]:
        """Get SymbolList associated with sub-expression of this Expression.

        Returns
        ----------
        symbol_list : SymbolList
            SymbolList.
        """
        return self._symbol_list

    @symbol_list.setter
    def symbol_list(self, symbol_list: SymbolList) -> None:
        """Set SymbolList associated with sub-expression of this Expression.

        Parameters
        ----------
        symbol_list : SymbolList
            SymbolList.
        """
        self._symbol_list = symbol_list

    @staticmethod
    def _find_closing_bracket_idx(expr: str) -> ty.Optional[int]:
        """Find closing bracket associated with the first found opening bracket
        and return its index.

        Return None if closing index was found matching the first found
        opening bracket.

        Parameters
        ----------
        expr : str
            String expression.

        Returns
        ----------
        closing_bracket_idx : int
            Index of closing bracket.
        """
        idx = [(1 if i.group() == '(' else -1, i.start()) for i in
               re.finditer('[()]', expr)]
        sum_res = 0
        closing_idx = None

        for i in idx:
            sum_res += i[0]
            if sum_res == 0:
                closing_idx = i[1]
                break

        return closing_idx

    @abstractmethod
    def find_sub_expr(self, expr: str) -> str:
        pass


class BracketExpression(Expression):
    """Symbol representing a bracket expression of the form : (...)."""

    def find_sub_expr(self, expr: str) -> str:
        """Find sub-expression of an expression, assuming the expression is
        of the form "(...)", representing a BracketExpression.

        Parameters
        ----------
        expr : str
            String expression.

        Returns
        ----------
        sub_expr : str
            String sub-expression.
        """
        try:
            idx = self._find_closing_bracket_idx(expr)
            self._sub_expr = expr[1:idx]

            expr = expr[idx + 1:]

            return expr
        except TypeError:
            raise ValueError("Bracket expression missing closing bracket.")

    @staticmethod
    def find(expr: str) -> ty.Tuple[ty.Optional["BracketExpression"], str]:
        r"""Factory method for creating BracketExpression symbols.

        Matches an expression to the regular expression
        "\(". If there is a match, find the sub expression.

        Return a BracketExpression Symbol if there is a match and
        closing bracket and the rest of the expression.

        Parameters
        ----------
        expr : str
            String expression.

        Returns
        ----------
        symbol : BracketExpression, optional
            Symbol matching regular expression.
        expr : str
            Remaining string expression after extraction of the symbol.
        """
        init_expr = expr

        symbol, expr = Symbol.find_expr(expr, r'\(', BracketExpression())

        if isinstance(symbol, BracketExpression):
            expr = symbol.find_sub_expr('(' + expr)

            symbol.expr = init_expr

        return symbol, expr

    def __str__(self):
        return "BracketExpression(" + self._symbol_list.__str__() + ")"


class SgnExpression(Expression):
    """Symbol representing a sign expression of the form :  sgn(...)."""

    def find_sub_expr(self, expr: str) -> str:
        """Find sub-expression of an expression, assuming the expression is
        of the form "sgn(...)", representing a SgnExpression.

        Parameters
        ----------
        expr : str
            String expression.

        Returns
        ----------
        sub_expr : str
            String sub-expression.
        """
        try:
            idx = self._find_closing_bracket_idx(expr)
            self.sub_expr = expr[4:idx]

            expr = expr[idx + 1:]

            return expr
        except TypeError:
            raise ValueError("Bracket expression missing closing bracket.")

    @staticmethod
    def find(expr: str) -> ty.Tuple[ty.Optional["SgnExpression"], str]:
        r"""Factory method for creating SgnExpression symbols.

        Matches an expression to the regular expression
        "sgn\(". If there is a match, find the sub expression.

        Return a SgnExpression Symbol if there is a match and
        closing bracket and the rest of the expression.

        Parameters
        ----------
        expr : str
            String expression.

        Returns
        ----------
        symbol : SgnExpression, optional
            Symbol matching regular expression.
        expr : str
            Remaining string expression after extraction of the symbol.
        """
        init_expr = expr

        symbol, expr = Symbol.find_expr(expr, r'sgn\(', SgnExpression())

        if isinstance(symbol, SgnExpression):
            expr = symbol.find_sub_expr('sgn(' + expr)

            symbol.expr = init_expr

        return symbol, expr

    def __str__(self):
        return "SgnExpression(" + self._symbol_list.__str__() + ")"


class Literal(Symbol):
    """Symbol representing a literal."""

    def __init__(self):
        super().__init__()
        self._mantissa = 0
        self._base = 1
        self._exponent = 0
        self._literal_type = None

    @property
    def mantissa(self) -> int:
        """Get mantissa of this Literal.

        Returns
        ----------
        mantissa : int
            Mantissa of this literal.
        """
        return self._mantissa

    @property
    def base(self) -> int:
        """Get base of this Literal.

        Returns
        ----------
        base : int
            Base of this literal.
        """
        return self._base

    @property
    def exponent(self) -> int:
        """Get exponent of this Literal.

        Returns
        ----------
        exponent : int
            Exponent of this literal.
        """
        return self._exponent

    @property
    def literal_type(self) -> int:
        """Get literal type of this Literal.

        Returns
        ----------
        literal_type : int
            Literal type of this literal.
        """
        return self._literal_type

    @literal_type.setter
    def literal_type(self, literal_type: int) -> None:
        """Set literal type of this Literal.

        Parameters
        ----------
        literal_type : int
            Literal type of this literal.
        """
        self._literal_type = literal_type

    def to_int(self) -> None:
        """Extract mantissa, base and exponent of this Literal from string
        expression and store them."""
        if self._literal_type == 0:
            # Convert [+/-]x*2^[+/0]y
            val = self._expr.split('^')
            self._mantissa = ast.literal_eval(val[0][:-2])
            self._base = 2
            self._exponent = ast.literal_eval(val[1])
        elif self._literal_type == 1:
            # Convert [+/-]2^[+/0]y
            val = self._expr.split('^')
            self._mantissa = int(ast.literal_eval(val[0]) / 2)
            self._base = 2
            self._exponent = ast.literal_eval(val[1])
        elif self._literal_type == 2:
            # Convert [+/-]x
            self._mantissa = ast.literal_eval(self._expr)
            self._base = 1
            self._exponent = 0

    @property
    def val(self) -> int:
        """Get the integer value represented by this Literal.

        Returns
        ----------
        val : int
            Integer value of this Literal.
        """
        return int(self._mantissa * self._base ** self._exponent)

    @staticmethod
    def find(expr):
        r"""Factory method for creating Literal symbols.

        Matches an expression to the regular expressions
        "[\+\-]?\d+\*2\^[\+\-]?\d+", "[\+\-]?2\^[\+\-]?\d+", "[\+\-]?\d+".

        Return a Literal Symbol if there is a match and
        closing bracket and the rest of the expression.

        Parameters
        ----------
        expr : str
            String expression.

        Returns
        ----------
        symbol : Literal, optional
            Symbol matching regular expression.
        expr : str
            Remaining string expression after extraction of the symbol.
        """
        # Match x*2^y
        symbol, expr = Symbol.find_expr(expr, r'[\+\-]?\d+\*2\^[\+\-]?\d+',
                                        Literal())
        if isinstance(symbol, Literal):
            symbol.literal_type = 0
        else:
            # Match 2^y
            symbol, expr = Symbol.find_expr(expr, r'[\+\-]?2\^[\+\-]?\d+',
                                            Literal())
            if isinstance(symbol, Literal):
                symbol.literal_type = 1
            else:
                # Match x
                symbol, expr = Symbol.find_expr(expr, r'[\+\-]?\d+',
                                                Literal())
                if isinstance(symbol, Literal):
                    symbol.literal_type = 2

        if isinstance(symbol, Literal):
            symbol.to_int()

        return symbol, expr

    def __str__(self):
        return "Literal(" + self._expr + ")"


SYMBOL_CLASSES = [Addition, Subtraction, Multiplication, X0, Y0, Uk,
                  Variable, BracketExpression, SgnExpression, Literal]


class SymbolicEquation:
    """The SymbolicEquation represents a learning rule as a set of symbols.

    It provides means to generate a SymbolicEquation from a string following
    a fixed syntax.

    Parameters
    ----------
    target : str
        Target of the learning rule to be represented by this SymbolicEquation.
    str_learning_rule : str
        Learning rule in string format to be represented by
        this SymbolicEquation.
    """

    def __init__(self, target: str, str_learning_rule: str) -> None:
        self._target = target
        self._str_learning_rule = str_learning_rule

        self._symbol_list = self._generate_symbol_list_from_string()

    @property
    def target(self) -> str:
        """Get target of this SymbolicEquation.

        Returns
        ----------
        target : str
            Target of this SymbolicEquation.
        """
        return self._target

    @property
    def symbol_list(self) -> SymbolList:
        """Get SymbolList of this SymbolicEquation.

        Returns
        ----------
        symbol_list : SymbolList
            SymbolList of this SymbolicEquation.
        """
        return self._symbol_list

    def _find_next_symbol(self, expression: str) \
            -> ty.Tuple[ty.Optional[Symbol], str]:
        """Find next symbol in a string expression.

        If the new symbol is of type Expression, apply parse to
        its sub-expression.

        Parameters
        ----------
        expression : str
            String expression.

        Returns
        ----------
        new_symbol : Symbol
            Found new symbol.
        expression : str
            Remaining string expression after extraction of the new symbol.
        """
        new_symbol = None

        for symbol in SYMBOL_CLASSES:
            new_symbol, expression = symbol.find(expression)

            if isinstance(new_symbol, Expression):
                new_symbol.symbol_list = self._parse(new_symbol.sub_expr)

            if new_symbol is not None:
                break

        return new_symbol, expression

    def _parse(self, expression: str) -> SymbolList:
        """Parse a string expression to yield a SymbolList representation.

        Finds next symbols in the string expression one by one until
        the string expression is empty.

        Parameters
        ----------
        expression : str
            String expression.

        Returns
        ----------
        symbol_list : SymbolList
            SymbolList object resulting from the string parsing operation.
        """
        done = False
        symbol_list = SymbolList()

        while not done:
            new_symbol, expression = self._find_next_symbol(expression)
            symbol_list.append(new_symbol)
            done = not (len(expression) > 0)

        return symbol_list

    def _generate_symbol_list_from_string(self) -> SymbolList:
        """Generate a SymbolList representation from a string representation.

        Returns
        ----------
        symbol_list : SymbolList
            Learning rule in SymbolList representation.
        """
        learning_rule = re.sub(' ', '', self._str_learning_rule)

        symbol_list = self._parse(learning_rule)

        return symbol_list
