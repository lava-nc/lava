# Copyright (C) 2022-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty
from scipy.sparse import csr_matrix

from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess


class NetworkList(list):
    """NetworkList
    This is a list subclass to keep track of Network objects that
    are added using the '+' operator.
    """

    def __init__(self, iterable):
        super().__init__(iterable)


class Network:
    """Network
    Abstract Network object.

    Networks contain other networks and lava processes.
    """

    in_port: InPort
    out_port: OutPort
    main: AbstractProcess

    def run(self, **kwargs):
        self.main.run(**kwargs)

    def stop(self, **kwargs):
        self.main.stop(**kwargs)

    def __lshift__(self,
                   other):
        # Self-referential type hint is causing a NameError
        # other: ty.Union[Network, NetworkList]):
        """
        Operator overload of "<<" to connect Network objects.

        EPF: note that the precedence could matter if we include more
        operators. We want this assignment operator to have lowest
        precedence, which "<<" is lower than "+", so it works. However, it
        is higher than i.e. "^" which would not work. Comparisons have even
        lower precedence, "<=" could be better.
        """
        if isinstance(other, Network):
            other.out_port.connect(self.in_port)
            return self
        elif isinstance(other, NetworkList):
            for o in other:
                self << o
            return self
        else:
            return NotImplemented

    def __add__(self,
                other):
        # Self-referential typing is causing a NameError
        # other: ty.Union[Network, NetworkList]):
        """
        Operator overload of "+" to act as summation in algebraic syntax.
        """
        if isinstance(other, Network):
            return NetworkList([self, other])
        elif isinstance(other, NetworkList):
            other.append(self)
            return other
        else:
            return NotImplemented
    # When chaining operations this is used for [weights1, weights2] + weights3
    __radd__ = __add__


class AlgebraicVector(Network):
    """AlgebraicVector
    Provides vector operator syntax for Networks.
    """

    def __lshift__(self,
                   other):
        # Self-referential typing is causing a NameError
        # other: ty.Union[AlgebraicVector, Network, NetworkList]):
        """
        Operator overload of "<<" to connect AlgebraicVector objects.
        """

        if isinstance(other, AlgebraicVector):
            # If a vector is connected to another vector, an Identity
            # connection is generated and the two procs are connected.

            # This import statement needs to be here to avoid a circular
            # import error
            from lava.networks.gradedvecnetwork import GradedSparse
            weightsI = csr_matrix(np.eye(np.prod(self.shape)))
            I_syn = GradedSparse(weights=weightsI)
            other.out_port.connect(I_syn.in_port)
            I_syn.out_port.connect(self.in_port)
            return self

        elif isinstance(other, Network):
            # This will take care of the standard weights to neurons.
            other.out_port.connect(self.in_port)
            return self
        elif isinstance(other, NetworkList):
            # When using the plus operator to add
            for o in other:
                self << o
            return self
        else:
            return NotImplemented


class AlgebraicMatrix(Network):
    """AlgebraicMatrix
    Provides matrix operator syntax for Networks.
    """

    def __matmul__(self,
                   other):
        # Self-referential typing is causing a NameError
        # other: AlgebraicVector):
        """
        Operator overload of "@" to form matrix-vector product.
        """
        if isinstance(other, AlgebraicVector):
            other.out_port.connect(self.in_port)
            return self
        else:
            return NotImplemented

    def __mul__(self,
                other):
        # Self-referential typing is causing a NameError
        # other: AlgebraicMatrix):
        """
        Operator overload of "*" to for multiplication.
        """
        if isinstance(other, AlgebraicMatrix):
            from lava.networks.gradedvecnetwork import ProductVec
            # How to pass in exp?
            prod_layer = ProductVec(shape=self.shape, vth=1, exp=0)

            prod_layer << (self, other)

            return prod_layer
        else:
            return NotImplemented
