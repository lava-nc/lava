# Copyright (C) 2022-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess


class Network:
    """Network
    Abstract Network object.

    Networks contain other networks and lava processes
    """

    in_port: InPort
    out_port: OutPort
    main: AbstractProcess

    def run(self, **kwargs):
        self.main.run(**kwargs)

    def stop(self, **kwargs):
        self.main.stop(**kwargs)


class AlgebraicVector(Network):
    """AlgebraicVector
    Provides vector operator syntax for Networks.
    """

    def __lshift__(self, other):
        """
        Use lshift for connecting inputs.

        EPF: note that the precedence could matter if we include more
        operators. We want this assignment operator to have lowest
        precedence, which "<<" is lower than "+", so it works. However, it
        is higher than i.e. "^" which would not work. Comparisons have even
        lower precedence, "<=" could be better.
        """
        if isinstance(other, Network):
            print('connecting')
            other.out_port.connect(self.in_port)
            return self
        elif isinstance(other, (list, tuple)):
            for o in other:
                o.out_port.connect(self.in_port)
            return self
        else:
            return NotImplemented


class AlgebraicMatrix(Network):
    """AlgebraicMatrix
    Provides matrix operator syntax for Networks.
    """
    def __matmul__(self, other):
        if isinstance(other, AlgebraicVector):
            other.out_port.connect(self.in_port)
            return self
        else:
            return NotImplemented

    def __add__(self, other):
        if isinstance(other, AlgebraicMatrix):
            return [self, other]
        elif isinstance(other, list):
            other.append(self)
            return other
        else:
            return NotImplemented
    # when chaining operations this is used for [weights1, weights2] + weights3
    __radd__ = __add__
    
    def __mul__(self, other):
        if isinstance(other, AlgebraicMatrix):
            ## create the product network
            print('prod', self.exp)
            # how to pass in exp?
            prod_layer = ProductVec(shape=self.shape, vth=1, exp=0)
        
            prod_layer << (self, other)
            
            return prod_layer
        else:
            return NotImplemented

