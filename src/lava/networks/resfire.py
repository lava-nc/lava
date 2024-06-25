# Copyright (C) 2022-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from .network import AlgebraicVector
from lava.proc.resfire.process import RFZero


class ResFireVec(AlgebraicVector):
    """
    Network wrapper for resonate-and-fire neurons
    """

    def __init__(self, **kwargs):
        self.uth = kwargs.pop('uth', 10)
        self.shape = kwargs.pop('shape', (1,))
        self.freqs = kwargs.pop('freqs', np.array([10]))
        self.decay_tau = kwargs.pop('decay_tau', np.array([1]))
        self.dt = kwargs.pop('dt', 0.001)

        self.freqs = np.array(self.freqs)
        self.decay_tau = np.array(self.decay_tau)

        self.main = RFZero(shape=self.shape, uth=self.uth,
                           freqs=self.freqs, decay_tau=self.decay_tau,
                           dt=self.dt)

        self.in_port = self.main.u_in
        self.in_port2 = self.main.v_in

        self.out_port = self.main.s_out

    def __lshift__(self, other):
        # We're going to override the behavior here
        # since theres two ports the API idea is:
        #   rf_layer << (conn1, conn2)
        if isinstance(other, (list, tuple)):
            # it should be only length 2, and a Network object,
            # add checks
            other[0].out_port.connect(self.in_port)
            other[1].out_port.connect(self.in_port2)
        else:
            # in this case we will just connect to in_port
            super().__lshift__(other)
