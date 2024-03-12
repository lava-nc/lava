# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
from typing import Tuple
import lava.proc.io as io
from lava.magma.core.run_conditions import RunSteps
from lava.proc.sdn.process import ActivationMode, SigmaDelta
from lava.proc.s4d.process import SigmaS4dDelta, SigmaS4dDeltaLayer
from lava.proc.sparse.process import Sparse
from lava.magma.core.run_configs import Loihi2SimCfg
from tests.lava.proc.s4d.utils import get_coefficients, run_original_model


class TestSigmaS4DDeltaModels(unittest.TestCase):
    """Tests for SigmaS4Delta neuron"""
    def run_in_lava(
            self,
            inp,
            a: np.ndarray,
            b: np.ndarray,
            c: np.ndarray,
            num_steps: int,
            model_dim: int,
            d_states: int,
            use_layer: bool) -> Tuple[np.ndarray]:

        """ Run S4d model in lava.

        Parameters
        ----------
        inp : np.ndarray
            Input signal to the model.
        num_steps : int
            Number of time steps to simulate the model.
        model_dim : int
            Dimensionality of the model.
        d_states : int
            Number of model states.
        use_layer : bool
            Whether to use the layer implementation of the model
            (SigmaS4DeltaLayer, helpful for multiple d_states) or just
            the neuron model (SigmaS4Delta).

        Returns
        -------
        Tuple[np.ndarray]
            Tuple containing the output of the model simulation.
        """

        a = a[:model_dim * d_states]
        b = b[:model_dim * d_states]
        c = c[:model_dim * d_states]

        diff = inp[:, 1:] - inp[:, :-1]
        diff = np.concatenate((inp[:, :1], diff), axis=1)

        spiker = io.source.RingBuffer(data=diff)
        receiver = io.sink.RingBuffer(shape=(model_dim,), buffer=num_steps)

        if use_layer:
            s4d_layer = SigmaS4dDeltaLayer(shape=(model_dim,),
                                           d_states=d_states,
                                           num_message_bits=24,
                                           vth=0,
                                           a=a,
                                           b=b,
                                           c=c)
            buffer_neuron = SigmaDelta(shape=(model_dim,),
                                       vth=0,
                                       cum_error=True,
                                       act_mode=ActivationMode.UNIT)
            spiker.s_out.connect(s4d_layer.s_in)
            s4d_layer.a_out.connect(buffer_neuron.a_in)
            buffer_neuron.s_out.connect(receiver.a_in)

        else:
            sparse = Sparse(weights=np.eye(model_dim), num_message_bits=24)
            s4d_neuron = SigmaS4dDelta(shape=((model_dim,)),
                                       vth=0,
                                       a=a,
                                       b=b,
                                       c=c)
            spiker.s_out.connect(sparse.s_in)
            sparse.a_out.connect(s4d_neuron.a_in)
            s4d_neuron.s_out.connect(receiver.a_in)

        run_condition = RunSteps(num_steps=num_steps)
        run_config = Loihi2SimCfg()

        spiker.run(condition=run_condition, run_cfg=run_config)
        output = receiver.data.get()
        spiker.stop()

        output = np.cumsum(output, axis=1)

        return output

    def test_py_model_vs_original_equations(self) -> None:
        """Tests that the pymodel for SigmaS4dDelta outputs approximately
           the same values as the original S4D equations.
        """
        a, b, c = get_coefficients()
        model_dim = 3
        d_states = 1
        n_steps = 5
        np.random.seed(0)
        inp = np.random.random((model_dim, n_steps)) * 2**6

        out_chip = self.run_in_lava(inp=inp,
                                    a=a,
                                    b=b,
                                    c=c,
                                    num_steps=n_steps,
                                    model_dim=model_dim,
                                    d_states=d_states,
                                    use_layer=False
                                    )
        out_original_model = run_original_model(inp=inp,
                                                model_dim=model_dim,
                                                d_states=d_states,
                                                num_steps=n_steps,
                                                a=a,
                                                b=b,
                                                c=c)

        np.testing.assert_array_equal(out_original_model[:, :-1],
                                      out_chip[:, 1:])

    def test_py_model_layer_vs_original_equations(self) -> None:
        """ Tests that the pymodel for SigmaS4DeltaLayer outputs approximately
           the same values as the original S4D equations for multiple d_states.
        """
        a, b, c = get_coefficients()
        model_dim = 3
        d_states = 3
        n_steps = 5
        np.random.seed(1)
        inp = np.random.random((model_dim, n_steps)) * 2**6

        out_chip = self.run_in_lava(inp=inp,
                                    a=a,
                                    b=b,
                                    c=c,
                                    num_steps=n_steps,
                                    model_dim=model_dim,
                                    d_states=d_states,
                                    use_layer=True,
                                    )
        out_original_model = run_original_model(inp=inp,
                                                model_dim=model_dim,
                                                d_states=d_states,
                                                num_steps=n_steps,
                                                a=a,
                                                b=b,
                                                c=c)

        np.testing.assert_allclose(out_original_model[:, :-2], out_chip[:, 2:])

    def test_py_model_vs_py_model_layer(self) -> None:
        """Tests that the  pymodel for SigmaS4DeltaLayer outputs approximately
           the same values as just the SigmaS4DDelta Model with one hidden dim.
        """
        a, b, c = get_coefficients()
        model_dim = 3
        d_states = 1
        n_steps = 5
        np.random.seed(2)
        inp = np.random.random((model_dim, n_steps)) * 2**6

        out_just_model = self.run_in_lava(inp=inp,
                                          a=a,
                                          b=b,
                                          c=c,
                                          num_steps=n_steps,
                                          model_dim=model_dim,
                                          d_states=d_states,
                                          use_layer=False)

        out_layer = self.run_in_lava(inp=inp,
                                     a=a,
                                     b=b,
                                     c=c,
                                     num_steps=n_steps,
                                     model_dim=model_dim,
                                     d_states=d_states,
                                     use_layer=True)

        np.testing.assert_allclose(out_layer[:, 1:], out_just_model[:, :-1])


if __name__ == '__main__':
    unittest.main()
