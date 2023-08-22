# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import platform
import numpy as np

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol
from lava.magma.core.model.py.model import (
    PyLoihiProcessModel, PyLoihiModelToPyAsyncModel
)

from lava.proc.lif.process import LIF
from lava.proc.sdn.process import SigmaDelta
from lava.proc.dense.process import Dense

from lava.proc.lif.models import PyLifModelFloat, PyLifModelBitAcc
from lava.proc.sdn.models import PySigmaDeltaModelFixed
from lava.proc.dense.models import PyDenseModelFloat


py_loihi_models = [PyLifModelFloat, PyLifModelBitAcc, PySigmaDeltaModelFixed,
                   PyDenseModelFloat]


class CustomRunConfig(Loihi2SimCfg):
    """Custom run config that converts PyLoihi models to PyAsync models."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def select(self, proc, proc_models):
        pm = super().select(proc, proc_models)
        if issubclass(pm, PyLoihiProcessModel):
            return PyLoihiModelToPyAsyncModel(pm)
        return pm


class TestPyLoihiToPyAsync(unittest.TestCase):
    @unittest.skipIf(platform.system() == 'Windows',
                     'Model conversion is not supported on Windows.')
    def test_model_conversion(self):
        """Test model conversion"""
        for py_loihi_model in py_loihi_models:
            py_async_model = PyLoihiModelToPyAsyncModel(py_loihi_model)
            self.assertTrue(py_loihi_model.implements_process
                            == py_async_model.implements_process)
            self.assertTrue(
                py_async_model.implements_protocol == AsyncProtocol)
            self.assertTrue(hasattr(py_async_model, 'run_async'))

    @unittest.skipIf(platform.system() == 'Windows',
                     'Model conversion is not supported on Windows.')
    def test_lif_dense_lif(self):
        """Test LIF-Dense-LIF equivalency."""
        in_size = 10
        out_size = 8
        weights = np.arange(in_size * out_size).reshape(out_size, in_size) - 25
        weights *= 2
        bias = 20 + np.arange(in_size)

        input_lif_params = {'shape': (in_size,),
                            'du': 0,
                            'dv': 0,
                            'bias_mant': bias,
                            'bias_exp': 6,
                            'vth': 20}
        output_lif_params = {'shape': (out_size,),
                             'du': 4096,
                             'dv': 4096,
                             'vth': 1024}
        dense_params = {'weights': weights}

        input_lif = LIF(**input_lif_params)
        output_lif = LIF(**output_lif_params)
        dense = Dense(**dense_params)
        input_lif.s_out.connect(dense.s_in)
        dense.a_out.connect(output_lif.a_in)

        run_cnd = RunSteps(num_steps=2)
        # simply run the pyproc model
        output_lif.run(condition=run_cnd,
                       run_cfg=Loihi2SimCfg(select_tag='fixed_pt'))
        current_gt = output_lif.u.get()
        voltage_gt = output_lif.v.get()
        output_lif.stop()

        # Run the same network in async mode.
        # Currently we don't allow the same process to run twice
        # Copy the model used for pyproc model
        input_lif = LIF(**input_lif_params)
        output_lif = LIF(**output_lif_params)
        dense = Dense(**dense_params)
        input_lif.s_out.connect(dense.s_in)
        dense.a_out.connect(output_lif.a_in)
        output_lif.run(condition=run_cnd,
                       run_cfg=CustomRunConfig(select_tag='fixed_pt'))
        current = output_lif.u.get()
        voltage = output_lif.v.get()
        output_lif.stop()

        self.assertTrue(np.array_equal(current, current_gt))
        self.assertTrue(np.array_equal(voltage, voltage_gt))

    @unittest.skipIf(platform.system() == 'Windows',
                     'Model conversion is not supported on Windows.')
    def test_sdn_dense_sdn(self):
        """Test LIF-Dense-LIF equivalency."""
        in_size = 10
        out_size = 8
        weights = np.arange(in_size * out_size).reshape(out_size, in_size) - 25
        weights *= 2
        bias = 20 + np.arange(in_size)

        input_params = {'shape': (in_size,),
                        'vth': 22,
                        'bias': bias,
                        'spike_exp': 6,
                        'state_exp': 6}
        output_params = {'shape': (out_size,),
                         'vth': 2,
                         'spike_exp': 6,
                         'state_exp': 6}
        dense_params = {'weights': weights, 'num_message_bits': 16}

        input_ = SigmaDelta(**input_params)
        output = SigmaDelta(**output_params)
        dense = Dense(**dense_params)
        input_.s_out.connect(dense.s_in)
        dense.a_out.connect(output.a_in)

        run_cnd = RunSteps(num_steps=2)
        # simply run the pyproc model
        output.run(condition=run_cnd,
                   run_cfg=Loihi2SimCfg(select_tag='fixed_pt'))
        sigma_gt = output.sigma.get()
        output.stop()

        # Run the same network in async mode.
        # Currently, we don't allow the same process to run twice
        # Copy the model used for pyproc model
        input_ = SigmaDelta(**input_params)
        output = SigmaDelta(**output_params)
        dense = Dense(**dense_params)
        input_.s_out.connect(dense.s_in)
        dense.a_out.connect(output.a_in)

        output.run(condition=run_cnd,
                   run_cfg=CustomRunConfig(select_tag='fixed_pt'))
        sigma = output.sigma.get()
        output.stop()

        self.assertTrue(np.array_equal(sigma, sigma_gt))
