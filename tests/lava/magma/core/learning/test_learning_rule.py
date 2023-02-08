# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.magma.core.learning.learning_rule import LoihiLearningRule
from lava.magma.core.learning.product_series import ProductSeries
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.proc.lif.process import LIF, LearningLIF
from lava.proc.dense.process import LearningDense, Dense
from lava.proc.monitor.process import Monitor
from lava.proc.io.source import RingBuffer as SpikeIn


def create_network(size, num_steps, t_spike, learning_rule, weights_init):
        pre_spikes = np.zeros((size, num_steps * 2))
        pre_spikes[size-1, t_spike] = 1
        pre_spikes[size-1, t_spike + num_steps] = 1

        spike_gen = SpikeIn(data=pre_spikes)

        dense_inp = Dense(weights=np.eye(size, size) * 2.0)

        lif_0 = LIF(shape=(size,), 
                    du=1, 
                    dv=1, 
                    vth=1, 
                    bias_mant=0.)

        dense = LearningDense(weights=weights_init, 
                              learning_rule=learning_rule,
                              name='plastic_dense')

        lif_1 = LearningLIF(shape=(size,),
                            du=1,
                            dv=1,
                            vth=1,
                            bias_mant=0.0)

        spike_gen.s_out.connect(dense_inp.s_in)
        dense_inp.a_out.connect(lif_0.a_in)
        lif_0.s_out.connect(dense.s_in)
        dense.a_out.connect(lif_1.a_in)
        lif_1.s_out_bap.connect(dense.s_in_bap)


        return spike_gen, dense_inp, lif_0, dense, lif_1



class TestLoihiLearningRule(unittest.TestCase):
    def test_learning_rule_dw(self) -> None:
        """Tests that a LoihiLearningRule is instantiable with string
        learning rule for dw, impulse and tau values for x1 and y1,
        and t_epoch."""
        dw = 'x0*(-1)*2^-1*y1 + y0*1*2^1*x1'
        impulse = 16
        tau = 10
        t_epoch = 1

        learning_rule = LoihiLearningRule(dw=dw,
                                          x1_impulse=impulse, x1_tau=tau,
                                          y1_impulse=impulse, y1_tau=tau,
                                          t_epoch=t_epoch)

        self.assertIsInstance(learning_rule, LoihiLearningRule)
        self.assertIsInstance(learning_rule.dw, ProductSeries)
        self.assertEqual(learning_rule.dd, None)
        self.assertEqual(learning_rule.dt, None)
        self.assertEqual(learning_rule.x1_impulse, impulse)
        self.assertEqual(learning_rule.x1_tau, tau)
        self.assertEqual(learning_rule.y1_impulse, impulse)
        self.assertEqual(learning_rule.y1_tau, tau)
        self.assertEqual(learning_rule.t_epoch, t_epoch)
        self.assertEqual(learning_rule.decimate_exponent, None)
        self.assertEqual(len(learning_rule.active_product_series), 1)
        self.assertSetEqual(learning_rule.active_traces, {"x1", "y1"})
        self.assertDictEqual(learning_rule.active_traces_per_dependency, {
            "x0": {"y1"},
            "y0": {"x1"}
        })

    def test_learning_rule_dw_dd(self) -> None:
        """Tests that a LoihiLearningRule is instantiable with string
        learning rule for dw and dd, impulse and tau values for x1 and y1,
        and t_epoch."""
        dw = 'x0*(-1)*2^-1*y1 + y0*1*2^1*x1'
        dd = 'x0*y2*w'
        impulse = 16
        tau = 10
        t_epoch = 1

        learning_rule = LoihiLearningRule(dw=dw, dd=dd,
                                          x1_impulse=impulse, x1_tau=tau,
                                          y1_impulse=impulse, y1_tau=tau,
                                          t_epoch=t_epoch)

        self.assertIsInstance(learning_rule, LoihiLearningRule)
        self.assertIsInstance(learning_rule.dw, ProductSeries)
        self.assertIsInstance(learning_rule.dd, ProductSeries)
        self.assertEqual(learning_rule.dt, None)
        self.assertEqual(learning_rule.x1_impulse, impulse)
        self.assertEqual(learning_rule.x1_tau, tau)
        self.assertEqual(learning_rule.y1_impulse, impulse)
        self.assertEqual(learning_rule.y1_tau, tau)
        self.assertEqual(learning_rule.t_epoch, t_epoch)
        self.assertEqual(learning_rule.decimate_exponent, None)
        self.assertEqual(len(learning_rule.active_product_series), 2)
        self.assertSetEqual(learning_rule.active_traces, {"x1", "y1", "y2"})
        self.assertDictEqual(learning_rule.active_traces_per_dependency, {
            "x0": {"y1", "y2"},
            "y0": {"x1"}
        })

    def test_learning_rule_dw_dd_dt(self) -> None:
        """Tests that a LoihiLearningRule is instantiable with string
        learning rule for dw, dd and dt, impulse and tau values for x1 and y1,
        and t_epoch."""
        dw = 'x0*(-1)*2^-1*y1 + y0*1*2^1*x1'
        dd = 'x0*y2*w'
        dt = 'x0*y3*sgn(d) + y0*x2'
        impulse = 16
        tau = 10
        t_epoch = 1

        learning_rule = LoihiLearningRule(dw=dw, dd=dd, dt=dt,
                                          x1_impulse=impulse, x1_tau=tau,
                                          y1_impulse=impulse, y1_tau=tau,
                                          t_epoch=t_epoch)

        self.assertIsInstance(learning_rule, LoihiLearningRule)
        self.assertIsInstance(learning_rule.dw, ProductSeries)
        self.assertIsInstance(learning_rule.dd, ProductSeries)
        self.assertIsInstance(learning_rule.dt, ProductSeries)
        self.assertEqual(learning_rule.x1_impulse, impulse)
        self.assertEqual(learning_rule.x1_tau, tau)
        self.assertEqual(learning_rule.y1_impulse, impulse)
        self.assertEqual(learning_rule.y1_tau, tau)
        self.assertEqual(learning_rule.t_epoch, t_epoch)
        self.assertEqual(learning_rule.decimate_exponent, None)
        self.assertEqual(len(learning_rule.active_product_series), 3)
        self.assertSetEqual(learning_rule.active_traces,
                            {"x1", "x2", "y1", "y2", "y3"})
        self.assertDictEqual(learning_rule.active_traces_per_dependency, {
            "x0": {"y1", "y2", "y3"},
            "y0": {"x1", "x2"}
        })

    def test_learning_rule_uk_dependency(self) -> None:
        """Tests that a LoihiLearningRule is instantiable with a string
        learning rule containing a uk dependency."""
        dw = 'x0*(-1)*2^-1*y1 + y0*1*2^1*x1'
        dd = 'u0*x2*y2'
        impulse = 16
        tau = 10
        t_epoch = 1

        learning_rule = LoihiLearningRule(dw=dw, dd=dd,
                                          x1_impulse=impulse, x1_tau=tau,
                                          y1_impulse=impulse, y1_tau=tau,
                                          t_epoch=t_epoch)

        self.assertIsInstance(learning_rule, LoihiLearningRule)
        self.assertIsInstance(learning_rule.dw, ProductSeries)
        self.assertIsInstance(learning_rule.dd, ProductSeries)
        self.assertEqual(learning_rule.dt, None)
        self.assertEqual(learning_rule.x1_impulse, impulse)
        self.assertEqual(learning_rule.x1_tau, tau)
        self.assertEqual(learning_rule.y1_impulse, impulse)
        self.assertEqual(learning_rule.y1_tau, tau)
        self.assertEqual(learning_rule.t_epoch, t_epoch)
        self.assertEqual(learning_rule.decimate_exponent, 0)
        self.assertEqual(len(learning_rule.active_product_series), 2)
        self.assertSetEqual(learning_rule.active_traces,
                            {"x1", "x2", "y1", "y2"})
        self.assertDictEqual(learning_rule.active_traces_per_dependency, {
            "x0": {"y1"},
            "y0": {"x1"},
            "u": {"x2", "y2"}
        })

    def test_invalid_impulse(self) -> None:
        """Tests that instantiating a LoihiLearningRule throws error when
        impulse is negative."""
        dw = 'x0*(-1)*2^-1*y1 + y0*1*2^1*x1'
        impulse = -16
        tau = 10
        t_epoch = 1

        with self.assertRaises(ValueError):
            LoihiLearningRule(dw=dw,
                              x1_impulse=impulse, x1_tau=tau,
                              y1_impulse=impulse, y1_tau=tau,
                              t_epoch=t_epoch)

    def test_invalid_tau(self) -> None:
        """Tests that instantiating a LoihiLearningRule throws error when
        tau is negative."""
        dw = 'x0*(-1)*2^-1*y1 + y0*1*2^1*x1'
        impulse = 16
        tau = -10
        t_epoch = 1

        with self.assertRaises(ValueError):
            LoihiLearningRule(dw=dw,
                              x1_impulse=impulse, x1_tau=tau,
                              y1_impulse=impulse, y1_tau=tau,
                              t_epoch=t_epoch)

    def test_invalid_t_epoch(self) -> None:
        """Tests that instantiating a LoihiLearningRule throws error when
        t_epoch is negative."""
        dw = 'x0*(-1)*2^-1*y1 + y0*1*2^1*x1'
        impulse = 16
        tau = 10
        t_epoch = -1

        with self.assertRaises(ValueError):
            LoihiLearningRule(dw=dw,
                              x1_impulse=impulse, x1_tau=tau,
                              y1_impulse=impulse, y1_tau=tau,
                              t_epoch=t_epoch)

    def test_different_decimate_exponent_same_learning_rule(self) -> None:
        """Tests that instantiating a LoihiLearningRule throws error when
        providing a learning rule with uk dependencies with different
        decimate exponents."""
        dw = 'x0*(-1)*2^-1*y1 + y0*1*2^1*x1 + u0*x2*y2 + u1*y3'
        impulse = 16
        tau = 10
        t_epoch = 1

        with self.assertRaises(ValueError):
            LoihiLearningRule(dw=dw,
                              x1_impulse=impulse, x1_tau=tau,
                              y1_impulse=impulse, y1_tau=tau,
                              t_epoch=t_epoch)

    def test_different_decimate_exponent_different_learning_rule(self) -> None:
        """Tests that instantiating a LoihiLearningRule throws error when
        providing different learning rules with uk dependencies with different
        decimate exponents."""
        dw = 'x0*(-1)*2^-1*y1 + y0*1*2^1*x1 + u1*y3'
        dd = 'u0*x2*y2'
        impulse = 16
        tau = 10
        t_epoch = 1

        with self.assertRaises(ValueError):
            LoihiLearningRule(dw=dw, dd=dd,
                              x1_impulse=impulse, x1_tau=tau,
                              y1_impulse=impulse, y1_tau=tau,
                              t_epoch=t_epoch)

    def test_get_set_x1_tau_float(self) -> None:
        """Tests changing x1_tau during runtime in a floating point simulation"""

        dw = "x0 * x1 * 0"
        t_epoch = 1
        x1_tau_init = 2
        x1_tau_new = 10
        x1_impulse = 16

        learning_rule = LoihiLearningRule(dw=dw,
                                          t_epoch=t_epoch,
                                          x1_tau=x1_tau_init,
                                          x1_impulse=x1_impulse)

        size = 1
        weights_init = np.eye(size) * 0
        num_steps = 10 
        t_spike = 3

        _, _, lif_0, dense, _ = create_network(size, 
                                               num_steps, 
                                               t_spike, 
                                               learning_rule, 
                                               weights_init)

       
        mon_x1 = Monitor()
        mon_x1.probe(dense.x1, 2 * num_steps)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt")
        run_cnd = RunSteps(num_steps=num_steps)

        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        # reset x1 and set new tau
        dense.x1.set(np.zeros(size))
        x1_tau_init_got = dense.x1_tau.get()
        dense.x1_tau.set(np.ones(size) * x1_tau_new)
        x1_tau_new_got = dense.x1_tau.get()


        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        x1 = mon_x1.get_data()['plastic_dense']['x1'].flatten()

        lif_0.stop()

        assert(x1_tau_init == x1_tau_init_got)
        assert(x1_tau_new == x1_tau_new_got)

        assert(x1[t_spike + 1] == x1_impulse)
        assert(x1[t_spike + 2] == x1_impulse * np.exp(-1/x1_tau_init))

        assert(x1[t_spike + num_steps + 1] == x1_impulse)
        assert(x1[t_spike + num_steps + 2] == x1_impulse * np.exp(-1/x1_tau_new))


    def test_get_set_x1_impulse_float(self) -> None:
        """Tests changing x1_impulse during runtime in a floating point simulation"""

        dw = "x0 * x1 * 0"
        t_epoch = 1
        x1_impulse_init = 16
        x1_impulse_new = 32
        x1_tau = 10

        learning_rule = LoihiLearningRule(dw=dw,
                                          t_epoch=t_epoch,
                                          x1_tau=x1_tau,
                                          x1_impulse=x1_impulse_init)

        size = 1
        weights_init = np.eye(size) * 0
        num_steps = 10 
        t_spike = 3

        _, _, lif_0, dense, _ = create_network(size, 
                                               num_steps, 
                                               t_spike, 
                                               learning_rule, 
                                               weights_init)

       
        mon_x1 = Monitor()
        mon_x1.probe(dense.x1, 2 * num_steps)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt")
        run_cnd = RunSteps(num_steps=num_steps)

        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        # reset x1 and set new impulse
        dense.x1.set(np.zeros(size))
        x1_impulse_init_got = dense.x1_impulse.get()
        dense.x1_impulse.set(np.ones(size) * x1_impulse_new)
        x1_impulse_new_got = dense.x1_impulse.get()


        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        x1 = mon_x1.get_data()['plastic_dense']['x1'].flatten()

        lif_0.stop()

        assert(x1_impulse_init == x1_impulse_init_got)
        assert(x1_impulse_new == x1_impulse_new_got)

        assert(x1[t_spike + 1] == x1_impulse_init)
        assert(x1[t_spike + num_steps + 1] == x1_impulse_new)

        

    def test_get_set_x2_tau_float(self) -> None:
        """Tests changing x2_tau during runtime in a floating point simulation"""

        dw = "x0 * x2 * 0"
        t_epoch = 1
        x2_tau_init = 2
        x2_tau_new = 10
        x2_impulse = 16

        learning_rule = LoihiLearningRule(dw=dw,
                                          t_epoch=t_epoch,
                                          x2_tau=x2_tau_init,
                                          x2_impulse=x2_impulse)

        size = 1
        weights_init = np.eye(size) * 0
        num_steps = 10 
        t_spike = 3

        _, _, lif_0, dense, _ = create_network(size, 
                                               num_steps, 
                                               t_spike, 
                                               learning_rule, 
                                               weights_init)

       
        mon_x2 = Monitor()
        mon_x2.probe(dense.x2, 2 * num_steps)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt")
        run_cnd = RunSteps(num_steps=num_steps)

        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        # reset x2 and set new tau
        dense.x2.set(np.zeros(size))
        x2_tau_init_got = dense.x2_tau.get()
        dense.x2_tau.set(np.ones(size) * x2_tau_new)
        x2_tau_new_got = dense.x2_tau.get()


        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        x2 = mon_x2.get_data()['plastic_dense']['x2'].flatten()

        lif_0.stop()

        assert(x2_tau_init == x2_tau_init_got)
        assert(x2_tau_new == x2_tau_new_got)

        assert(x2[t_spike + 1] == x2_impulse)
        assert(x2[t_spike + 2] == x2_impulse * np.exp(-1/x2_tau_init))

        assert(x2[t_spike + num_steps + 1] == x2_impulse)
        assert(x2[t_spike + num_steps + 2] == x2_impulse * np.exp(-1/x2_tau_new))


    def test_get_set_x2_impulse_float(self) -> None:
        """Tests changing x2_impulse during runtime in a floating point simulation"""

        dw = "x0 * x2 * 0"
        t_epoch = 1
        x2_impulse_init = 16
        x2_impulse_new = 32
        x2_tau = 10

        learning_rule = LoihiLearningRule(dw=dw,
                                          t_epoch=t_epoch,
                                          x2_tau=x2_tau,
                                          x2_impulse=x2_impulse_init)

        size = 1
        weights_init = np.eye(size) * 0
        num_steps = 10 
        t_spike = 3

        _, _, lif_0, dense, _ = create_network(size, 
                                               num_steps, 
                                               t_spike, 
                                               learning_rule, 
                                               weights_init)

       
        mon_x2 = Monitor()
        mon_x2.probe(dense.x2, 2 * num_steps)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt")
        run_cnd = RunSteps(num_steps=num_steps)

        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        # reset x2 and set new impulse
        dense.x2.set(np.zeros(size))
        x2_impulse_init_got = dense.x2_impulse.get()
        dense.x2_impulse.set(np.ones(size) * x2_impulse_new)
        x2_impulse_new_got = dense.x2_impulse.get()


        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        x2 = mon_x2.get_data()['plastic_dense']['x2'].flatten()

        lif_0.stop()

        assert(x2_impulse_init == x2_impulse_init_got)
        assert(x2_impulse_new == x2_impulse_new_got)

        assert(x2[t_spike + 1] == x2_impulse_init)
        assert(x2[t_spike + num_steps + 1] == x2_impulse_new)



    def test_get_set_dw_float(self) -> None:
        """Tests changing dw during runtime in a floating point simulation"""

        dw_init = "u0 * 1"
        dw_new= "u0 *  2 "
        t_epoch = 1

        learning_rule = LoihiLearningRule(dw=dw_init,
                                          t_epoch=t_epoch)

        size = 1
        weights_init = np.eye(size) * 0
        num_steps = 10 
        t_spike = 3

        _, _, lif_0, dense, _ = create_network(size, 
                                               num_steps, 
                                               t_spike, 
                                               learning_rule, 
                                               weights_init)

       
        mon_weights = Monitor()
        mon_weights.probe(dense.weights, 2 * num_steps)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt")
        run_cnd = RunSteps(num_steps=num_steps)

        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        dw_init_got = dense.dw.get()

        dense.dw.set(dw_new)
        dw_new_got = dense.dw.get()

        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        weights = mon_weights.get_data()['plastic_dense']['weights'].flatten()

        lif_0.stop()

        assert(dw_init == dw_init_got[:len(dw_init)])
        assert(dw_new == dw_new_got[:len(dw_new)])

        assert(weights[-1] == 30)




    def test_get_set_dt_float(self) -> None:
        """Tests changing dt during runtime in a floating point simulation"""

        dt_init = "u0 * 1"
        dt_new= "u0 *  2 "
        t_epoch = 1

        learning_rule = LoihiLearningRule(dt=dt_init,
                                          t_epoch=t_epoch)

        size = 1
        weights_init = np.eye(size) * 0
        num_steps = 10 
        t_spike = 3

        _, _, lif_0, dense, _ = create_network(size, 
                                               num_steps, 
                                               t_spike, 
                                               learning_rule, 
                                               weights_init)

       
        mon_tag1 = Monitor()
        mon_tag1.probe(dense.tag_1, 2 * num_steps)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt")
        run_cnd = RunSteps(num_steps=num_steps)

        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        dt_init_got = dense.dt.get()

        dense.dt.set(dt_new)
        dt_new_got = dense.dt.get()

        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        tags= mon_tag1.get_data()['plastic_dense']['tag_1'].flatten()

        lif_0.stop()

        assert(dt_init == dt_init_got[:len(dt_init)])
        assert(dt_new == dt_new_got[:len(dt_new)])

        assert(tags[-1] == 30)



    def test_get_set_dd_float(self) -> None:
        """Tests changing dd during runtime in a floating point simulation"""

        dd_init = "u0 * 1"
        dd_new= "u0 *  2 "
        t_epoch = 1

        learning_rule = LoihiLearningRule(dd=dd_init,
                                          t_epoch=t_epoch)

        size = 1
        weights_init = np.eye(size) * 0
        num_steps = 10 
        t_spike = 3

        _, _, lif_0, dense, _ = create_network(size, 
                                               num_steps, 
                                               t_spike, 
                                               learning_rule, 
                                               weights_init)

       
        mon_tag1 = Monitor()
        mon_tag1.probe(dense.tag_2, 2 * num_steps)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt")
        run_cnd = RunSteps(num_steps=num_steps)

        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        dd_init_got = dense.dd.get()

        dense.dd.set(dd_new)
        dd_new_got = dense.dd.get()

        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        delays = mon_tag1.get_data()['plastic_dense']['tag_2'].flatten()

        lif_0.stop()

        assert(dd_init == dd_init_got[:len(dd_init)])
        assert(dd_new == dd_new_got[:len(dd_new)])

        assert(delays[-1] == 30)



        

        



 






if __name__ == "__main__":
    unittest.main()
