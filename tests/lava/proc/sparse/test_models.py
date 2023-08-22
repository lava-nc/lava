# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
from lava.magma.core.learning.learning_rule import Loihi2FLearningRule
from scipy.sparse import csr_matrix

from lava.proc.dense.process import LearningDense
from lava.proc.sparse.process import Sparse, DelaySparse, LearningSparse
from lava.proc.learning_rules.stdp_learning_rule import STDPLoihi
from lava.proc.dense.process import Dense
from lava.proc.sparse.models import AbstractPyDelaySparseModel as APDSM
from lava.proc.io.source import RingBuffer as Source
from lava.proc.io.sink import RingBuffer as Sink

from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort, PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort, InPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.dense.models import AbstractPyDelayDenseModel
from lava.utils.weightutils import SignMode


def create_network(input_data, conn, weights):
    source = Source(data=input_data)
    sink = Sink(shape=(weights.shape[0], ),
                buffer=input_data.shape[1])

    source.s_out.connect(conn.s_in)
    conn.a_out.connect(sink.a_in)

    return source, conn, sink


def create_learning_network(data_pre, conn, data_post, weights=None):
    pre = Source(data=data_pre)
    post = Source(data=data_post)

    if weights is not None:
        sink = Sink(shape=(weights.shape[0], ),
                    buffer=data_post.shape[1])

        conn.a_out.connect(sink.a_in)

    pre.s_out.connect(conn.s_in)
    post.s_out.connect(conn.s_in_bap)

    if weights is not None:
        return pre, conn, post, sink
    else:
        return pre, conn, post


class TestSparseProcessModelFloat(unittest.TestCase):
    """Tests for Sparse class in floating point precision. """

    def test_consistency_with_dense_random_shape(self):
        """Tests if the results of Sparse and Dense are consistent. """

        simtime = 10
        shape = np.random.randint(1, 300, 2).tolist()
        weights = (np.random.random(shape) - 0.5) * 2

        # Sparsify
        weights[np.abs(weights) < 0.7] = 0

        inp = (np.random.rand(shape[1], simtime) > 0.7).astype(int)

        conn = Dense(weights=weights)
        dense_net = create_network(inp, conn, weights)

        run_cond = RunSteps(num_steps=simtime)
        run_cfg = Loihi2SimCfg(select_tag='floating_pt')

        conn.run(condition=run_cond, run_cfg=run_cfg)
        # Weights_got_dense = conn.weights.get()
        result_dense = dense_net[2].data.get()
        conn.stop()

        # Run the same network with Sparse

        # Convert to spmatrix
        weights_sparse = csr_matrix(weights)

        conn = Sparse(weights=weights_sparse)
        sparse_net = create_network(inp, conn, weights_sparse)
        conn.run(condition=run_cond, run_cfg=run_cfg)

        # Weights_got_sparse = conn.weights.get()
        result_sparse = sparse_net[2].data.get()
        conn.stop()

        np.testing.assert_array_almost_equal(result_sparse, result_dense)

    def test_consistency_with_dense_random_shape_graded(self):
        """Tests if the results of Sparse and Dense are consistent. """

        simtime = 10
        shape = np.random.randint(1, 300, 2).tolist()
        weights = (np.random.random(shape) - 0.5) * 2

        # Sparsify
        weights[np.abs(weights) < 0.7] = 0

        inp = (np.random.rand(shape[1], simtime) * 10).astype(int)

        conn = Dense(weights=weights, num_message_bits=8)
        dense_net = create_network(inp, conn, weights)

        run_cond = RunSteps(num_steps=simtime)
        run_cfg = Loihi2SimCfg(select_tag='floating_pt')

        dense_net[0].run(condition=run_cond, run_cfg=run_cfg)
        result_dense = dense_net[2].data.get()
        dense_net[0].stop()

        # Run the same network with Sparse

        # Convert to spmatrix
        weights_sparse = csr_matrix(weights)

        conn = Sparse(weights=weights_sparse, num_message_bits=8)
        sparse_net = create_network(inp, conn, weights_sparse)
        sparse_net[0].run(condition=run_cond, run_cfg=run_cfg)

        result_sparse = sparse_net[2].data.get()
        sparse_net[0].stop()

        np.testing.assert_array_almost_equal(result_sparse, result_dense)

    def test_weights_get(self):
        """Tests the get method on weights."""

        simtime = 10
        shape = np.random.randint(1, 300, 2).tolist()
        weights = (np.random.random(shape) - 0.5) * 2

        # Sparsify
        weights[np.abs(weights) < 0.7] = 0
        # Convert to spmatrix
        weights_sparse = csr_matrix(weights)

        inp = (np.random.rand(shape[1], simtime) * 10).astype(int)

        run_cond = RunSteps(num_steps=simtime)
        run_cfg = Loihi2SimCfg(select_tag='floating_pt')

        conn = Sparse(weights=weights_sparse)
        create_network(inp, conn, weights_sparse)
        conn.run(condition=run_cond, run_cfg=run_cfg)

        weights_got = conn.weights.get()
        conn.stop()

        self.assertIsInstance(weights_got, csr_matrix)
        np.testing.assert_array_equal(weights_got.toarray(), weights)

    def test_weights_set(self):
        """Tests the set method on weights."""
        simtime = 2
        shape = np.random.randint(1, 300, 2).tolist()
        weights_init = (np.random.random(shape) - 0.5) * 2

        # Sparsify
        weights_init[np.abs(weights_init) < 0.7] = 0
        weights_init_sparse = csr_matrix(weights_init)

        new_weights_sparse = weights_init_sparse.copy()
        new_weights_sparse.data = (np.random.random(
            new_weights_sparse.data.shape) - 0.5) * 2

        inp = (np.random.rand(shape[1], simtime) * 10).astype(int)

        run_cond = RunSteps(num_steps=1)
        run_cfg = Loihi2SimCfg(select_tag='floating_pt')

        conn = Sparse(weights=weights_init_sparse)
        create_network(inp, conn, weights_init_sparse)
        conn.run(condition=run_cond, run_cfg=run_cfg)

        new_weights_sparse = conn.weights.init.copy()
        new_weights_sparse.data = np.random.permutation(new_weights_sparse.data)

        conn.weights.set(new_weights_sparse)

        conn.run(condition=run_cond, run_cfg=run_cfg)

        weights_got_ts_2 = conn.weights.get()

        conn.stop()

        self.assertIsInstance(weights_got_ts_2, csr_matrix)
        np.testing.assert_array_equal(weights_got_ts_2.toarray(),
                                      new_weights_sparse.toarray())


class TestSparseProcessModelFixed(unittest.TestCase):
    """Tests for Sparse class in fixed point precision. """

    def test_consitency_with_dense_random_shape(self):
        """Tests if the results of Sparse and Dense are consistent. """

        simtime = 10
        shape = np.random.randint(1, 300, 2).tolist()
        weights = (np.random.random(shape) - 0.5) * 2

        # Sparsify
        weights[np.abs(weights) < 0.7] = 0
        weights *= 20
        weights = weights.astype(int)

        inp = (np.random.rand(shape[1], simtime) > 0.7).astype(int)

        conn = Dense(weights=weights)
        dense_net = create_network(inp, conn, weights)

        run_cond = RunSteps(num_steps=simtime)
        run_cfg = Loihi2SimCfg(select_tag='fixed_pt')

        dense_net[0].run(condition=run_cond, run_cfg=run_cfg)
        result_dense = dense_net[2].data.get()
        dense_net[0].stop()

        # Run the same network with Sparse

        # Convert to spmatrix
        weights_sparse = csr_matrix(weights)

        conn = Sparse(weights=weights_sparse)
        sparse_net = create_network(inp, conn, weights_sparse)
        sparse_net[0].run(condition=run_cond, run_cfg=run_cfg)

        result_sparse = sparse_net[2].data.get()
        sparse_net[0].stop()

        np.testing.assert_array_almost_equal(result_sparse, result_dense)

    def test_consitency_with_dense_random_shape_graded(self):
        """Tests if the results of Sparse and Dense are consistent. """

        simtime = 10
        shape = np.random.randint(1, 300, 2).tolist()
        weights = (np.random.random(shape) - 0.5) * 2

        # Sparsify
        weights[np.abs(weights) < 0.7] = 0
        weights *= 20
        weights = weights.astype(int)

        inp = (np.random.rand(shape[1], simtime) * 10).astype(int)

        conn = Dense(weights=weights, num_message_bits=8)
        dense_net = create_network(inp, conn, weights)

        run_cond = RunSteps(num_steps=simtime)
        run_cfg = Loihi2SimCfg(select_tag='fixed_pt')

        dense_net[0].run(condition=run_cond, run_cfg=run_cfg)
        result_dense = dense_net[2].data.get()
        dense_net[0].stop()

        # Run the same network with Sparse

        # Convert to spmatrix
        weights_sparse = csr_matrix(weights)

        conn = Sparse(weights=weights_sparse, num_message_bits=8)
        sparse_net = create_network(inp, conn, weights_sparse)
        sparse_net[0].run(condition=run_cond, run_cfg=run_cfg)

        result_sparse = sparse_net[2].data.get()
        sparse_net[0].stop()

        np.testing.assert_array_almost_equal(result_sparse, result_dense)

    def test_weights_get(self):
        """Tests the get method on weights."""

        simtime = 10
        shape = np.random.randint(1, 300, 2).tolist()
        weights_init = (np.random.random(shape) - 0.5) * 2

        # Sparsify
        weights_init[np.abs(weights_init) < 0.7] = 0
        weights_init *= 20
        weights_init = weights_init.astype(int)
        weights_sparse = csr_matrix(weights_init)

        inp = (np.random.rand(shape[1], simtime) * 10).astype(int)

        run_cond = RunSteps(num_steps=simtime)
        run_cfg = Loihi2SimCfg(select_tag='floating_pt')

        conn = Sparse(weights=weights_sparse)
        sparse_net = create_network(inp, conn, weights_sparse)
        conn.run(condition=run_cond, run_cfg=run_cfg)

        weights_got = conn.weights.get()
        conn.stop()

        self.assertIsInstance(weights_got, csr_matrix)
        np.testing.assert_array_equal(weights_got.toarray(), weights_init)

    def test_weights_set(self):
        """tests the set method on weights."""
        simtime = 2
        shape = np.random.randint(1, 300, 2).tolist()
        weights_init = (np.random.random(shape) - 0.5) * 2

        # Sparsify
        weights_init[np.abs(weights_init) < 0.7] = 0
        weights_init *= 20
        weights_init = weights_init.astype(int)
        weights_init_sparse = csr_matrix(weights_init)

        inp = (np.random.rand(shape[1], simtime) * 10).astype(int)

        run_cond = RunSteps(num_steps=1)
        run_cfg = Loihi2SimCfg(select_tag='floating_pt')

        conn = Sparse(weights=weights_init_sparse)
        sparse_net = create_network(inp, conn, weights_init_sparse)
        conn.run(condition=run_cond, run_cfg=run_cfg)

        new_weights_sparse = conn.weights.init.copy()
        new_weights_sparse.data = np.random.permutation(new_weights_sparse.data)

        conn.weights.set(new_weights_sparse)

        conn.run(condition=run_cond, run_cfg=run_cfg)

        weights_got_ts_2 = conn.weights.get()

        conn.stop()

        self.assertIsInstance(weights_got_ts_2, csr_matrix)
        np.testing.assert_array_equal(weights_got_ts_2.toarray(),
                                      new_weights_sparse.toarray())

    def test_weights_set_failure(self):
        """This tests tries to use set() to change weights but fails as
        the number of non-zero weights and their indices change"""
        simtime = 2
        shape = (2, 2)
        weights_init = np.array([[0, 32], [13, 0]])
        new_weights = np.array([[0, 0], [13, 0]])
        new_weights_2 = np.array([[32, 0], [13, 0]])

        weights_init_sparse = csr_matrix(weights_init)
        new_weights_sparse = csr_matrix(new_weights)
        new_weights_sparse_2 = csr_matrix(new_weights_2)

        inp = (np.random.rand(shape[1], simtime) * 10).astype(int)

        run_cond = RunSteps(num_steps=1)
        run_cfg = Loihi2SimCfg(select_tag='floating_pt')

        conn = Sparse(weights=weights_init_sparse)
        sparse_net = create_network(inp, conn, weights_init_sparse)
        conn.run(condition=run_cond, run_cfg=run_cfg)

        self.assertRaises(ValueError, conn.weights.set, new_weights_sparse)
        self.assertRaises(ValueError, conn.weights.set, new_weights_sparse_2)
        conn.stop()


class TestLearningSparseProcessModelFloat(unittest.TestCase):
    """Tests for LearningSparse class in floating point precision. """

    def test_consistency_with_learning_dense_random_shape(self):
        """Tests if the results of LearningSparse and LearningDense
        are consistent. """

        simtime = 10
        shape = np.random.randint(1, 100, 2).tolist()
        weights = (np.random.random(shape) - 0.5) * 2
        weights[weights == 0] = 0.1

        learning_rule = STDPLoihi(
            learning_rate=1,
            A_plus=1,
            A_minus=-1,
            tau_plus=10,
            tau_minus=10,
            t_epoch=2,
        )

        pre = (np.random.rand(shape[1], simtime) > 0.7).astype(int)
        post = (np.random.rand(shape[0], simtime) > 0.7).astype(int)

        conn = LearningDense(weights=weights,
                             tag_1=weights.copy(),
                             tag_2=weights.copy(),
                             learning_rule=learning_rule)
        create_learning_network(pre, conn, post)

        run_cond = RunSteps(num_steps=simtime)
        run_cfg = Loihi2SimCfg(select_tag='floating_pt')

        conn.run(condition=run_cond, run_cfg=run_cfg)
        weights_got_dense = conn.weights.get()
        conn.stop()

        # Run the same network with Sparse

        # Convert to spmatrix
        weights_sparse = csr_matrix(weights)

        conn = LearningSparse(weights=weights_sparse,
                              tag_1=weights_sparse.copy(),
                              tag_2=weights_sparse.copy(),
                              learning_rule=learning_rule)
        create_learning_network(pre, conn, post)
        conn.run(condition=run_cond, run_cfg=run_cfg)

        weights_got_sparse = conn.weights.get()
        conn.stop()

        np.testing.assert_array_equal(weights_got_dense,
                                      weights_got_sparse.toarray())

    def test_consistency_with_learning_dense_random_shape_dt(self):
        """Tests if the results of LearningSparse and LearningDense
        are consistent. """

        simtime = 10
        shape = np.random.randint(1, 100, 2).tolist()
        weights = (np.random.random(shape) - 0.5) * 2
        weights[weights == 0] = 0.1

        learning_rule = Loihi2FLearningRule(dt="x0 * y1 - y0 * x1",
                                            x1_tau=10,
                                            y1_tau=16,
                                            x1_impulse=16,
                                            y1_impulse=15,
                                            t_epoch=2)

        pre = (np.random.rand(shape[1], simtime) > 0.7).astype(int)
        post = (np.random.rand(shape[0], simtime) > 0.7).astype(int)

        conn = LearningDense(weights=weights,
                             tag_1=weights.copy(),
                             tag_2=weights.copy(),
                             learning_rule=learning_rule)
        dense_net = create_learning_network(pre, conn, post)

        run_cond = RunSteps(num_steps=simtime)
        run_cfg = Loihi2SimCfg(select_tag='floating_pt')

        conn.run(condition=run_cond, run_cfg=run_cfg)
        tags_got_dense = conn.tag_1.get()
        conn.stop()

        # Run the same network with Sparse

        # Convert to spmatrix
        weights_sparse = csr_matrix(weights)

        conn = LearningSparse(weights=weights_sparse,
                              tag_1=weights_sparse.copy(),
                              tag_2=weights_sparse.copy(),
                              learning_rule=learning_rule)
        sparse_net = create_learning_network(pre, conn, post)
        conn.run(condition=run_cond, run_cfg=run_cfg)

        tags_got_sparse = conn.tag_1.get()
        conn.stop()

        np.testing.assert_array_equal(tags_got_dense,
                                      tags_got_sparse.toarray())

    def test_consistency_with_learning_dense_random_shape_dd(self):
        """Tests if the results of LearningSparse and LearningDense
        are consistent. """

        simtime = 10
        shape = np.random.randint(1, 100, 2).tolist()
        weights = (np.random.random(shape) - 0.5) * 2
        weights[weights == 0] = 0.1

        learning_rule = Loihi2FLearningRule(dd="x0 * y1 - y0 * x1",
                                            x1_tau=10,
                                            y1_tau=16,
                                            x1_impulse=16,
                                            y1_impulse=15,
                                            t_epoch=2)

        pre = (np.random.rand(shape[1], simtime) > 0.7).astype(int)
        post = (np.random.rand(shape[0], simtime) > 0.7).astype(int)

        conn = LearningDense(weights=weights,
                             tag_1=weights.copy(),
                             tag_2=weights.copy(),
                             learning_rule=learning_rule)
        dense_net = create_learning_network(pre, conn, post)

        run_cond = RunSteps(num_steps=simtime)
        run_cfg = Loihi2SimCfg(select_tag='floating_pt')

        conn.run(condition=run_cond, run_cfg=run_cfg)
        tags_got_dense = conn.tag_2.get()
        conn.stop()

        # Run the same network with Sparse

        # Convert to spmatrix
        weights_sparse = csr_matrix(weights)

        conn = LearningSparse(weights=weights_sparse,
                              tag_1=weights_sparse.copy(),
                              tag_2=weights_sparse.copy(),
                              learning_rule=learning_rule)
        sparse_net = create_learning_network(pre, conn, post)
        conn.run(condition=run_cond, run_cfg=run_cfg)

        tags_got_sparse = conn.tag_2.get()
        conn.stop()

        np.testing.assert_array_equal(tags_got_dense,
                                      tags_got_sparse.toarray())

    def test_consistency_with_learning_dense_random_shape_graded(self):
        """Tests if the results of LearningSparse and LearningDense
        are consistent. """

        simtime = 10
        shape = np.random.randint(1, 100, 2).tolist()
        weights = (np.random.random(shape) - 0.5) * 2
        weights[weights == 0] = 0.1

        learning_rule = STDPLoihi(
            learning_rate=1,
            A_plus=1,
            A_minus=-1,
            tau_plus=10,
            tau_minus=10,
            t_epoch=2,
        )

        pre = (np.random.rand(shape[1], simtime) > 0.7) * 3
        post = (np.random.rand(shape[0], simtime) > 0.7) * 2

        conn = LearningDense(weights=weights,
                             tag_1=weights.copy(),
                             tag_2=weights.copy(),
                             learning_rule=learning_rule,
                             num_message_bits=8)

        dense_net = create_learning_network(pre, conn, post, weights=weights)

        run_cond = RunSteps(num_steps=simtime)
        run_cfg = Loihi2SimCfg(select_tag='floating_pt')

        conn.run(condition=run_cond, run_cfg=run_cfg)
        weights_got_dense = conn.weights.get()
        result_dense = dense_net[-1].data.get()
        conn.stop()

        # Run the same network with Sparse

        # Convert to spmatrix
        weights_sparse = csr_matrix(weights)

        conn = LearningSparse(weights=weights_sparse,
                              tag_1=weights_sparse.copy(),
                              tag_2=weights_sparse.copy(),
                              learning_rule=learning_rule,
                              num_message_bits=8)

        sparse_net = create_learning_network(pre,
                                             conn,
                                             post,
                                             weights=weights_sparse)
        conn.run(condition=run_cond, run_cfg=run_cfg)

        weights_got_sparse = conn.weights.get()
        result_sparse = sparse_net[-1].data.get()
        conn.stop()

        np.testing.assert_array_equal(weights_got_dense,
                                      weights_got_sparse.toarray())

        np.testing.assert_array_almost_equal(result_sparse, result_dense)


class TestLearningSparseProcessModelFixed(unittest.TestCase):
    """Tests for LearningSparse class in fixed point precision. """

    def test_consistency_with_learning_dense_random_shape(self):
        """Tests if the results of LearningSparse and LearningDense
        are consistent. """

        simtime = 100
        shape = np.random.randint(1, 100, 2).tolist()
        weights = ((np.random.random(shape) - 0.5) * 20).astype(int)
        weights[abs(weights) < 2] = 2

        learning_rule = STDPLoihi(
            learning_rate=1,
            A_plus=1,
            A_minus=-1,
            tau_plus=10,
            tau_minus=10,
            t_epoch=2,
        )

        pre = (np.random.rand(shape[1], simtime) > 0.7).astype(int)
        post = (np.random.rand(shape[0], simtime) > 0.7).astype(int)

        conn = LearningDense(weights=weights,
                             tag_1=weights.copy(),
                             tag_2=weights.copy(),
                             learning_rule=learning_rule)
        dense_net = create_learning_network(pre, conn, post)

        run_cond = RunSteps(num_steps=simtime)
        run_cfg = Loihi2SimCfg(select_tag='fixed_pt')

        conn.run(condition=run_cond, run_cfg=run_cfg)
        weights_got_dense = conn.weights.get()
        conn.stop()

        # Run the same network with Sparse

        # Convert to spmatrix
        weights_sparse = csr_matrix(weights)

        conn = LearningSparse(weights=weights_sparse,
                              tag_1=weights_sparse.copy(),
                              tag_2=weights_sparse.copy(),
                              learning_rule=learning_rule)
        sparse_net = create_learning_network(pre, conn, post)
        conn.run(condition=run_cond, run_cfg=run_cfg)

        weights_got_sparse = conn.weights.get()
        conn.stop()

        np.testing.assert_array_equal(weights_got_dense,
                                      weights_got_sparse.toarray())

    def test_consistency_with_learning_dense_random_shape_dt(self):
        """Tests if the results of LearningSparse and LearningDense
        are consistent using dt in the learning rule. """

        simtime = 10
        shape = np.random.randint(1, 100, 2).tolist()
        weights = ((np.random.random(shape) - 0.5) * 20).astype(int)
        weights[abs(weights) < 2] = 2

        learning_rule = Loihi2FLearningRule(dt="x0 * y1 - y0 * x1",
                                            x1_tau=10,
                                            y1_tau=16,
                                            x1_impulse=16,
                                            y1_impulse=15,
                                            t_epoch=2)

        pre = (np.random.rand(shape[1], simtime) > 0.7).astype(int)
        post = (np.random.rand(shape[0], simtime) > 0.7).astype(int)

        conn = LearningDense(weights=weights,
                             tag_1=weights.copy(),
                             tag_2=weights.copy(),
                             learning_rule=learning_rule)
        dense_net = create_learning_network(pre, conn, post)

        run_cond = RunSteps(num_steps=simtime)
        run_cfg = Loihi2SimCfg(select_tag='fixed_pt')

        conn.run(condition=run_cond, run_cfg=run_cfg)
        tags_got_dense = conn.tag_1.get()
        conn.stop()

        # Run the same network with Sparse

        # Convert to spmatrix
        weights_sparse = csr_matrix(weights)

        conn = LearningSparse(weights=weights_sparse,
                              tag_1=weights_sparse.copy(),
                              tag_2=weights_sparse.copy(),
                              learning_rule=learning_rule)
        sparse_net = create_learning_network(pre, conn, post)
        conn.run(condition=run_cond, run_cfg=run_cfg)

        tags_got_sparse = conn.tag_1.get()
        conn.stop()

        np.testing.assert_array_equal(tags_got_dense,
                                      tags_got_sparse.toarray())

    def test_consistency_with_learning_dense_random_shape_dd(self):
        """Tests if the results of LearningSparse and LearningDense
        are consistent using dd in the learning rule. """

        simtime = 10
        shape = np.random.randint(1, 100, 2).tolist()
        weights = ((np.random.random(shape) - 0.5) * 20).astype(int)
        weights[abs(weights) < 2] = 2

        learning_rule = Loihi2FLearningRule(dd="x0 * y1 - y0 * x1",
                                            x1_tau=10,
                                            y1_tau=16,
                                            x1_impulse=16,
                                            y1_impulse=15,
                                            t_epoch=2)

        pre = (np.random.rand(shape[1], simtime) > 0.7).astype(int)
        post = (np.random.rand(shape[0], simtime) > 0.7).astype(int)

        conn = LearningDense(weights=weights,
                             tag_1=weights.copy(),
                             tag_2=weights.copy(),
                             learning_rule=learning_rule)
        dense_net = create_learning_network(pre, conn, post)

        run_cond = RunSteps(num_steps=simtime)
        run_cfg = Loihi2SimCfg(select_tag='fixed_pt')

        conn.run(condition=run_cond, run_cfg=run_cfg)
        tags_got_dense = conn.tag_2.get()
        conn.stop()

        # Run the same network with Sparse

        # Convert to spmatrix
        weights_sparse = csr_matrix(weights)

        conn = LearningSparse(weights=weights_sparse,
                              tag_1=weights_sparse.copy(),
                              tag_2=weights_sparse.copy(),
                              learning_rule=learning_rule)
        sparse_net = create_learning_network(pre, conn, post)
        conn.run(condition=run_cond, run_cfg=run_cfg)

        tags_got_sparse = conn.tag_2.get()
        conn.stop()

        np.testing.assert_array_equal(tags_got_dense,
                                      tags_got_sparse.toarray())

    def test_consistency_with_learning_dense_random_shape_graded(self):
        """Tests if the results of LearningSparse and LearningDense
        are consistent. """

        simtime = 10
        shape = np.random.randint(1, 100, 2).tolist()
        weights = ((np.random.random(shape) - 0.5) * 20).astype(int)
        weights[abs(weights) < 2] = 2

        learning_rule = STDPLoihi(
            learning_rate=1,
            A_plus=1,
            A_minus=-1,
            tau_plus=10,
            tau_minus=10,
            t_epoch=2,
        )

        pre = (np.random.rand(shape[1], simtime) > 0.7) * 3
        post = (np.random.rand(shape[0], simtime) > 0.7) * 2

        conn = LearningDense(weights=weights,
                             tag_1=weights.copy(),
                             tag_2=weights.copy(),
                             learning_rule=learning_rule,
                             num_message_bits=8)

        dense_net = create_learning_network(pre, conn, post, weights=weights)

        run_cond = RunSteps(num_steps=simtime)
        run_cfg = Loihi2SimCfg(select_tag='fixed_pt')

        conn.run(condition=run_cond, run_cfg=run_cfg)
        weights_got_dense = conn.weights.get()
        result_dense = dense_net[-1].data.get()
        conn.stop()

        # Run the same network with Sparse

        # Convert to spmatrix
        weights_sparse = csr_matrix(weights)

        conn = LearningSparse(weights=weights_sparse,
                              tag_1=weights_sparse.copy(),
                              tag_2=weights_sparse.copy(),
                              learning_rule=learning_rule,
                              num_message_bits=8)

        sparse_net = create_learning_network(pre,
                                             conn,
                                             post,
                                             weights=weights_sparse)
        conn.run(condition=run_cond, run_cfg=run_cfg)

        weights_got_sparse = conn.weights.get()
        result_sparse = sparse_net[-1].data.get()
        conn.stop()

        np.testing.assert_array_equal(weights_got_dense,
                                      weights_got_sparse.toarray())

        np.testing.assert_array_almost_equal(result_sparse, result_dense)


class VecSendandRecvProcess(AbstractProcess):
    """
    Process of a user-defined shape that sends an arbitrary vector

    Process also listens for incoming connections via InPort a_in. This
    allows the test Process to validate that network behavior won't deadlock
    in the presence of recurrent connections.

    Parameters
    ----------
    shape: tuple, shape of the process
    vec_to_send: np.ndarray, vector of spike values to send
    send_at_times: np.ndarray, vector bools. Send the `vec_to_send` at times
    when there is a True
    """

    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs.pop("shape", (1,))
        vec_to_send = kwargs.pop("vec_to_send")
        send_at_times = kwargs.pop("send_at_times")
        num_steps = kwargs.pop("num_steps", 1)
        self.shape = shape
        self.num_steps = num_steps
        self.vec_to_send = Var(shape=shape, init=vec_to_send)
        self.send_at_times = Var(shape=(num_steps,), init=send_at_times)
        self.s_out = OutPort(shape=shape)
        self.a_in = InPort(shape=shape)


class VecRecvProcess(AbstractProcess):
    """
    Process that receives arbitrary vectors

    Parameters
    ----------
    shape: tuple, shape of the process
    """

    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs.get("shape", (1,))
        self.shape = shape
        self.s_in = InPort(shape=(shape[1],))
        self.spk_data = Var(shape=shape, init=0)


@implements(proc=VecSendandRecvProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyVecSendModelFloat(PyLoihiProcessModel):
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    vec_to_send: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)
    send_at_times: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)

    def run_spk(self):
        """
        Send `spikes_to_send` if current time-step requires it
        """
        self.a_in.recv()

        if self.send_at_times[self.time_step - 1]:
            self.s_out.send(self.vec_to_send)
        else:
            self.s_out.send(np.zeros_like(self.vec_to_send))


@implements(proc=VecSendandRecvProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyVecSendModelFixed(PyLoihiProcessModel):
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=16)
    vec_to_send: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)
    send_at_times: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)

    def run_spk(self):
        """
        Send `spikes_to_send` if current time-step requires it
        """
        self.a_in.recv()

        if self.send_at_times[self.time_step - 1]:
            self.s_out.send(self.vec_to_send)
        else:
            self.s_out.send(np.zeros_like(self.vec_to_send))


@implements(proc=VecRecvProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PySpkRecvModelFloat(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    spk_data: np.ndarray = LavaPyType(np.ndarray, float)

    def run_spk(self):
        """Receive spikes and store in an internal variable"""
        spk_in = self.s_in.recv()
        self.spk_data[self.time_step - 1, :] = spk_in


@implements(proc=VecRecvProcess, protocol=LoihiProtocol)
@tag('fixed_pt')
class PySpkRecvModelFixed(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    spk_data: np.ndarray = LavaPyType(np.ndarray, int, precision=1)

    def run_spk(self):
        """Receive spikes and store in an internal variable"""
        spk_in = self.s_in.recv()
        self.spk_data[self.time_step - 1, :] = spk_in


class TestDelaySparseProcessModel(unittest.TestCase):
    """Tests for ProcessModels of Sparse with synaptic delay."""

    def test_matrix_weight_delay_expansion(self):
        """Tests if the weight-delay matrix is consistent between Dense
        and Sparse"""
        shape = (5, 4)
        weights = np.zeros(shape, dtype=float)
        weights[3, 3] = 1
        weights[1, 2] = 3
        weights[3, 1] = 1
        delays = np.zeros(shape, dtype=int)
        delays[3, 3] = 1
        delays[1, 2] = 3
        delays[3, 1] = 2
        max_delay = 10

        weights_sparse = csr_matrix(weights)
        delays_sparse = csr_matrix(delays)
        wgt_dly_dense = AbstractPyDelayDenseModel.get_delay_wgts_mat(weights,
                                                                     delays,
                                                                     max_delay)
        wgt_dly_sparse = APDSM.get_delay_wgts_mat(weights_sparse,
                                                  delays_sparse,
                                                  max_delay)
        self.assertTrue(np.all(wgt_dly_sparse == wgt_dly_dense))

    def test_float_pm_buffer_delay(self):
        """Tests floating point Sparse ProcessModel connectivity and temporal
        dynamics. All input 'neurons' from the VecSendandRcv fire
        once at time t=4, and only 1 connection weight
        in the Sparse Process is non-zero. The value of the delay matrix for
        this weight is 2. The non-zero connection should have an activation
        of 1 at timestep t=7.
        """
        shape = (3, 4)
        num_steps = 8
        # Set up external input to emulate every neuron spiking once on
        # timestep 4
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(shape=(shape[1],),
                                    num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up Sparse Process with a single non-zero connection weight at
        # entry [2, 2] of the connectivity matrix and a delay of 2 at entry
        # [2, 2] in the delay matrix.
        weights = np.zeros(shape, dtype=float)
        weights[2, 2] = 1
        delays = np.zeros(shape, dtype=int)
        delays[2, 2] = 2
        weights = csr_matrix(weights)
        delays = csr_matrix(delays)
        sparse = DelaySparse(weights=weights, delays=delays)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(sparse.s_in)
        sparse.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = Loihi2SimCfg(select_tag='floating_pt')
        sparse.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        sparse.stop()
        # Gold standard for the test
        # a_out will be equal to 1 at timestep 7, because the dendritic
        #  accumulators work on inputs from the previous timestep + 2.
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[6, 2] = 1.
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_float_pm_fan_in_delay(self):
        """
        Tests floating point Sparse ProcessModel dendritic accumulation
        behavior when the fan-in to a receiving neuron is greater than 1
        and synaptic delays are configured.
        """
        shape = (3, 4)
        num_steps = 10
        # Set up external input to emulate every neuron spiking once on
        # timestep 4
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up a Sparse Process where all input layer neurons project to a
        # single output layer neuron with varying delays.
        weights = np.zeros(shape, dtype=float)
        weights[2, :] = [2, -3, 4, -5]
        delays = np.zeros(shape, dtype=int)
        delays[2, :] = [1, 2, 2, 4]
        weights = csr_matrix(weights)
        delays = csr_matrix(delays)
        sparse = DelaySparse(weights=weights, delays=delays)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(sparse.s_in)
        sparse.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = Loihi2SimCfg(select_tag='floating_pt')
        sparse.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        sparse.stop()
        # Gold standard for the test
        # Expected behavior is that a_out corresponding to output
        # neuron 3 will be equal to 2 at timestep 6, 1=-3+4 at timestep 7
        # and -5 at timestep 9
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[5, 2] = 2
        expected_spk_data[6, 2] = 1
        expected_spk_data[8, 2] = -5
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_float_pm_fan_out_delay(self):
        """
        Tests floating point Sparse ProcessModel dendritic accumulation
        behavior when the fan-out of a projecting neuron is greater than 1
        and synaptic delays are configured.
        """
        shape = (3, 4)
        num_steps = 8
        # Set up external input to emulate every neuron spiking once on
        # timestep t=4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up a Sparse Process where a single input layer neuron
        # projects to all output layer neurons with a delay of 2 for
        # all synapses.
        weights = np.zeros(shape, dtype=float)
        weights[:, 2] = [3, 4, 5]
        delays = np.zeros(shape, dtype=int)
        delays = 2
        weights = csr_matrix(weights)
        sparse = DelaySparse(weights=weights, delays=delays)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(sparse.s_in)
        sparse.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = Loihi2SimCfg(select_tag='floating_pt')
        sparse.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        sparse.stop()
        # Gold standard for the test
        # Expected behavior is that a_out corresponding to output
        # neurons 1-3 will be equal to 3, 4, and 5, respectively, at
        # timestep 7.
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[6, :] = [3, 4, 5]
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_float_pm_fan_out_delay_2(self):
        """
        Tests floating point Sparse ProcessModel dendritic accumulation
        behavior when the fan-out of a projecting neuron is greater than 1
        and synaptic delays are configured.
        """
        shape = (3, 4)
        num_steps = 8
        # Set up external input to emulate every neuron spiking once on
        # timestep t=4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up a Sparse Process where a single input layer neuron projects
        # to all output layer neurons with varying delays.
        weights = np.zeros(shape, dtype=float)
        weights[:, 2] = [3, 4, 5]
        delays = np.zeros(shape, dtype=int)
        delays[:, 2] = [0, 1, 2]
        weights = csr_matrix(weights)
        delays = csr_matrix(delays)
        sparse = DelaySparse(weights=weights, delays=delays)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(sparse.s_in)
        sparse.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = Loihi2SimCfg(select_tag='floating_pt')
        sparse.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        sparse.stop()
        # Gold standard for the test
        # Expected behavior is that a_out corresponding to output
        # neurons 1-3 will be equal to 3, 4, and 5, respectively, at
        # timestep 5, 6 and 7, respectively.
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[4, 0] = 3
        expected_spk_data[5, 1] = 4
        expected_spk_data[6, 2] = 5
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_float_pm_recurrence_delays(self):
        """
         Tests that floating Sparse ProcessModel has non-blocking dynamics
         for recurrent connectivity architectures and synaptic delays are
         configured.
         """
        shape = (3, 3)
        num_steps = 8
        # Set up external input to emulate every neuron spiking once on
        # timestep 4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(True, (num_steps,))
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up Sparse Process with fully connected recurrent connectivity
        # architecture
        weights = np.ones(shape, dtype=float)
        delays = 2
        weights = csr_matrix(weights)
        sparse = DelaySparse(weights=weights, delays=delays)
        # Receive neuron spikes
        sps.s_out.connect(sparse.s_in)
        sparse.a_out.connect(sps.a_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = Loihi2SimCfg(select_tag='floating_pt')
        sparse.run(condition=rcnd, run_cfg=rcfg)
        sparse.stop()

    def test_bitacc_pm_fan_out_excitatory_delay(self):
        """
        Tests fixed-point Sparse ProcessModel dendritic accumulation
        behavior when the fan-out of a projecting neuron is greater than 1
        and all connections are excitatory (sign_mode = 2) and synaptic
        delays are configured.
        """
        shape = (3, 4)
        num_steps = 8
        # Set up external input to emulate every neuron spiking once on
        # timestep 4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up Sparse Process in which a single input neuron projects
        # to all output neurons.
        weights = np.zeros(shape, dtype=float)
        weights[:, 2] = [0.5, 300, 40]
        delays = np.zeros(shape, dtype=int)
        delays[:, 2] = [0, 1, 2]
        weights = csr_matrix(weights)
        delays = csr_matrix(delays)
        sparse = DelaySparse(weights=weights,
                             delays=delays,
                             sign_mode=SignMode.EXCITATORY)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(sparse.s_in)
        sparse.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = Loihi2SimCfg(select_tag='fixed_pt')
        sparse.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        sparse.stop()
        # Gold standard for the test
        # Expected behavior is that a_out corresponding to output
        # neurons 1-3 will be equal to 0, 255, and 40, respectively,
        # at timestep 5, 6 and 7, because a_out can only have integer values
        # between 0 and 255 and we have a delay of 0, 1, 2 on the synapses,
        # respectively.
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[4, 0] = 0
        expected_spk_data[5, 1] = 255
        expected_spk_data[6, 2] = 40
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_bitacc_pm_fan_out_mixed_sign_delay(self):
        """
        Tests fixed-point Sparse ProcessModel dendritic accumulation
        behavior when the fan-out of a projecting neuron is greater than 1
        and connections are both excitatory and inhibitory (sign_mode = 1).
        When using mixed sign weights and full 8 bit weight precision,
        a_out can take even values from -256 to 254. A delay of 2 for all
        synapses is configured.
        """
        shape = (3, 4)
        num_steps = 8
        # Set up external input to emulate every neuron spiking once on
        # timestep 4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up Sparse Process in which a single input neuron projects to
        # all output neurons with both excitatory and inhibitory weights.
        weights = np.zeros(shape, dtype=float)
        weights[:, 2] = [300, -300, 39]
        delays = np.zeros(shape, dtype=int)
        delays = 2
        weights = csr_matrix(weights)
        sparse = DelaySparse(weights=weights,
                             delays=delays,
                             sign_mode=SignMode.MIXED)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(sparse.s_in)
        sparse.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = Loihi2SimCfg(select_tag='fixed_pt')
        sparse.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        sparse.stop()
        # Gold standard for the test
        # Expected behavior is that a_out corresponding to output
        # neurons 1-3 will be equal to 254, -256, and 38, respectively,
        # at timestep 7, because a_out can only have even values between
        # -256 and 254 and a delay of 2 is configured.
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[6, :] = [254, -256, 38]
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_bitacc_pm_fan_out_weight_exp_delay(self):
        """
         Tests fixed-point Sparse ProcessModel dendritic accumulation
         behavior when the fan-out of a projecting neuron is greater than 1
         , connections are both excitatory and inhibitory (sign_mode = 1),
         and weight_exp = 1.
         When using mixed sign weights, full 8 bit weight precision,
         and weight_exp = 1, a_out can take even values from -512 to 508.
         As a result of setting weight_exp = 1, the expected a_out result
         is 2x that of the previous unit test. A delay of 0, 1, 2 is
         configured for respective synapses.
         """

        shape = (3, 4)
        num_steps = 8
        # Set up external input to emulate every neuron spiking once on
        # timestep 4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up Sparse Process in which all input neurons project to a
        # single output neuron with mixed sign connection weights.
        weights = np.zeros(shape, dtype=float)
        weights[:, 2] = [300, -300, 39]
        delays = np.zeros(shape, dtype=int)
        delays[:, 2] = [0, 1, 2]
        weights = csr_matrix(weights)
        delays = csr_matrix(delays)
        # Set weight_exp = 1. This affects weight scaling.
        sparse = DelaySparse(weights=weights, weight_exp=1, delays=delays)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(sparse.s_in)
        sparse.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = Loihi2SimCfg(select_tag='fixed_pt')
        sparse.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        sparse.stop()
        # Gold standard for the test
        # Expected behavior is that a_out corresponding to output
        # neurons 1-3 will be equal to 508, -512, and 76, respectively,
        # at timestep 5, 6, and 7, respectively, because a_out can only
        # have values between -512 and 508 such that a_out % 4 = 0.
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[4, 0] = 508
        expected_spk_data[5, 1] = -512
        expected_spk_data[6, 2] = 76
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_bitacc_pm_fan_out_weight_precision_delay(self):
        """
         Tests fixed-point Sparse ProcessModel dendritic accumulation
         behavior when the fan-out of a projecting neuron is greater than 1
         , connections are both excitatory and inhibitory (sign_mode = 1),
         and num_weight_bits = 7.
         When using mixed sign weights and 7 bit weight precision,
         a_out can take values from -256 to 252 such that a_out % 4 = 0.
         All synapses have a delay of 2 configured.
         """

        shape = (3, 4)
        num_steps = 8
        # Set up external input to emulate every neuron spiking once on
        # timestep 4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up Sparse Process in which all input neurons project to a
        # single output neuron with mixed sign connection weights.
        weights = np.zeros(shape, dtype=float)
        weights[:, 2] = [300, -300, 39]
        delays = np.zeros(shape, dtype=int)
        delays = 2
        weights = csr_matrix(weights)
        # Set num_weight_bits = 7. This affects weight scaling.
        sparse = DelaySparse(weights=weights,
                             num_weight_bits=7,
                             delays=delays)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(sparse.s_in)
        sparse.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = Loihi2SimCfg(select_tag='fixed_pt')
        sparse.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        sparse.stop()
        # Gold standard for the test
        # Expected behavior is that a_out corresponding to output
        # neurons 1-3 will be equal to 252, -256, and 36, respectively,
        # at timestep 7, because a_out can only have values between -256
        # and 252 such that a_out % 4 = 0.
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[6, :] = [252, -256, 36]
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_bitacc_pm_fan_in_mixed_sign_delay(self):
        """
        Tests fixed-point Sparse ProcessModel dendritic accumulation
        behavior when the fan-in of a receiving neuron is greater than 1
        and connections are both excitatory and inhibitory (sign_mode = 1).
        When using mixed sign weights and full 8 bit weight precision,
        a_out can take even values from -256 to 254. All synapses have a
        delay of 2 configured.
        """
        shape = (3, 4)
        num_steps = 8
        # Set up external input to emulate every neuron spiking once on
        # timestep 4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up Sparse Process in which all input layer neurons project to
        # a single output layer neuron with both excitatory and inhibitory
        # weights.
        weights = np.zeros(shape, dtype=float)
        weights[2, :] = [300, -300, 39, -0.4]
        delays = np.zeros(shape, dtype=int)
        delays = 2
        weights = csr_matrix(weights)
        sparse = DelaySparse(weights=weights,
                             sign_mode=SignMode.MIXED,
                             delays=delays)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(sparse.s_in)
        sparse.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = Loihi2SimCfg(select_tag='fixed_pt')
        sparse.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        sparse.stop()
        # Gold standard for the test
        # Expected behavior is that a_out corresponding to output
        # neuron 3 will be equal to 36=254-256+38-0 at timestep 7, because
        # weights can only have even values between -256 and 254.
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[6, 2] = 36
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_bitacc_pm_recurrence_delay(self):
        """
        Tests that bit accurate Sparse ProcessModel has non-blocking
        dynamics for recurrent connectivity architectures. All
        synapses have a delay of 2 configured.
        """
        shape = (3, 3)
        num_steps = 6
        # Set up external input to emulate every neuron spiking once on
        # timestep 4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(True, (num_steps,))
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up Sparse Process with fully connected recurrent connectivity
        # architecture.
        weights = np.ones(shape, dtype=float)
        delays = 2
        weights = csr_matrix(weights)
        sparse = DelaySparse(weights=weights, delays=delays)
        # Receive neuron spikes
        sps.s_out.connect(sparse.s_in)
        sparse.a_out.connect(sps.a_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = Loihi2SimCfg(select_tag='floating_pt')
        sparse.run(condition=rcnd, run_cfg=rcfg)
        sparse.stop()
