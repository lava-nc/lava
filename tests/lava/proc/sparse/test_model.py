# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
from scipy.sparse import csr_matrix

from lava.proc.sparse.process import Sparse
from lava.proc.dense.process import Dense
from lava.proc.io.source import RingBuffer as Source
from lava.proc.io.sink import RingBuffer as Sink
from lava.magma.core.run_configs import Loihi2SimCfg 
from lava.magma.core.run_conditions import RunSteps

def create_network(input_data, conn, weights):
    source = Source(data=input_data)
    sink = Sink(shape=(weights.shape[0], ),
                buffer=input_data.shape[1])

    source.s_out.connect(conn.s_in)
    conn.a_out.connect(sink.a_in)

    return source, conn, sink

class TestSparseProcessModelFloat(unittest.TestCase):
    """Tests for Sparse class in floating point precision. """

    # TODO add dedicated tests for get and set
    def test_consistency_with_dense_random_shape(self):
        """Tests if the results of Sparse and Dense are consistent. """
 
        simtime = 10
        shape = np.random.randint(1, 300, 2).tolist() 
        weights = (np.random.random(shape) - 0.5) * 2
        
        # sparsify
        weights[np.abs(weights) < 0.7] = 0
        
        inp = (np.random.rand(shape[1], simtime) > 0.7).astype(int)

        conn = Dense(weights=weights)
        dense_net = create_network(inp, conn, weights)

        run_cond = RunSteps(num_steps=simtime)
        run_cfg = Loihi2SimCfg(select_tag='floating_pt')

        conn.run(condition=run_cond, run_cfg=run_cfg)
        # weights_got_dense = conn.weights.get()
        result_dense = dense_net[2].data.get()
        conn.stop()

        # Run the same network with Sparse

        # convert to spmatrix
        weights_sparse = csr_matrix(weights)

        conn = Sparse(weights=weights_sparse)
        sparse_net = create_network(inp, conn, weights_sparse)
        conn.run(condition=run_cond, run_cfg=run_cfg)

        # weights_got_sparse = conn.weights.get()
        result_sparse = sparse_net[2].data.get()
        conn.stop()

        # print(weights_got_sparse, weights_got_dense)
        
        # np.testing.assert_array_equal(weights_got_dense, weights_got_sparse)
        np.testing.assert_array_almost_equal(result_sparse, result_dense)


    def test_consistency_with_dense_random_shape_graded(self):
        """Tests if the results of Sparse and Dense are consistent. """
 
        simtime = 10
        shape = np.random.randint(1, 300, 2).tolist() 
        weights = (np.random.random(shape) - 0.5) * 2
        
        # sparsify
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

        # convert to spmatrix
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

        # sparsify
        weights[np.abs(weights) < 0.7] = 0
        # convert to spmatrix
        weights_sparse = csr_matrix(weights)

        inp = (np.random.rand(shape[1], simtime) * 10).astype(int)

        run_cond = RunSteps(num_steps=simtime)
        run_cfg = Loihi2SimCfg(select_tag='floating_pt')

        conn = Sparse(weights=weights_sparse, num_message_bits=8)
        sparse_net = create_network(inp, conn, weights_sparse)
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

        # sparsify
        weights_init[np.abs(weights_init) < 0.7] = 0
        weights_init_sparse = csr_matrix(weights_init)

        weights_to_set_sparse = weights_init_sparse.copy()
        weights_to_set_sparse.data = (np.random.random(weights_to_set_sparse.data.shape) - 0.5) * 2

        inp = (np.random.rand(shape[1], simtime) * 10).astype(int)

        run_cond = RunSteps(num_steps=1)
        run_cfg = Loihi2SimCfg(select_tag='floating_pt')

        conn = Sparse(weights=weights_init_sparse, num_message_bits=8)
        sparse_net = create_network(inp, conn, weights_init_sparse)
        conn.run(condition=run_cond, run_cfg=run_cfg)

        conn.weights.set(weights_to_set_sparse)

        conn.run(condition=run_cond, run_cfg=run_cfg)

        weights_got_ts_2 = conn.weights.get()

        conn.stop()

        self.assertIsInstance(weights_got_ts_2, csr_matrix)
        np.testing.assert_array_equal(weights_got_ts_2.toarray(), weights_to_set_sparse.toarray())

    # def test_weights_get_set(self):
    #     """Tests the get and set method on weights."""
    #     simtime = 2
    #     shape = np.random.randint(1, 300, 2).tolist()
    #     weights_init = (np.random.random(shape) - 0.5) * 2
    #
    #     # sparsify
    #     weights_init[np.abs(weights_init) < 0.7] = 0
    #     weights_init_sparse = csr_matrix(weights_init)
    #
    #     weights_to_set_sparse = weights_init_sparse.copy()
    #     weights_to_set_sparse.data = (np.random.random(weights_to_set_sparse.data.shape) - 0.5) * 2
    #
    #     inp = (np.random.rand(shape[1], simtime) * 10).astype(int)
    #
    #     run_cond = RunSteps(num_steps=1)
    #     run_cfg = Loihi2SimCfg(select_tag='floating_pt')
    #
    #     conn = Sparse(weights=weights_init_sparse, num_message_bits=8)
    #     sparse_net = create_network(inp, conn, weights_init_sparse)
    #     conn.run(condition=run_cond, run_cfg=run_cfg)
    #
    #     weights_got_ts_1 = conn.weights.get()
    #
    #     conn.weights.set(weights_to_set_sparse)
    #
    #     conn.run(condition=run_cond, run_cfg=run_cfg)
    #
    #     weights_got_ts_2 = conn.weights.get()
    #
    #     conn.stop()
    #
    #     self.assertIsInstance(weights_got_ts_1, csr_matrix)
    #     self.assertIsInstance(weights_got_ts_2, csr_matrix)
    #     np.testing.assert_array_equal(weights_got_ts_1.toarray(), weights_init)
    #     np.testing.assert_array_equal(weights_got_ts_2.toarray(), weights_to_set_sparse.toarray())

class TestSparseProcessModelFixed(unittest.TestCase):
    """Tests for Sparse class in fixed point precision. """

    def test_consitency_with_dense_random_shape(self):
        """Tests if the results of Sparse and Dense are consistent. """
 
        simtime = 10
        shape = np.random.randint(1, 300, 2).tolist() 
        weights = (np.random.random(shape) - 0.5) * 2
        
        # sparsify
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

        # convert to spmatrix
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
        
        # sparsify
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

        # convert to spmatrix
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

        # sparsify
        weights_init[np.abs(weights_init) < 0.7] = 0
        weights_init *= 20
        weights_init = weights_init.astype(int)
        weights_sparse = csr_matrix(weights_init)

        inp = (np.random.rand(shape[1], simtime) * 10).astype(int)

        run_cond = RunSteps(num_steps=simtime)
        run_cfg = Loihi2SimCfg(select_tag='floating_pt')

        conn = Sparse(weights=weights_sparse, num_message_bits=8)
        sparse_net = create_network(inp, conn, weights_sparse)
        conn.run(condition=run_cond, run_cfg=run_cfg)

        weights_got = conn.weights.get()
        conn.stop()

        self.assertIsInstance(weights_got, csr_matrix)
        np.testing.assert_array_equal(weights_got.toarray(), weights_init)

    def test_weights_set(self):
        """Tests the set method on weights."""
        simtime = 2
        shape = np.random.randint(1, 300, 2).tolist()
        weights_init = (np.random.random(shape) - 0.5) * 2

        # sparsify
        weights_init[np.abs(weights_init) < 0.7] = 0
        weights_init *= 20
        weights_init = weights_init.astype(int)
        weights_init_sparse = csr_matrix(weights_init)

        weights_to_set_sparse = weights_init_sparse.copy()
        weights_to_set_sparse.data = (np.random.random(weights_to_set_sparse.data.shape) - 0.5) * 2
        weights_to_set_sparse *= 20
        weights_to_set_sparse = weights_to_set_sparse.astype(int)

        inp = (np.random.rand(shape[1], simtime) * 10).astype(int)

        run_cond = RunSteps(num_steps=1)
        run_cfg = Loihi2SimCfg(select_tag='floating_pt')

        conn = Sparse(weights=weights_init_sparse, num_message_bits=8)
        sparse_net = create_network(inp, conn, weights_init_sparse)
        conn.run(condition=run_cond, run_cfg=run_cfg)

        conn.weights.set(weights_to_set_sparse)

        conn.run(condition=run_cond, run_cfg=run_cfg)

        weights_got_ts_2 = conn.weights.get()

        conn.stop()

        self.assertIsInstance(weights_got_ts_2, csr_matrix)
        np.testing.assert_array_equal(weights_got_ts_2.toarray(), weights_to_set_sparse.toarray())

    # def test_weights_get_set(self):
    #     """Tests the get and set method on weights."""
    #     simtime = 2
    #     shape = np.random.randint(3, 4, 2).tolist()
    #     weights_init = (np.random.random(shape) - 0.5) * 2
    #
    #     # sparsify
    #     weights_init[np.abs(weights_init) < 0.7] = 0
    #     weights_init *= 20
    #     weights_init = weights_init.astype(int)
    #     weights_init_sparse = csr_matrix(weights_init)
    #
    #     weights_to_set_sparse = weights_init_sparse.copy()
    #     weights_to_set_sparse.data = (np.random.random(weights_to_set_sparse.data.shape) - 0.5) * 2
    #     weights_to_set_sparse *= 20
    #     weights_to_set_sparse = weights_to_set_sparse.astype(int)
    #
    #     inp = (np.random.rand(shape[1], simtime) * 10).astype(int)
    #
    #     run_cond = RunSteps(num_steps=1)
    #     run_cfg = Loihi2SimCfg(select_tag='floating_pt')
    #
    #     conn = Sparse(weights=weights_init_sparse, num_message_bits=8)
    #     sparse_net = create_network(inp, conn, weights_init_sparse)
    #     conn.run(condition=run_cond, run_cfg=run_cfg)
    #
    #     weights_got_ts_1 = conn.weights.get()
    #
    #     conn.weights.set(weights_to_set_sparse)
    #
    #     conn.run(condition=run_cond, run_cfg=run_cfg)
    #
    #     weights_got_ts_2 = conn.weights.get()
    #
    #     conn.stop()
    #
    #     self.assertIsInstance(weights_got_ts_1, csr_matrix)
    #     self.assertIsInstance(weights_got_ts_2, csr_matrix)
    #     np.testing.assert_array_equal(weights_got_ts_1.toarray(), weights_init)
    #     np.testing.assert_array_equal(weights_got_ts_2.toarray(), weights_to_set_sparse.toarray())