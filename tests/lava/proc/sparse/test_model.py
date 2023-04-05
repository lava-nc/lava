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

def create_network(input_data, conn_type, weights):
    source = Source(data=input_data)
    conn = conn_type(weights=weights)
    sink = Sink(shape=(weights.shape[0], ),
                buffer=input_data.shape[1])

    source.s_out.connect(conn.s_in)
    conn.a_out.connect(sink.a_in)

    return source, conn, sink

class TestSparseProcessModelFloat(unittest.TestCase):
    """Tests for Sparse class in floating point precision. """

    def test_consitency_with_dense_random_shape(self):
        """Tests if the results of Sparse and Dense are consistent. """
 
        simtime = 10
        shape = np.random.randint(1, 300, 2).tolist() 
        weights = (np.random.random(shape) - 0.5) * 2
        
        # sparsify
        weights[np.abs(weights) < 0.7] = 0
        
        inp = np.random.rand(shape[1], simtime) 

        dense_net = create_network(inp, Dense, weights)

        run_cond = RunSteps(num_steps=simtime)
        run_cfg = Loihi2SimCfg(select_tag='floating_pt')

        dense_net[0].run(condition=run_cond, run_cfg=run_cfg)
        result_dense = dense_net[2].data.get()
        dense_net[0].stop()

        # Run the same network with Sparse

        # convert to spmatrix
        weights_sparse = csr_matrix(weights)

        sparse_net = create_network(inp, Sparse, weights_sparse)
        sparse_net[0].run(condition=run_cond, run_cfg=run_cfg)

        result_sparse = sparse_net[2].data.get()
        sparse_net[0].stop()

        np.testing.assert_array_almost_equal(result_sparse, result_dense)


    # TODO add test for num_message_bits > 0



