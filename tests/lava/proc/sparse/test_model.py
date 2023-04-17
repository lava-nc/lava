# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
from scipy.sparse import csr_matrix

from lava.proc.sparse.process import Sparse
from lava.proc.dense.process import Dense
from lava.proc.sparse.process import Sparse, DelaySparse
from lava.proc.sparse.models import AbstractPyDelaySparseModel
from lava.proc.io.source import RingBuffer as Source
from lava.proc.io.sink import RingBuffer as Sink
from lava.magma.core.run_configs import Loihi2SimCfg 
import unittest
import numpy as np
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort, PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort, InPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.run_configs import RunConfig
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.dense.models import AbstractPyDelayDenseModel

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

        weights_to_set_sparse = conn.weights.init.copy()
        weights_to_set_sparse.data = np.random.permutation(weights_to_set_sparse.data)

        conn.weights.set(weights_to_set_sparse)

        conn.run(condition=run_cond, run_cfg=run_cfg)

        weights_got_ts_2 = conn.weights.get()

        conn.stop()

        self.assertIsInstance(weights_got_ts_2, csr_matrix)
        np.testing.assert_array_equal(weights_got_ts_2.toarray(), weights_to_set_sparse.toarray())


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

        inp = (np.random.rand(shape[1], simtime) * 10).astype(int)

        run_cond = RunSteps(num_steps=1)
        run_cfg = Loihi2SimCfg(select_tag='floating_pt')

        conn = Sparse(weights=weights_init_sparse, num_message_bits=8)
        sparse_net = create_network(inp, conn, weights_init_sparse)
        conn.run(condition=run_cond, run_cfg=run_cfg)

        weights_to_set_sparse = conn.weights.init.copy()
        weights_to_set_sparse.data = np.random.permutation(weights_to_set_sparse.data)

        conn.weights.set(weights_to_set_sparse)

        conn.run(condition=run_cond, run_cfg=run_cfg)

        weights_got_ts_2 = conn.weights.get()

        conn.stop()

        self.assertIsInstance(weights_got_ts_2, csr_matrix)
        np.testing.assert_array_equal(weights_got_ts_2.toarray(), weights_to_set_sparse.toarray())


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
        self.a_in = InPort(shape=shape)  # enables recurrence test


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
        self.spk_data = Var(shape=shape, init=0)  # This Var expands with time


@implements(proc=VecSendandRecvProcess, protocol=LoihiProtocol)
@requires(CPU)
# need the following tag to discover the ProcessModel using DenseRunConfig
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

@implements(proc=VecRecvProcess, protocol=LoihiProtocol)
@requires(CPU)
# need the following tag to discover the ProcessModel using DenseRunConfig
@tag('floating_pt')
class PySpkRecvModelFloat(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    spk_data: np.ndarray = LavaPyType(np.ndarray, float)

    def run_spk(self):
        """Receive spikes and store in an internal variable"""
        spk_in = self.s_in.recv()
        self.spk_data[self.time_step - 1, :] = spk_in

class TestDelayDenseProcessModel(unittest.TestCase):
        """Tests for ProcessModels of Dense with synaptic delay."""

        def test_matrix_weight_delay_expansion(self):
            """"""
            shape = (5, 4)
            weights = np.zeros(shape, dtype=float)
            weights[3, 3] = 1
            weights[1,2] = 3
            weights[3,1] = 1
            delays = np.zeros(shape, dtype=int)
            delays[3, 3] = 1
            delays[1,2] = 3
            delays[3,1] = 2

            weights_sparse = csr_matrix(weights)
            delays_sparse = csr_matrix(delays)
            wgt_dly_dense = AbstractPyDelayDenseModel.get_del_wgts(weights, delays)
            wgt_dly_sparse = AbstractPyDelaySparseModel.get_del_wgts(weights_sparse, delays_sparse)
            self.assertTrue(np.all(wgt_dly_sparse==wgt_dly_dense))

        def test_float_pm_buffer_delay(self):
            """Tests floating point Dense ProcessModel connectivity and temporal
            dynamics. All input 'neurons' from the VecSendandRcv fire
            once at time t=4, and only 1 connection weight
            in the Dense Process is non-zero. The value of the delay matrix for
            this weight is 2. The non-zero connection should have an activation of
            1 at timestep t=7.
            """
            shape = (3, 4)
            num_steps = 8
            # Set up external input to emulate every neuron spiking once on
            # timestep 4
            vec_to_send = np.ones((shape[1],), dtype=float)
            send_at_times = np.repeat(False, (num_steps,))
            send_at_times[3] = True
            sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                        vec_to_send=vec_to_send,
                                        send_at_times=send_at_times)
            # Set up Dense Process with a single non-zero connection weight at
            # entry [2, 2] of the connectivity matrix and a delay of 2 at entry
            # [2, 2] in the delay matrix.
            weights = np.zeros(shape, dtype=float)
            weights[2, 2] = 1
            delays = np.zeros(shape, dtype=int)
            delays[2, 2] = 2
            print(weights)
            print(delays)
            weights = csr_matrix(weights)
            delays = csr_matrix(delays)
            dense = DelaySparse(weights=weights, delays=delays)
            # Receive neuron spikes
            spr = VecRecvProcess(shape=(num_steps, shape[0]))
            sps.s_out.connect(dense.s_in)
            dense.a_out.connect(spr.s_in)
            # Configure execution and run
            rcnd = RunSteps(num_steps=num_steps)
            rcfg = Loihi2SimCfg(select_tag='floating_pt')
            dense.run(condition=rcnd, run_cfg=rcfg)
            # Gather spike data and stop
            spk_data_through_run = spr.spk_data.get()
            dense.stop()
            # Gold standard for the test
            # a_out will be equal to 1 at timestep 7, because the dendritic
            #  accumulators work on inputs from the previous timestep + 2.
            expected_spk_data = np.zeros((num_steps, shape[0]))
            expected_spk_data[6, 2] = 1.
            self.assertTrue(np.all(expected_spk_data == spk_data_through_run))
