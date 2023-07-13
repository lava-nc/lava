# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
#
# Copyright © 2021-2023 Intel Corporation.
#
# This software and the related documents are Intel copyrighted
# materials, and your use of them is governed by the express
# license under which they were provided to you (License). Unless
# the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit  this software or the
# related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with
# no express or implied warranties, other than those that are
# expressly stated in the License.
# See: https://spdx.org/licenses/


import unittest
import numpy as np
from scipy.sparse import csr_matrix, find

from lava.proc.graded.gradedvec import GradedVec
from lava.proc.dense.process import Dense
from lava.proc.sparse.process import Sparse
from lava.proc import io
from lava.proc import embedded_io as eio

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg

class TestGradedVecProc(unittest.TestCase):
    #@unittest.skipUnless(run_loihi2_tests, "Loihi2 unavailable.")
    #@unittest.skip("Skipping for merge.")
    def test_gradedvec_dot_dense(self):
        num_steps = 10
        v_thresh = 1

        weights1 = np.zeros((10,1)) 
        weights1[:, 0] = (np.arange(10)-5) * 0.2
        
        inp_data = np.zeros((weights1.shape[1], num_steps))
        inp_data[:, 2] = 1000
        inp_data[:, 6] = 20000
        
        weight_exp = 7
        weights1 *= 2**weight_exp
        weights1 = weights1.astype('int')
        
        dense1 = Dense(weights=weights1, num_message_bits=24,
                       weight_exp=-weight_exp)
        vec1 = GradedVec(shape=(weights1.shape[0],), 
                     vth=v_thresh)
        
        generator = io.source.RingBuffer(data=inp_data)
        logger = io.sink.RingBuffer(shape=(weights1.shape[0],), 
                                    buffer=num_steps)
        
        generator.s_out.connect(dense1.s_in)
        dense1.a_out.connect(vec1.a_in)
        vec1.s_out.connect(logger.a_in)
        
        vec1.run(condition=RunSteps(num_steps=num_steps),
                  run_cfg=Loihi2SimCfg(select_tag='fixed_pt'))
        out_data = logger.data.get().astype('int')
        vec1.stop()
        
        ww = np.floor(weights1/2)*2
        expected_out = np.floor((ww @ inp_data) /  2**weight_exp)

        self.assertTrue(np.all(out_data[:, (3,7)] == expected_out[:,(2,6)]))
        
    def test_thvec_dot_sparse(self):
        num_steps = 10
        v_thresh = 1

        weights1 = np.zeros((10,1)) 
        weights1[:, 0] = (np.arange(10)-5) * 0.2
        
        inp_data = np.zeros((weights1.shape[1], num_steps))
        inp_data[:, 2] = 1000
        inp_data[:, 6] = 20000
        
        weight_exp = 7
        weights1 *= 2**weight_exp
        weights1 = weights1.astype('int')
        
        sparse1 = Sparse(weights=csr_matrix(weights1), 
                        num_message_bits=24,
                        weight_exp=-weight_exp)
        vec1 = GradedVec(shape=(weights1.shape[0],), 
                     vth=v_thresh)
        
        generator = io.source.RingBuffer(data=inp_data)
        logger = io.sink.RingBuffer(shape=(weights1.shape[0],), 
                                    buffer=num_steps)
        
        generator.s_out.connect(sparse1.s_in)
        sparse1.a_out.connect(vec1.a_in)
        vec1.s_out.connect(logger.a_in)
        
        vec1.run(condition=RunSteps(num_steps=num_steps),
                  run_cfg=Loihi2SimCfg(select_tag='fixed_pt'))
        out_data = logger.data.get().astype('int')
        vec1.stop()
        
        ww = np.floor(weights1/2)*2
        expected_out = np.floor((ww @ inp_data) /  2**weight_exp)

        self.assertTrue(np.all(out_data[:, (3,7)] == expected_out[:,(2,6)]))
        
        
if __name__ == '__main__':
    unittest.main()
    