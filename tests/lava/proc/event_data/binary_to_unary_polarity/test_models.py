# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import unittest

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.proc.event_data.binary_to_unary_polarity.process \
    import BinaryToUnaryPolarity
from lava.proc.event_data.binary_to_unary_polarity.models \
    import PyBinaryToUnaryPolarityPM

from ..utils import SendSparse, RecvSparse


class TestProcessModelBinaryToUnaryPolarity(unittest.TestCase):
    def test_init(self):
        """Tests instantiation of the BinaryToUnary ProcessModel."""
        pm = PyBinaryToUnaryPolarityPM()
        self.assertIsInstance(pm, PyBinaryToUnaryPolarityPM)

    def test_binary_to_unary_polarity_encoding(self):
        """Tests whether the encoding from binary to unary works correctly."""
        data = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0])
        indices = np.array([1, 5, 4, 3, 3, 2, 0, 1, 0])

        expected_data = data
        expected_data[expected_data == 0] = 1

        expected_indices = indices

        send_sparse = SendSparse(shape=(10, ), data=data, indices=indices)
        binary_to_unary_encoder = BinaryToUnaryPolarity(shape=(10,))
        recv_sparse = RecvSparse(shape=(10, ))

        send_sparse.out_port.connect(binary_to_unary_encoder.in_port)
        binary_to_unary_encoder.out_port.connect(recv_sparse.in_port)

        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=1)

        send_sparse.run(condition=run_cnd, run_cfg=run_cfg)

        sent_and_received_data = \
            recv_sparse.data.get()[:expected_data.shape[0]]
        sent_and_received_indices = \
            recv_sparse.idx.get()[:expected_indices.shape[0]]

        send_sparse.stop()

        np.testing.assert_equal(sent_and_received_data,
                                expected_data)
        np.testing.assert_equal(sent_and_received_indices,
                                expected_indices)


if __name__ == '__main__':
    unittest.main()
