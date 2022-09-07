
# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
#
# Copyright © 2021-2022 Intel Corporation.
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
import unittest
from lava.proc.scif.process import SCIF


class TestSCIFProcess(unittest.TestCase):
    """Tests for SigmaDelta class"""
    def test_init(self) -> None:
        """Tests instantiation of SigmaDelta"""
        scif = SCIF(shape=(10,),
                    bias=2,
                    theta=8,
                    neg_tau_ref=-10,
                    enable_noise=1)

        self.assertEqual(scif.shape, (10,))
        self.assertEqual(scif.bias.init, 2)
        self.assertEqual(scif.theta.init, 8)
        self.assertEqual(scif.neg_tau_ref.init, -10)
        self.assertEqual(scif.enable_noise.init, 1)
