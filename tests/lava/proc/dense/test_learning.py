# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
import typing as ty

from lava.magma.core.learning.constants import GradedSpikeCfg
from lava.magma.core.learning.learning_rule import Loihi2FLearningRule
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.proc.dense.process import LearningDense
from lava.proc.monitor.process import Monitor
from lava.proc.io.source import RingBuffer, PySendModelFixed, PySendModelFloat


class TestLearningSimGradedSpikeFloatingPoint(unittest.TestCase):
    """Known value test suite. Tests have been run once and validated.
    Resulting values are stored as 'expected' values.
    Any deviation from these expected values in the future would be a symptom
    of breaking changes"""

    @staticmethod
    def create_network_single_synapse(num_steps: int,
                                      weights_init: np.ndarray,
                                      learning_rule_cnd: str,
                                      graded_spike_cfg: GradedSpikeCfg) \
            -> ty.Tuple[RingBuffer, LearningDense, RingBuffer]:
        """Create a network of RingBuffer -> LearningDense -> RingBuffer.

        Parameters
        ----------
        num_steps : int
            Number of simulation time steps.
        weights_init : ndarray
            Initial weights matrix for the LearningDense.
        learning_rule_cnd : str
            String specifying which learning rule condition to use.
        graded_spike_cfg : GradedSpikeCfg
            GradedSpikeCfg to use for the LearningDense.

        Returns
        ----------
        pattern_pre : RingBuffer
            Pre-synaptic RingBuffer Process.
        learning_dense : LearningDense
            LearningDense Process.
        pattern_post : RingBuffer
            Post-synaptic RingBuffer Process.
        """

        if learning_rule_cnd == "x" or learning_rule_cnd == "y":
            scaling_exp = -1
        elif learning_rule_cnd == "u":
            scaling_exp = -2
        else:
            scaling_exp = 0

        dw = f"2^{scaling_exp} * {learning_rule_cnd}0 * x1"

        x1_impulse = \
            16 if graded_spike_cfg == GradedSpikeCfg.USE_REGULAR_IMPULSE else 0

        learning_rule = Loihi2FLearningRule(dw=dw,
                                            x1_impulse=x1_impulse, x1_tau=12,
                                            x2_impulse=24, x2_tau=5,
                                            y1_impulse=16, y1_tau=10,
                                            y2_impulse=24, y2_tau=5,
                                            t_epoch=4)

        spike_raster_pre = np.zeros((1, num_steps))
        spike_raster_pre[0, 4] = 51
        spike_raster_pre[0, 11] = 100
        spike_raster_pre[0, 17] = 60
        spike_raster_pre[0, 21] = 246
        spike_raster_pre[0, 38] = 30
        spike_raster_pre[0, 55] = 22
        spike_raster_post = np.zeros((1, num_steps))
        spike_raster_post[0, 3] = 1
        spike_raster_post[0, 21] = 1
        spike_raster_post[0, 39] = 1
        spike_raster_post[0, 57] = 1

        pattern_pre = RingBuffer(data=spike_raster_pre.astype(int))
        learning_dense = \
            LearningDense(weights=weights_init,
                          learning_rule=learning_rule,
                          name="learning_dense",
                          num_message_bits=8,
                          graded_spike_cfg=graded_spike_cfg)
        pattern_post = RingBuffer(data=spike_raster_post.astype(int))

        pattern_pre.s_out.connect(learning_dense.s_in)
        pattern_post.s_out.connect(learning_dense.s_in_bap)

        return pattern_pre, learning_dense, pattern_post

    expected_x1_data_default = \
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.4608125, 12.4608125, 12.4608125,
         12.4608125, 24.9285623, 24.9285623, 24.9285623, 24.9285623, 17.8620954,
         17.8620954, 17.8620954, 17.8620954, 26.3424583, 26.3424583, 26.3424583,
         26.3424583, 32.4189037, 32.4189037, 32.4189037, 32.4189037, 23.2291596,
         23.2291596, 23.2291596, 23.2291596, 16.6444202, 16.6444202, 16.6444202,
         16.6444202, 11.9262482, 11.9262482, 11.9262482, 11.9262482, 23.2662409,
         23.2662409, 23.2662409, 23.2662409, 16.6709901, 16.6709901, 16.6709901,
         16.6709901, 11.9452864, 11.9452864, 11.9452864, 11.9452864, 8.5591717,
         8.5591717, 8.5591717, 8.5591717, 22.1329145, 22.1329145, 22.1329145,
         22.1329145, 15.8589262, 15.8589262, 15.8589262, 15.8589262]
    expected_x1_data_overwrite = \
        [0.0, 0.0, 0.0, 0.0, 25.5, 25.5, 25.5, 18.2715484, 18.2715484,
         18.2715484, 18.2715484, 35.8265655, 35.8265655, 35.8265655, 35.8265655,
         25.670856, 25.670856, 30.0, 30.0, 21.4959393, 21.4959393, 123.0, 123.0,
         88.1333512, 88.1333512, 88.1333512, 88.1333512, 63.1503056, 63.1503056,
         63.1503056, 63.1503056, 45.2491713, 45.2491713, 45.2491713, 45.2491713,
         32.422448, 32.422448, 32.422448, 15.0, 10.7479697, 10.7479697,
         10.7479697, 10.7479697, 7.7012568, 7.7012568, 7.7012568, 7.7012568,
         5.5181916, 5.5181916, 5.5181916, 5.5181916, 3.9539571, 3.9539571,
         3.9539571, 3.9539571, 7.8818444, 7.8818444, 7.8818444, 7.8818444,
         5.6475883, 5.6475883, 5.6475883, 5.6475883]
    expected_x1_data_sat = \
        [0.0, 0.0, 0.0, 0.0, 25.5, 25.5, 25.5, 18.2715484, 18.2715484,
         18.2715484, 18.2715484, 48.9187021, 48.9187021, 48.9187021, 48.9187021,
         35.0517817, 35.0517817, 65.0517817, 65.0517817, 46.6116384, 46.6116384,
         169.6116384, 169.6116384, 121.5320495, 121.5320495, 121.5320495,
         121.5320495, 87.0815187, 87.0815187, 87.0815187, 87.0815187,
         62.3966347, 62.3966347, 62.3966347, 62.3966347, 44.7091425, 44.7091425,
         44.7091425, 59.7091425, 42.7834701, 42.7834701, 42.7834701, 42.7834701,
         30.6556959, 30.6556959, 30.6556959, 30.6556959, 21.965766, 21.965766,
         21.965766, 21.965766, 15.7391591, 15.7391591, 15.7391591, 15.7391591,
         19.1594447, 19.1594447, 19.1594447, 19.1594447, 13.728342, 13.728342,
         13.728342, 13.728342]
    expected_x1_data_no_sat = \
        [0.0, 0.0, 0.0, 0.0, 25.5, 25.5, 25.5, 18.2715484, 18.2715484,
         18.2715484, 18.2715484, 48.9187021, 48.9187021, 48.9187021, 48.9187021,
         35.0517817, 35.0517817, 65.0517817, 65.0517817, 46.6116384, 46.6116384,
         169.6116384, 169.6116384, 121.5320495, 121.5320495, 121.5320495,
         121.5320495, 87.0815187, 87.0815187, 87.0815187, 87.0815187,
         62.3966347, 62.3966347, 62.3966347, 62.3966347, 44.7091425, 44.7091425,
         44.7091425, 59.7091425, 42.7834701, 42.7834701, 42.7834701, 42.7834701,
         30.6556959, 30.6556959, 30.6556959, 30.6556959, 21.965766, 21.965766,
         21.965766, 21.965766, 15.7391591, 15.7391591, 15.7391591, 15.7391591,
         19.1594447, 19.1594447, 19.1594447, 19.1594447, 13.728342, 13.728342,
         13.728342, 13.728342]

    expected_x2_data = \
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.1714793, 13.1714793, 13.1714793,
         13.1714793, 29.9183271, 29.9183271, 29.9183271, 29.9183271, 13.4431709,
         13.4431709, 13.4431709, 13.4431709, 22.1280872, 22.1280872, 22.1280872,
         22.1280872, 26.0304716, 26.0304716, 26.0304716, 26.0304716, 11.6962448,
         11.6962448, 11.6962448, 11.6962448, 5.2554616, 5.2554616, 5.2554616,
         5.2554616, 2.3614311, 2.3614311, 2.3614311, 2.3614311, 20.7105975,
         20.7105975, 20.7105975, 20.7105975, 9.3058713, 9.3058713, 9.3058713,
         9.3058713, 4.1813975, 4.1813975, 4.1813975, 4.1813975, 1.878823,
         1.878823, 1.878823, 1.878823, 24.8442096, 24.8442096, 24.8442096,
         24.8442096, 11.163223, 11.163223, 11.163223, 11.163223]
    expected_x2_data_no_sat = \
        expected_x2_data

    expected_wgt_data_default_x = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 18.0, 18.0, 18.0, 18.0,
         30.4642812, 30.4642812, 30.4642812, 30.4642812, 30.4642812, 30.4642812,
         30.4642812, 30.4642812, 46.0242498, 46.0242498, 46.0242498, 46.0242498,
         65.1734546, 65.1734546, 65.1734546, 65.1734546, 65.1734546, 65.1734546,
         65.1734546, 65.1734546, 65.1734546, 65.1734546, 65.1734546, 65.1734546,
         65.1734546, 65.1734546, 65.1734546, 65.1734546, 77.8175403, 77.8175403,
         77.8175403, 77.8175403, 77.8175403, 77.8175403, 77.8175403, 77.8175403,
         77.8175403, 77.8175403, 77.8175403, 77.8175403, 77.8175403, 77.8175403,
         77.8175403, 77.8175403, 88.8839976, 88.8839976, 88.8839976, 88.8839976,
         88.8839976, 88.8839976, 88.8839976, 88.8839976]
    expected_wgt_data_default_y = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         29.1492047, 29.1492047, 29.1492047, 29.1492047, 29.1492047, 29.1492047,
         29.1492047, 29.1492047, 29.1492047, 29.1492047, 29.1492047, 29.1492047,
         29.1492047, 29.1492047, 29.1492047, 29.1492047, 40.7823252, 40.7823252,
         40.7823252, 40.7823252, 40.7823252, 40.7823252, 40.7823252, 40.7823252,
         40.7823252, 40.7823252, 40.7823252, 40.7823252, 40.7823252, 40.7823252,
         40.7823252, 40.7823252, 40.7823252, 40.7823252, 40.7823252, 40.7823252,
         50.149879, 50.149879, 50.149879, 50.149879]
    expected_wgt_data_default_u = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 13.1152031, 13.1152031,
         13.1152031, 13.1152031, 19.3473437, 19.3473437, 19.3473437, 19.3473437,
         23.8128676, 23.8128676, 23.8128676, 23.8128676, 30.3984821, 30.3984821,
         30.3984821, 30.3984821, 38.5032081, 38.5032081, 38.5032081, 38.5032081,
         44.310498, 44.310498, 44.310498, 44.310498, 48.471603, 48.471603,
         48.471603, 48.471603, 51.4531651, 51.4531651, 51.4531651, 51.4531651,
         57.2697253, 57.2697253, 57.2697253, 57.2697253, 61.4374728, 61.4374728,
         61.4374728, 61.4374728, 64.4237944, 64.4237944, 64.4237944, 64.4237944,
         66.5635873, 66.5635873, 66.5635873, 66.5635873, 72.0968159, 72.0968159,
         72.0968159, 72.0968159, 76.0615475, 76.0615475, 76.0615475, 76.0615475]
    expected_wgt_data_overwrite_x = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 21.7305663, 21.7305663,
         21.7305663, 21.7305663, 39.6438491, 39.6438491, 39.6438491, 39.6438491,
         39.6438491, 39.6438491, 39.6438491, 39.6438491, 52.3410749, 52.3410749,
         52.3410749, 52.3410749, 104.399701, 104.399701, 104.399701, 104.399701,
         104.399701, 104.399701, 104.399701, 104.399701, 104.399701, 104.399701,
         104.399701, 104.399701, 104.399701, 104.399701, 104.399701, 104.399701,
         110.2407069, 110.2407069, 110.2407069, 110.2407069, 110.2407069,
         110.2407069, 110.2407069, 110.2407069, 110.2407069, 110.2407069,
         110.2407069, 110.2407069, 110.2407069, 110.2407069, 110.2407069,
         110.2407069, 114.1816291, 114.1816291, 114.1816291, 114.1816291,
         114.1816291, 114.1816291, 114.1816291, 114.1816291]
    expected_wgt_data_overwrite_y = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         62.0586261, 62.0586261, 62.0586261, 62.0586261, 62.0586261, 62.0586261,
         62.0586261, 62.0586261, 62.0586261, 62.0586261, 62.0586261, 62.0586261,
         62.0586261, 62.0586261, 62.0586261, 62.0586261, 67.4326109, 67.4326109,
         67.4326109, 67.4326109, 67.4326109, 67.4326109, 67.4326109, 67.4326109,
         67.4326109, 67.4326109, 67.4326109, 67.4326109, 67.4326109, 67.4326109,
         67.4326109, 67.4326109, 67.4326109, 67.4326109, 67.4326109, 67.4326109,
         70.7685295, 70.7685295, 70.7685295, 70.7685295]
    expected_wgt_data_overwrite_u = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 14.5678871, 14.5678871,
         14.5678871, 14.5678871, 23.5245285, 23.5245285, 23.5245285, 23.5245285,
         29.9422425, 29.9422425, 29.9422425, 29.9422425, 35.3162273, 35.3162273,
         35.3162273, 35.3162273, 57.3495651, 57.3495651, 57.3495651, 57.3495651,
         73.1371415, 73.1371415, 73.1371415, 73.1371415, 84.4494343, 84.4494343,
         84.4494343, 84.4494343, 92.5550463, 92.5550463, 92.5550463, 92.5550463,
         95.2420387, 95.2420387, 95.2420387, 95.2420387, 97.1673529, 97.1673529,
         97.1673529, 97.1673529, 98.5469008, 98.5469008, 98.5469008, 98.5469008,
         99.5353901, 99.5353901, 99.5353901, 99.5353901, 101.5058512,
         101.5058512, 101.5058512, 101.5058512, 102.9177483, 102.9177483,
         102.9177483, 102.9177483]
    expected_wgt_data_sat_x = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 21.7305663, 21.7305663,
         21.7305663, 21.7305663, 46.1899173, 46.1899173, 46.1899173, 46.1899173,
         46.1899173, 46.1899173, 46.1899173, 46.1899173, 73.7224895, 73.7224895,
         73.7224895, 73.7224895, 145.5090656, 145.5090656, 145.5090656,
         145.5090656, 145.5090656, 145.5090656, 145.5090656, 145.5090656,
         145.5090656, 145.5090656, 145.5090656, 145.5090656, 145.5090656,
         145.5090656, 145.5090656, 145.5090656, 168.7598291, 168.7598291,
         168.7598291, 168.7598291, 168.7598291, 168.7598291, 168.7598291,
         168.7598291, 168.7598291, 168.7598291, 168.7598291, 168.7598291,
         168.7598291, 168.7598291, 168.7598291, 168.7598291, 178.3395514,
         178.3395514, 178.3395514, 178.3395514, 178.3395514, 178.3395514,
         178.3395514, 178.3395514]
    expected_wgt_data_sat_y = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         81.7865761, 81.7865761, 81.7865761, 81.7865761, 81.7865761, 81.7865761,
         81.7865761, 81.7865761, 81.7865761, 81.7865761, 81.7865761, 81.7865761,
         81.7865761, 81.7865761, 81.7865761, 81.7865761, 103.1783112,
         103.1783112, 103.1783112, 103.1783112, 103.1783112, 103.1783112,
         103.1783112, 103.1783112, 103.1783112, 103.1783112, 103.1783112,
         103.1783112, 103.1783112, 103.1783112, 103.1783112, 103.1783112,
         103.1783112, 103.1783112, 103.1783112, 103.1783112, 111.2873711,
         111.2873711, 111.2873711, 111.2873711]
    expected_wgt_data_sat_u = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 14.5678871, 14.5678871,
         14.5678871, 14.5678871, 26.7975626, 26.7975626, 26.7975626, 26.7975626,
         35.560508, 35.560508, 35.560508, 35.560508, 47.2134176, 47.2134176,
         47.2134176, 47.2134176, 77.59643, 77.59643, 77.59643, 77.59643,
         99.3668097, 99.3668097, 99.3668097, 99.3668097, 114.9659684,
         114.9659684, 114.9659684, 114.9659684, 126.143254, 126.143254,
         126.143254, 126.143254, 136.8391216, 136.8391216, 136.8391216,
         136.8391216, 144.5030455, 144.5030455, 144.5030455, 144.5030455,
         149.994487, 149.994487, 149.994487, 149.994487, 153.9292768,
         153.9292768, 153.9292768, 153.9292768, 158.719138, 158.719138,
         158.719138, 158.719138, 162.1512235, 162.1512235, 162.1512235,
         162.1512235]
    expected_wgt_data_no_sat_x = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 21.7305663, 21.7305663,
         21.7305663, 21.7305663, 46.1899173, 46.1899173, 46.1899173, 46.1899173,
         46.1899173, 46.1899173, 46.1899173, 46.1899173, 73.7224895, 73.7224895,
         73.7224895, 73.7224895, 145.5090656, 145.5090656, 145.5090656,
         145.5090656, 145.5090656, 145.5090656, 145.5090656, 145.5090656,
         145.5090656, 145.5090656, 145.5090656, 145.5090656, 145.5090656,
         145.5090656, 145.5090656, 145.5090656, 168.7598291, 168.7598291,
         168.7598291, 168.7598291, 168.7598291, 168.7598291, 168.7598291,
         168.7598291, 168.7598291, 168.7598291, 168.7598291, 168.7598291,
         168.7598291, 168.7598291, 168.7598291, 168.7598291, 178.3395514,
         178.3395514, 178.3395514, 178.3395514, 178.3395514, 178.3395514,
         178.3395514, 178.3395514]
    expected_wgt_data_no_sat_y = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         81.7865761, 81.7865761, 81.7865761, 81.7865761, 81.7865761, 81.7865761,
         81.7865761, 81.7865761, 81.7865761, 81.7865761, 81.7865761, 81.7865761,
         81.7865761, 81.7865761, 81.7865761, 81.7865761, 103.1783112,
         103.1783112, 103.1783112, 103.1783112, 103.1783112, 103.1783112,
         103.1783112, 103.1783112, 103.1783112, 103.1783112, 103.1783112,
         103.1783112, 103.1783112, 103.1783112, 103.1783112, 103.1783112,
         103.1783112, 103.1783112, 103.1783112, 103.1783112, 111.2873711,
         111.2873711, 111.2873711, 111.2873711]
    expected_wgt_data_no_sat_u = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 14.5678871, 14.5678871,
         14.5678871, 14.5678871, 26.7975626, 26.7975626, 26.7975626, 26.7975626,
         35.560508, 35.560508, 35.560508, 35.560508, 47.2134176, 47.2134176,
         47.2134176, 47.2134176, 77.59643, 77.59643, 77.59643, 77.59643,
         99.3668097, 99.3668097, 99.3668097, 99.3668097, 114.9659684,
         114.9659684, 114.9659684, 114.9659684, 126.143254, 126.143254,
         126.143254, 126.143254, 136.8391216, 136.8391216, 136.8391216,
         136.8391216, 144.5030455, 144.5030455, 144.5030455, 144.5030455,
         149.994487, 149.994487, 149.994487, 149.994487, 153.9292768,
         153.9292768, 153.9292768, 153.9292768, 158.719138, 158.719138,
         158.719138, 158.719138, 162.1512235, 162.1512235, 162.1512235,
         162.1512235]

    def test_learning_graded_spike_reg_imp_floating_pt_x0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, USE_REGULAR_IMPULSE GradedSpikeCfg and
        dw=x0 * x1 learning rule."""

        expected_x1_data = self.expected_x1_data_default
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_default_x

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "x"
        graded_spike_cfg = GradedSpikeCfg.USE_REGULAR_IMPULSE

        pattern_pre, learning_dense, pattern_post = \
            self.create_network_single_synapse(num_steps,
                                               weights_init,
                                               learning_rule_cnd,
                                               graded_spike_cfg)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        exception_map = {
            RingBuffer: PySendModelFloat
        }
        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_reg_imp_floating_pt_y0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, USE_REGULAR_IMPULSE GradedSpikeCfg and
        dw=y0 * x1 learning rule."""

        expected_x1_data = self.expected_x1_data_default
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_default_y

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "y"
        graded_spike_cfg = GradedSpikeCfg.USE_REGULAR_IMPULSE

        pattern_pre, learning_dense, pattern_post = \
            self.create_network_single_synapse(num_steps,
                                               weights_init,
                                               learning_rule_cnd,
                                               graded_spike_cfg)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        exception_map = {
            RingBuffer: PySendModelFloat
        }
        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_reg_imp_floating_pt_u0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, USE_REGULAR_IMPULSE GradedSpikeCfg and
        dw=u0 * x1 learning rule."""

        expected_x1_data = self.expected_x1_data_default
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_default_u

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "u"
        graded_spike_cfg = GradedSpikeCfg.USE_REGULAR_IMPULSE

        pattern_pre, learning_dense, pattern_post = \
            self.create_network_single_synapse(num_steps,
                                               weights_init,
                                               learning_rule_cnd,
                                               graded_spike_cfg)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        exception_map = {
            RingBuffer: PySendModelFloat
        }
        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_overwrite_floating_pt_x0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, OVERWRITE GradedSpikeCfg and dw=x0 * x1
        learning rule."""

        expected_x1_data = self.expected_x1_data_overwrite
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_overwrite_x

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "x"
        graded_spike_cfg = GradedSpikeCfg.OVERWRITE

        pattern_pre, learning_dense, pattern_post = \
            self.create_network_single_synapse(num_steps,
                                               weights_init,
                                               learning_rule_cnd,
                                               graded_spike_cfg)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        exception_map = {
            RingBuffer: PySendModelFloat
        }
        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_overwrite_floating_pt_y0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, OVERWRITE GradedSpikeCfg and dw=y0 * x1
        learning rule."""

        expected_x1_data = self.expected_x1_data_overwrite
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_overwrite_y

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "y"
        graded_spike_cfg = GradedSpikeCfg.OVERWRITE

        pattern_pre, learning_dense, pattern_post = \
            self.create_network_single_synapse(num_steps,
                                               weights_init,
                                               learning_rule_cnd,
                                               graded_spike_cfg)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        exception_map = {
            RingBuffer: PySendModelFloat
        }
        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_overwrite_floating_pt_u0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, OVERWRITE GradedSpikeCfg and dw=u0 * x1
        learning rule."""

        expected_x1_data = self.expected_x1_data_overwrite
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_overwrite_u

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "u"
        graded_spike_cfg = GradedSpikeCfg.OVERWRITE

        pattern_pre, learning_dense, pattern_post = \
            self.create_network_single_synapse(num_steps,
                                               weights_init,
                                               learning_rule_cnd,
                                               graded_spike_cfg)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        exception_map = {
            RingBuffer: PySendModelFloat
        }
        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_sat_floating_pt_x0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, ADD_SATURATION GradedSpikeCfg and dw=x0 * x1
        learning rule."""

        expected_x1_data = self.expected_x1_data_sat
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_sat_x

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "x"
        graded_spike_cfg = GradedSpikeCfg.ADD_SATURATION

        pattern_pre, learning_dense, pattern_post = \
            self.create_network_single_synapse(num_steps,
                                               weights_init,
                                               learning_rule_cnd,
                                               graded_spike_cfg)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        exception_map = {
            RingBuffer: PySendModelFloat
        }
        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_sat_floating_pt_y0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, ADD_SATURATION GradedSpikeCfg and dw=y0 * x1
        learning rule."""

        expected_x1_data = self.expected_x1_data_sat
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_sat_y

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "y"
        graded_spike_cfg = GradedSpikeCfg.ADD_SATURATION

        pattern_pre, learning_dense, pattern_post = \
            self.create_network_single_synapse(num_steps,
                                               weights_init,
                                               learning_rule_cnd,
                                               graded_spike_cfg)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        exception_map = {
            RingBuffer: PySendModelFloat
        }
        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_sat_floating_pt_u0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, ADD_SATURATION GradedSpikeCfg and dw=u0 * x1
        learning rule."""

        expected_x1_data = self.expected_x1_data_sat
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_sat_u

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "u"
        graded_spike_cfg = GradedSpikeCfg.ADD_SATURATION

        pattern_pre, learning_dense, pattern_post = \
            self.create_network_single_synapse(num_steps,
                                               weights_init,
                                               learning_rule_cnd,
                                               graded_spike_cfg)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        exception_map = {
            RingBuffer: PySendModelFloat
        }
        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_no_sat_floating_pt_x0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, ADD_NO_SATURATION GradedSpikeCfg and
        dw=x0 * x1 learning rule."""

        expected_x1_data = self.expected_x1_data_no_sat
        expected_x2_data = self.expected_x2_data_no_sat
        expected_wgt_data = self.expected_wgt_data_no_sat_x

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "x"
        graded_spike_cfg = GradedSpikeCfg.ADD_NO_SATURATION

        pattern_pre, learning_dense, pattern_post = \
            self.create_network_single_synapse(num_steps,
                                               weights_init,
                                               learning_rule_cnd,
                                               graded_spike_cfg)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        exception_map = {
            RingBuffer: PySendModelFloat
        }
        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_no_sat_floating_pt_y0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, ADD_NO_SATURATION GradedSpikeCfg and
        dw=y0 * x1 learning rule."""

        expected_x1_data = self.expected_x1_data_no_sat
        expected_x2_data = self.expected_x2_data_no_sat
        expected_wgt_data = self.expected_wgt_data_no_sat_y

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "y"
        graded_spike_cfg = GradedSpikeCfg.ADD_NO_SATURATION

        pattern_pre, learning_dense, pattern_post = \
            self.create_network_single_synapse(num_steps,
                                               weights_init,
                                               learning_rule_cnd,
                                               graded_spike_cfg)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        exception_map = {
            RingBuffer: PySendModelFloat
        }
        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_no_sat_floating_pt_u0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, ADD_NO_SATURATION GradedSpikeCfg and
        dw=u0 * x1 learning rule."""

        expected_x1_data = self.expected_x1_data_no_sat
        expected_x2_data = self.expected_x2_data_no_sat
        expected_wgt_data = self.expected_wgt_data_no_sat_u

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "u"
        graded_spike_cfg = GradedSpikeCfg.ADD_NO_SATURATION

        pattern_pre, learning_dense, pattern_post = \
            self.create_network_single_synapse(num_steps,
                                               weights_init,
                                               learning_rule_cnd,
                                               graded_spike_cfg)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        exception_map = {
            RingBuffer: PySendModelFloat
        }
        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)


class TestLearningSimGradedSpikeBitApprox(unittest.TestCase):
    """Known value test suite. Tests have been run once and validated.
    Resulting values are stored as 'expected' values.
    Any deviation from these expected values in the future would be a symptom
    of breaking changes"""

    @staticmethod
    def create_network_single_synapse(num_steps: int,
                                      weights_init: np.ndarray,
                                      learning_rule_cnd: str,
                                      graded_spike_cfg: GradedSpikeCfg) \
            -> ty.Tuple[RingBuffer, LearningDense, RingBuffer]:
        """Create a network of RingBuffer -> LearningDense -> RingBuffer.

        Parameters
        ----------
        num_steps : int
            Number of simulation time steps.
        weights_init : ndarray
            Initial weights matrix for the LearningDense.
        learning_rule_cnd : str
            String specifying which learning rule condition to use.
        graded_spike_cfg : GradedSpikeCfg
            GradedSpikeCfg to use for the LearningDense.

        Returns
        ----------
        pattern_pre : RingBuffer
            Pre-synaptic RingBuffer Process.
        learning_dense : LearningDense
            LearningDense Process.
        pattern_post : RingBuffer
            Post-synaptic RingBuffer Process.
        """

        if learning_rule_cnd == "x" or learning_rule_cnd == "y":
            scaling_exp = -1
        elif learning_rule_cnd == "u":
            scaling_exp = -2
        else:
            scaling_exp = 0

        dw = f"2^{scaling_exp} * {learning_rule_cnd}0 * x1"

        x1_impulse = \
            16 if graded_spike_cfg == GradedSpikeCfg.USE_REGULAR_IMPULSE else 0

        learning_rule = Loihi2FLearningRule(dw=dw,
                                            x1_impulse=x1_impulse, x1_tau=12,
                                            x2_impulse=24, x2_tau=5,
                                            y1_impulse=16, y1_tau=10,
                                            y2_impulse=24, y2_tau=5,
                                            t_epoch=4, rng_seed=0)

        spike_raster_pre = np.zeros((1, num_steps))
        spike_raster_pre[0, 4] = 51
        spike_raster_pre[0, 11] = 100
        spike_raster_pre[0, 17] = 60
        spike_raster_pre[0, 21] = 246
        spike_raster_pre[0, 38] = 30
        spike_raster_pre[0, 55] = 22
        spike_raster_post = np.zeros((1, num_steps))
        spike_raster_post[0, 3] = 1
        spike_raster_post[0, 21] = 1
        spike_raster_post[0, 39] = 1
        spike_raster_post[0, 57] = 1

        pattern_pre = RingBuffer(data=spike_raster_pre.astype(int))
        learning_dense = \
            LearningDense(weights=weights_init,
                          learning_rule=learning_rule,
                          name="learning_dense",
                          num_message_bits=8,
                          graded_spike_cfg=graded_spike_cfg)
        pattern_post = RingBuffer(data=spike_raster_post.astype(int))

        pattern_pre.s_out.connect(learning_dense.s_in)
        pattern_post.s_out.connect(learning_dense.s_in_bap)

        return pattern_pre, learning_dense, pattern_post

    expected_x1_data_default = \
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 12.0, 12.0, 12.0, 24.0, 24.0,
         24.0, 24.0, 19.0, 19.0, 19.0, 19.0, 27.0, 27.0, 27.0, 27.0, 34.0, 34.0,
         34.0, 34.0, 25.0, 25.0, 25.0, 25.0, 19.0, 19.0, 19.0, 19.0, 15.0, 15.0,
         15.0, 15.0, 26.0, 26.0, 26.0, 26.0, 20.0, 20.0, 20.0, 20.0, 16.0, 16.0,
         16.0, 16.0, 12.0, 12.0, 12.0, 12.0, 24.0, 24.0, 24.0, 24.0, 18.0, 18.0,
         18.0, 18.0]
    expected_x1_data_overwrite = \
        [0.0, 0.0, 0.0, 0.0, 26.0, 26.0, 26.0, 18.0, 18.0, 18.0, 18.0, 36.0,
         36.0, 36.0, 36.0, 27.0, 27.0, 30.0, 30.0, 22.0, 22.0, 123.0, 123.0,
         90.0, 90.0, 90.0, 90.0, 64.0, 64.0, 64.0, 64.0, 46.0, 46.0, 46.0, 46.0,
         33.0, 33.0, 33.0, 15.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0,
         10.0, 10.0, 10.0, 10.0, 9.0, 9.0, 9.0, 9.0, 7.0, 7.0, 7.0, 7.0, 7.0,
         7.0, 7.0, 7.0]
    expected_x1_data_sat = \
        [0.0, 0.0, 0.0, 0.0, 26.0, 26.0, 26.0, 18.0, 18.0, 18.0, 18.0, 48.0,
         48.0, 48.0, 48.0, 36.0, 36.0, 66.0, 66.0, 47.0, 47.0, 127.0, 127.0,
         92.0, 92.0, 92.0, 92.0, 66.0, 66.0, 66.0, 66.0, 48.0, 48.0, 48.0, 48.0,
         35.0, 35.0, 35.0, 50.0, 35.0, 35.0, 35.0, 35.0, 27.0, 27.0, 27.0, 27.0,
         21.0, 21.0, 21.0, 21.0, 17.0, 17.0, 17.0, 17.0, 20.0, 20.0, 20.0, 20.0,
         16.0, 16.0, 16.0, 16.0]
    expected_x1_data_no_sat = \
        [0.0, 0.0, 0.0, 0.0, 26.0, 26.0, 26.0, 18.0, 18.0, 18.0, 18.0, 48.0,
         48.0, 48.0, 48.0, 36.0, 36.0, 66.0, 66.0, 47.0, 47.0, 43.0, 43.0, 32.0,
         32.0, 32.0, 32.0, 23.0, 23.0, 23.0, 23.0, 18.0, 18.0, 18.0, 18.0, 14.0,
         14.0, 14.0, 29.0, 20.0, 20.0, 20.0, 20.0, 16.0, 16.0, 16.0, 16.0, 12.0,
         12.0, 12.0, 12.0, 9.0, 9.0, 9.0, 9.0, 14.0, 14.0, 14.0, 14.0, 10.0,
         10.0, 10.0, 10.0]

    expected_x2_data = \
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 12.0, 12.0, 12.0, 29.0, 29.0,
         29.0, 29.0, 14.0, 14.0, 14.0, 14.0, 22.0, 22.0, 22.0, 22.0, 27.0, 27.0,
         27.0, 27.0, 12.0, 12.0, 12.0, 12.0, 6.0, 6.0, 6.0, 6.0, 3.0, 3.0, 3.0,
         3.0, 20.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0, 10.0, 6.0, 6.0, 6.0,
         6.0, 4.0, 4.0, 4.0, 4.0, 25.0, 25.0, 25.0, 25.0, 12.0, 12.0, 12.0,
         12.0]
    expected_x2_data_no_sat = \
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17.0, 17.0, 17.0, 17.0,
         8.0, 8.0, 8.0, 8.0, 4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    expected_wgt_data_default_x = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 18.0, 18.0, 18.0, 18.0, 30.0,
         30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 46.0, 46.0, 46.0, 46.0, 66.0,
         66.0, 66.0, 66.0, 66.0, 66.0, 66.0, 66.0, 66.0, 66.0, 66.0, 66.0, 66.0,
         66.0, 66.0, 66.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0,
         80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 92.0, 92.0, 92.0, 92.0, 92.0,
         92.0, 92.0, 92.0]
    expected_wgt_data_default_y = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 30.0,
         30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0,
         30.0, 30.0, 30.0, 43.0, 43.0, 43.0, 43.0, 43.0, 43.0, 43.0, 43.0, 43.0,
         43.0, 43.0, 43.0, 43.0, 43.0, 43.0, 43.0, 43.0, 43.0, 43.0, 43.0, 53.0,
         53.0, 53.0, 53.0]
    expected_wgt_data_default_u = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 13.0, 13.0, 13.0, 13.0, 19.0,
         19.0, 19.0, 19.0, 24.0, 24.0, 24.0, 24.0, 31.0, 31.0, 31.0, 31.0, 40.0,
         40.0, 40.0, 40.0, 47.0, 47.0, 47.0, 47.0, 52.0, 52.0, 52.0, 52.0, 56.0,
         56.0, 56.0, 56.0, 62.0, 62.0, 62.0, 62.0, 67.0, 67.0, 67.0, 67.0, 71.0,
         71.0, 71.0, 71.0, 74.0, 74.0, 74.0, 74.0, 80.0, 80.0, 80.0, 80.0, 84.0,
         84.0, 84.0, 84.0]
    expected_wgt_data_overwrite_x = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 22.0, 22.0, 22.0, 22.0, 40.0,
         40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 53.0, 53.0, 53.0, 53.0,
         106.0, 106.0, 106.0, 106.0, 106.0, 106.0, 106.0, 106.0, 106.0, 106.0,
         106.0, 106.0, 106.0, 106.0, 106.0, 106.0, 112.0, 112.0, 112.0, 112.0,
         112.0, 112.0, 112.0, 112.0, 112.0, 112.0, 112.0, 112.0, 112.0, 112.0,
         112.0, 112.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0]
    expected_wgt_data_overwrite_y = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 63.0,
         63.0, 63.0, 63.0, 63.0, 63.0, 63.0, 63.0, 63.0, 63.0, 63.0, 63.0, 63.0,
         63.0, 63.0, 63.0, 68.0, 68.0, 68.0, 68.0, 68.0, 68.0, 68.0, 68.0, 68.0,
         68.0, 68.0, 68.0, 68.0, 68.0, 68.0, 68.0, 68.0, 68.0, 68.0, 68.0, 71.0,
         71.0, 71.0, 71.0]
    expected_wgt_data_overwrite_u = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 14.0, 14.0, 14.0, 14.0, 23.0,
         23.0, 23.0, 23.0, 30.0, 30.0, 30.0, 30.0, 35.0, 35.0, 35.0, 35.0, 58.0,
         58.0, 58.0, 58.0, 74.0, 74.0, 74.0, 74.0, 86.0, 86.0, 86.0, 86.0, 94.0,
         94.0, 94.0, 94.0, 97.0, 97.0, 97.0, 97.0, 100.0, 100.0, 100.0, 100.0,
         103.0, 103.0, 103.0, 103.0, 105.0, 105.0, 105.0, 105.0, 107.0, 107.0,
         107.0, 107.0, 109.0, 109.0, 109.0, 109.0]
    expected_wgt_data_sat_x = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 22.0, 22.0, 22.0, 22.0, 46.0,
         46.0, 46.0, 46.0, 46.0, 46.0, 46.0, 46.0, 74.0, 74.0, 74.0, 74.0,
         128.0, 128.0, 128.0, 128.0, 128.0, 128.0, 128.0, 128.0, 128.0, 128.0,
         128.0, 128.0, 128.0, 128.0, 128.0, 128.0, 147.0, 147.0, 147.0, 147.0,
         147.0, 147.0, 147.0, 147.0, 147.0, 147.0, 147.0, 147.0, 147.0, 147.0,
         147.0, 147.0, 157.0, 157.0, 157.0, 157.0, 157.0, 157.0, 157.0, 157.0]
    expected_wgt_data_sat_y = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 64.0,
         64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0,
         64.0, 64.0, 64.0, 81.0, 81.0, 81.0, 81.0, 81.0, 81.0, 81.0, 81.0, 81.0,
         81.0, 81.0, 81.0, 81.0, 81.0, 81.0, 81.0, 81.0, 81.0, 81.0, 81.0, 90.0,
         90.0, 90.0, 90.0]
    expected_wgt_data_sat_u = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 14.0, 14.0, 14.0, 14.0, 26.0,
         26.0, 26.0, 26.0, 35.0, 35.0, 35.0, 35.0, 47.0, 47.0, 47.0, 47.0, 70.0,
         70.0, 70.0, 70.0, 87.0, 87.0, 87.0, 87.0, 99.0, 99.0, 99.0, 99.0,
         108.0, 108.0, 108.0, 108.0, 117.0, 117.0, 117.0, 117.0, 124.0, 124.0,
         124.0, 124.0, 129.0, 129.0, 129.0, 129.0, 133.0, 133.0, 133.0, 133.0,
         138.0, 138.0, 138.0, 138.0, 142.0, 142.0, 142.0, 142.0]
    expected_wgt_data_no_sat_x = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 29.0,
         29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0,
         29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0,
         29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0,
         29.0, 29.0, 29.0]
    expected_wgt_data_no_sat_y = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 29.0,
         29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0,
         29.0, 29.0, 29.0, 39.0, 39.0, 39.0, 39.0, 39.0, 39.0, 39.0, 39.0, 39.0,
         39.0, 39.0, 39.0, 39.0, 39.0, 39.0, 39.0, 39.0, 39.0, 39.0, 39.0, 45.0,
         45.0, 45.0, 45.0]
    expected_wgt_data_no_sat_u = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 14.0, 14.0, 14.0, 14.0, 26.0,
         26.0, 26.0, 26.0, 35.0, 35.0, 35.0, 35.0, 47.0, 47.0, 47.0, 47.0, 55.0,
         55.0, 55.0, 55.0, 61.0, 61.0, 61.0, 61.0, 66.0, 66.0, 66.0, 66.0, 69.0,
         69.0, 69.0, 69.0, 74.0, 74.0, 74.0, 74.0, 78.0, 78.0, 78.0, 78.0, 81.0,
         81.0, 81.0, 81.0, 83.0, 83.0, 83.0, 83.0, 87.0, 87.0, 87.0, 87.0, 89.0,
         89.0, 89.0, 89.0]

    def test_learning_graded_spike_reg_imp_bit_approx_x0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, USE_REGULAR_IMPULSE GradedSpikeCfg and
        dw=x0 * x1 learning rule."""

        expected_x1_data = self.expected_x1_data_default
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_default_x

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "x"
        graded_spike_cfg = GradedSpikeCfg.USE_REGULAR_IMPULSE

        pattern_pre, learning_dense, pattern_post = \
            self.create_network_single_synapse(num_steps,
                                               weights_init,
                                               learning_rule_cnd,
                                               graded_spike_cfg)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        exception_map = {
            RingBuffer: PySendModelFixed
        }
        run_cfg = Loihi2SimCfg(select_tag="bit_approximate_loihi",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_reg_imp_bit_approx_y0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, USE_REGULAR_IMPULSE GradedSpikeCfg and
        dw=y0 * x1 learning rule."""

        expected_x1_data = self.expected_x1_data_default
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_default_y

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "y"
        graded_spike_cfg = GradedSpikeCfg.USE_REGULAR_IMPULSE

        pattern_pre, learning_dense, pattern_post = \
            self.create_network_single_synapse(num_steps,
                                               weights_init,
                                               learning_rule_cnd,
                                               graded_spike_cfg)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        exception_map = {
            RingBuffer: PySendModelFixed
        }
        run_cfg = Loihi2SimCfg(select_tag="bit_approximate_loihi",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_reg_imp_bit_approx_u0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, USE_REGULAR_IMPULSE GradedSpikeCfg and
        dw=u0 * x1 learning rule."""

        expected_x1_data = self.expected_x1_data_default
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_default_u

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "u"
        graded_spike_cfg = GradedSpikeCfg.USE_REGULAR_IMPULSE

        pattern_pre, learning_dense, pattern_post = \
            self.create_network_single_synapse(num_steps,
                                               weights_init,
                                               learning_rule_cnd,
                                               graded_spike_cfg)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        exception_map = {
            RingBuffer: PySendModelFixed
        }
        run_cfg = Loihi2SimCfg(select_tag="bit_approximate_loihi",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_overwrite_bit_approx_x0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, OVERWRITE GradedSpikeCfg and dw=x0 * x1
        learning rule."""

        expected_x1_data = self.expected_x1_data_overwrite
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_overwrite_x

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "x"
        graded_spike_cfg = GradedSpikeCfg.OVERWRITE

        pattern_pre, learning_dense, pattern_post = \
            self.create_network_single_synapse(num_steps,
                                               weights_init,
                                               learning_rule_cnd,
                                               graded_spike_cfg)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        exception_map = {
            RingBuffer: PySendModelFixed
        }
        run_cfg = Loihi2SimCfg(select_tag="bit_approximate_loihi",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_overwrite_bit_approx_y0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, OVERWRITE GradedSpikeCfg and dw=y0 * x1
        learning rule."""

        expected_x1_data = self.expected_x1_data_overwrite
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_overwrite_y

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "y"
        graded_spike_cfg = GradedSpikeCfg.OVERWRITE

        pattern_pre, learning_dense, pattern_post = \
            self.create_network_single_synapse(num_steps,
                                               weights_init,
                                               learning_rule_cnd,
                                               graded_spike_cfg)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        exception_map = {
            RingBuffer: PySendModelFixed
        }
        run_cfg = Loihi2SimCfg(select_tag="bit_approximate_loihi",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_overwrite_bit_approx_u0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, OVERWRITE GradedSpikeCfg and dw=u0 * x1
        learning rule."""

        expected_x1_data = self.expected_x1_data_overwrite
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_overwrite_u

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "u"
        graded_spike_cfg = GradedSpikeCfg.OVERWRITE

        pattern_pre, learning_dense, pattern_post = \
            self.create_network_single_synapse(num_steps,
                                               weights_init,
                                               learning_rule_cnd,
                                               graded_spike_cfg)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        exception_map = {
            RingBuffer: PySendModelFixed
        }
        run_cfg = Loihi2SimCfg(select_tag="bit_approximate_loihi",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_sat_bit_approx_x0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, ADD_SATURATION GradedSpikeCfg and dw=x0 * x1
        learning rule."""

        expected_x1_data = self.expected_x1_data_sat
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_sat_x

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "x"
        graded_spike_cfg = GradedSpikeCfg.ADD_SATURATION

        pattern_pre, learning_dense, pattern_post = \
            self.create_network_single_synapse(num_steps,
                                               weights_init,
                                               learning_rule_cnd,
                                               graded_spike_cfg)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        exception_map = {
            RingBuffer: PySendModelFixed
        }
        run_cfg = Loihi2SimCfg(select_tag="bit_approximate_loihi",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_sat_bit_approx_y0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, ADD_SATURATION GradedSpikeCfg and dw=y0 * x1
        learning rule."""

        expected_x1_data = self.expected_x1_data_sat
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_sat_y

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "y"
        graded_spike_cfg = GradedSpikeCfg.ADD_SATURATION

        pattern_pre, learning_dense, pattern_post = \
            self.create_network_single_synapse(num_steps,
                                               weights_init,
                                               learning_rule_cnd,
                                               graded_spike_cfg)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        exception_map = {
            RingBuffer: PySendModelFixed
        }
        run_cfg = Loihi2SimCfg(select_tag="bit_approximate_loihi",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_sat_bit_approx_u0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, ADD_SATURATION GradedSpikeCfg and dw=u0 * x1
        learning rule."""

        expected_x1_data = self.expected_x1_data_sat
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_sat_u

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "u"
        graded_spike_cfg = GradedSpikeCfg.ADD_SATURATION

        pattern_pre, learning_dense, pattern_post = \
            self.create_network_single_synapse(num_steps,
                                               weights_init,
                                               learning_rule_cnd,
                                               graded_spike_cfg)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        exception_map = {
            RingBuffer: PySendModelFixed
        }
        run_cfg = Loihi2SimCfg(select_tag="bit_approximate_loihi",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_no_sat_bit_approx_x0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, ADD_NO_SATURATION GradedSpikeCfg and
        dw=x0 * x1 learning rule."""

        expected_x1_data = self.expected_x1_data_no_sat
        expected_x2_data = self.expected_x2_data_no_sat
        expected_wgt_data = self.expected_wgt_data_no_sat_x

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "x"
        graded_spike_cfg = GradedSpikeCfg.ADD_NO_SATURATION

        pattern_pre, learning_dense, pattern_post = \
            self.create_network_single_synapse(num_steps,
                                               weights_init,
                                               learning_rule_cnd,
                                               graded_spike_cfg)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        exception_map = {
            RingBuffer: PySendModelFixed
        }
        run_cfg = Loihi2SimCfg(select_tag="bit_approximate_loihi",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_no_sat_bit_approx_y0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, ADD_NO_SATURATION GradedSpikeCfg and
        dw=y0 * x1 learning rule."""

        expected_x1_data = self.expected_x1_data_no_sat
        expected_x2_data = self.expected_x2_data_no_sat
        expected_wgt_data = self.expected_wgt_data_no_sat_y

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "y"
        graded_spike_cfg = GradedSpikeCfg.ADD_NO_SATURATION

        pattern_pre, learning_dense, pattern_post = \
            self.create_network_single_synapse(num_steps,
                                               weights_init,
                                               learning_rule_cnd,
                                               graded_spike_cfg)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        exception_map = {
            RingBuffer: PySendModelFixed
        }
        run_cfg = Loihi2SimCfg(select_tag="bit_approximate_loihi",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_no_sat_bit_approx_u0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, ADD_NO_SATURATION GradedSpikeCfg and
        dw=u0 * x1 learning rule."""

        expected_x1_data = self.expected_x1_data_no_sat
        expected_x2_data = self.expected_x2_data_no_sat
        expected_wgt_data = self.expected_wgt_data_no_sat_u

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "u"
        graded_spike_cfg = GradedSpikeCfg.ADD_NO_SATURATION

        pattern_pre, learning_dense, pattern_post = \
            self.create_network_single_synapse(num_steps,
                                               weights_init,
                                               learning_rule_cnd,
                                               graded_spike_cfg)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        exception_map = {
            RingBuffer: PySendModelFixed
        }
        run_cfg = Loihi2SimCfg(select_tag="bit_approximate_loihi",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)
