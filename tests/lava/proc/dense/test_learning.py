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
    def create_network_single_synapse(self,
                                      num_steps: int,
                                      weights_init: np.ndarray,
                                      learning_rule_cnd: str,
                                      graded_spike_cfg: GradedSpikeCfg) \
            -> ty.Tuple[RingBuffer, LearningDense, RingBuffer]:
        """TODO : WRITE"""
        if learning_rule_cnd == "x" or learning_rule_cnd == "y":
            scaling_exp = -1
        elif learning_rule_cnd == "u":
            scaling_exp = -2
        else:
            scaling_exp = 0

        dw = f"2^{scaling_exp} * {learning_rule_cnd}0 * x1"

        print("dw", dw)

        x1_impulse = 16 if graded_spike_cfg == GradedSpikeCfg.DEFAULT else 0

        learning_rule = Loihi2FLearningRule(dw=dw,
                                            x1_impulse=x1_impulse, x1_tau=12,
                                            x2_impulse=24, x2_tau=5,
                                            y1_impulse=16, y1_tau=10,
                                            y2_impulse=24, y2_tau=5,
                                            t_epoch=4)

        spike_raster_pre = np.zeros((1, num_steps))
        spike_raster_pre[0, 4] = 51
        spike_raster_pre[0, 21] = 248
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
         12.4608125, 8.9285623, 8.9285623, 8.9285623, 8.9285623, 6.3975945,
         6.3975945, 6.3975945, 6.3975945, 4.5840767, 4.5840767, 4.5840767,
         4.5840767, 16.8283421, 16.8283421, 16.8283421, 16.8283421, 12.058034,
         12.058034, 12.058034, 12.058034, 8.6399589, 8.6399589, 8.6399589,
         8.6399589, 6.1908011, 6.1908011, 6.1908011, 6.1908011, 19.1566135,
         19.1566135, 19.1566135, 19.1566135, 13.7263133, 13.7263133,
         13.7263133, 13.7263133, 9.8353333, 9.8353333, 9.8353333, 9.8353333,
         7.0473243, 7.0473243, 7.0473243, 7.0473243, 21.0496285, 21.0496285,
         21.0496285, 21.0496285, 15.0827179, 15.0827179, 15.0827179, 15.0827179]
    expected_x1_data_overwrite = \
        [0.0, 0.0, 0.0, 0.0, 25.5, 25.5, 25.5, 18.2715484, 18.2715484,
         18.2715484, 18.2715484, 13.0921365, 13.0921365, 13.0921365,
         13.0921365, 9.3809257, 9.3809257, 9.3809257, 9.3809257, 6.721727,
         6.721727, 124.0, 124.0, 88.8498825, 88.8498825, 88.8498825,
         88.8498825, 63.6637228, 63.6637228, 63.6637228, 63.6637228,
         45.6170507, 45.6170507, 45.6170507, 45.6170507, 32.6860451,
         32.6860451, 32.6860451, 15.0, 10.7479697, 10.7479697, 10.7479697,
         10.7479697, 7.7012568, 7.7012568, 7.7012568, 7.7012568, 5.5181916,
         5.5181916, 5.5181916, 5.5181916, 3.9539571, 3.9539571, 3.9539571,
         3.9539571, 7.8818444, 7.8818444, 7.8818444, 7.8818444, 5.6475883,
         5.6475883, 5.6475883, 5.6475883]
    expected_x1_data_sat = \
        [0.0, 0.0, 0.0, 0.0, 25.5, 25.5, 25.5, 18.2715484, 18.2715484,
         18.2715484, 18.2715484, 13.0921365, 13.0921365, 13.0921365,
         13.0921365, 9.3809257, 9.3809257, 9.3809257, 9.3809257, 6.721727,
         6.721727, 130.721727, 130.721727, 93.6662104, 93.6662104, 93.6662104,
         93.6662104, 67.1147725, 67.1147725, 67.1147725, 67.1147725,
         48.0898359, 48.0898359, 48.0898359, 48.0898359, 34.4578731,
         34.4578731, 34.4578731, 49.4578731, 35.4381147, 35.4381147,
         35.4381147, 35.4381147, 25.3925187, 25.3925187, 25.3925187,
         25.3925187, 18.1945347, 18.1945347, 18.1945347, 18.1945347,
         13.0369538, 13.0369538, 13.0369538, 13.0369538, 17.22323,
         17.22323, 17.22323, 17.22323, 12.3409836, 12.3409836, 12.3409836,
         12.3409836]
    expected_x1_data_no_sat = \
        [0.0, 0.0, 0.0, 0.0, 25.5, 25.5, 25.5, 18.2715484, 18.2715484,
         18.2715484, 18.2715484, 13.0921365, 13.0921365, 13.0921365,
         13.0921365, 9.3809257, 9.3809257, 9.3809257, 9.3809257,
         6.721727, 6.721727, 130.721727, 130.721727, 93.6662104,
         93.6662104, 93.6662104, 93.6662104, 67.1147725, 67.1147725,
         67.1147725, 67.1147725, 48.0898359, 48.0898359, 48.0898359,
         48.0898359, 34.4578731, 34.4578731, 34.4578731, 49.4578731,
         35.4381147, 35.4381147, 35.4381147, 35.4381147, 25.3925187,
         25.3925187, 25.3925187, 25.3925187, 18.1945347, 18.1945347,
         18.1945347, 18.1945347, 13.0369538, 13.0369538, 13.0369538,
         13.0369538, 17.22323, 17.22323, 17.22323, 17.22323, 12.3409836,
         12.3409836, 12.3409836, 12.3409836]

    expected_x2_data = \
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.1714793, 13.1714793, 13.1714793,
         13.1714793, 5.9183271, 5.9183271, 5.9183271, 5.9183271, 2.6592758,
         2.6592758, 2.6592758, 2.6592758, 1.1948896, 1.1948896, 1.1948896,
         1.1948896, 16.6245796, 16.6245796, 16.6245796, 16.6245796, 7.4699051,
         7.4699051, 7.4699051, 7.4699051, 3.3564447, 3.3564447, 3.3564447,
         3.3564447, 1.5081478, 1.5081478, 1.5081478, 1.5081478, 20.3271926,
         20.3271926, 20.3271926, 20.3271926, 9.1335964, 9.1335964, 9.1335964,
         9.1335964, 4.1039894, 4.1039894, 4.1039894, 4.1039894, 1.8440413,
         1.8440413, 1.8440413, 1.8440413, 24.8285812, 24.8285812, 24.8285812,
         24.8285812, 11.1562007, 11.1562007, 11.1562007, 11.1562007]
    expected_x2_data_no_sat = \
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.0876811, 16.0876811,
         16.0876811, 16.0876811, 7.2286611, 7.2286611, 7.2286611, 7.2286611,
         3.2480468, 3.2480468, 3.2480468, 3.2480468, 1.4594415, 1.4594415,
         1.4594415, 1.4594415, 0.6557693, 0.6557693, 0.6557693, 0.6557693,
         0.2946562, 0.2946562, 0.2946562, 0.2946562, 0.1323975, 0.1323975,
         0.1323975, 0.1323975, 0.0594901, 0.0594901, 0.0594901, 0.0594901,
         0.0267306, 0.0267306, 0.0267306, 0.0267306, 0.0120108, 0.0120108,
         0.0120108, 0.0120108]

    expected_wgt_data_default_x = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 18.0, 18.0, 18.0, 18.0, 18.0,
         18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0,
         27.9401686, 27.9401686, 27.9401686, 27.9401686, 27.9401686, 27.9401686,
         27.9401686, 27.9401686, 27.9401686, 27.9401686, 27.9401686, 27.9401686,
         27.9401686, 27.9401686, 27.9401686, 27.9401686, 38.350869, 38.350869,
         38.350869, 38.350869, 38.350869, 38.350869, 38.350869, 38.350869,
         38.350869, 38.350869, 38.350869, 38.350869, 38.350869, 38.350869,
         38.350869, 38.350869, 48.8756832, 48.8756832, 48.8756832, 48.8756832,
         48.8756832, 48.8756832, 48.8756832, 48.8756832]
    expected_wgt_data_default_y = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         19.9401686, 19.9401686, 19.9401686, 19.9401686, 19.9401686, 19.9401686,
         19.9401686, 19.9401686, 19.9401686, 19.9401686, 19.9401686, 19.9401686,
         19.9401686, 19.9401686, 19.9401686, 19.9401686, 29.5184753, 29.5184753,
         29.5184753, 29.5184753, 29.5184753, 29.5184753, 29.5184753, 29.5184753,
         29.5184753, 29.5184753, 29.5184753, 29.5184753, 29.5184753, 29.5184753,
         29.5184753, 29.5184753, 29.5184753, 29.5184753, 29.5184753, 29.5184753,
         38.4275382, 38.4275382, 38.4275382, 38.4275382]
    expected_wgt_data_default_u = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 13.1152031, 13.1152031,
         13.1152031, 13.1152031, 15.3473437, 15.3473437, 15.3473437, 15.3473437,
         16.9467423, 16.9467423, 16.9467423, 16.9467423, 18.0927615, 18.0927615,
         18.0927615, 18.0927615, 22.2998471, 22.2998471, 22.2998471, 22.2998471,
         25.3143556, 25.3143556, 25.3143556, 25.3143556, 27.4743453, 27.4743453,
         27.4743453, 27.4743453, 29.0220456, 29.0220456, 29.0220456, 29.0220456,
         33.8111989, 33.8111989, 33.8111989, 33.8111989, 37.2427773, 37.2427773,
         37.2427773, 37.2427773, 39.7016106, 39.7016106, 39.7016106, 39.7016106,
         41.4634417, 41.4634417, 41.4634417, 41.4634417, 46.7258488, 46.7258488,
         46.7258488, 46.7258488, 50.4965282, 50.4965282, 50.4965282, 50.4965282]
    expected_wgt_data_overwrite_x = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 21.7305663, 21.7305663,
         21.7305663, 21.7305663, 21.7305663, 21.7305663, 21.7305663, 21.7305663,
         21.7305663, 21.7305663, 21.7305663, 21.7305663, 21.7305663, 21.7305663,
         21.7305663, 21.7305663, 74.2124332, 74.2124332, 74.2124332, 74.2124332,
         74.2124332, 74.2124332, 74.2124332, 74.2124332, 74.2124332, 74.2124332,
         74.2124332, 74.2124332, 74.2124332, 74.2124332, 74.2124332, 74.2124332,
         80.0534391, 80.0534391, 80.0534391, 80.0534391, 80.0534391, 80.0534391,
         80.0534391, 80.0534391, 80.0534391, 80.0534391, 80.0534391, 80.0534391,
         80.0534391, 80.0534391, 80.0534391, 80.0534391, 83.9943613, 83.9943613,
         83.9943613, 83.9943613, 83.9943613, 83.9943613, 83.9943613, 83.9943613]
    expected_wgt_data_overwrite_y = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         62.4818669, 62.4818669, 62.4818669, 62.4818669, 62.4818669, 62.4818669,
         62.4818669, 62.4818669, 62.4818669, 62.4818669, 62.4818669, 62.4818669,
         62.4818669, 62.4818669, 62.4818669, 62.4818669, 67.8558518, 67.8558518,
         67.8558518, 67.8558518, 67.8558518, 67.8558518, 67.8558518, 67.8558518,
         67.8558518, 67.8558518, 67.8558518, 67.8558518, 67.8558518, 67.8558518,
         67.8558518, 67.8558518, 67.8558518, 67.8558518, 67.8558518, 67.8558518,
         71.1917704, 71.1917704, 71.1917704, 71.1917704]
    expected_wgt_data_overwrite_u = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 14.5678871, 14.5678871,
         14.5678871, 14.5678871, 17.8409212, 17.8409212, 17.8409212, 17.8409212,
         20.1861527, 20.1861527, 20.1861527, 20.1861527, 21.8665844, 21.8665844,
         21.8665844, 21.8665844, 44.0790551, 44.0790551, 44.0790551, 44.0790551,
         59.9949857, 59.9949857, 59.9949857, 59.9949857, 71.3992484, 71.3992484,
         71.3992484, 71.3992484, 79.5707597, 79.5707597, 79.5707597, 79.5707597,
         82.2577521, 82.2577521, 82.2577521, 82.2577521, 84.1830663, 84.1830663,
         84.1830663, 84.1830663, 85.5626142, 85.5626142, 85.5626142, 85.5626142,
         86.5511035, 86.5511035, 86.5511035, 86.5511035, 88.5215646, 88.5215646,
         88.5215646, 88.5215646, 89.9334617, 89.9334617, 89.9334617, 89.9334617]
    expected_wgt_data_sat_x = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 21.7305663, 21.7305663,
         21.7305663, 21.7305663, 21.7305663, 21.7305663, 21.7305663, 21.7305663,
         21.7305663, 21.7305663, 21.7305663, 21.7305663, 21.7305663, 21.7305663,
         21.7305663, 21.7305663, 77.0573428, 77.0573428, 77.0573428, 77.0573428,
         77.0573428, 77.0573428, 77.0573428, 77.0573428, 77.0573428, 77.0573428,
         77.0573428, 77.0573428, 77.0573428, 77.0573428, 77.0573428, 77.0573428,
         96.3162579, 96.3162579, 96.3162579, 96.3162579, 96.3162579, 96.3162579,
         96.3162579, 96.3162579, 96.3162579, 96.3162579, 96.3162579, 96.3162579,
         96.3162579, 96.3162579, 96.3162579, 96.3162579, 104.9278729,
         104.9278729, 104.9278729, 104.9278729, 104.9278729, 104.9278729,
         104.9278729, 104.9278729]
    expected_wgt_data_sat_y = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         65.3267765, 65.3267765, 65.3267765, 65.3267765, 65.3267765, 65.3267765,
         65.3267765, 65.3267765, 65.3267765, 65.3267765, 65.3267765, 65.3267765,
         65.3267765, 65.3267765, 65.3267765, 65.3267765, 83.0458338, 83.0458338,
         83.0458338, 83.0458338, 83.0458338, 83.0458338, 83.0458338, 83.0458338,
         83.0458338, 83.0458338, 83.0458338, 83.0458338, 83.0458338, 83.0458338,
         83.0458338, 83.0458338, 83.0458338, 83.0458338, 83.0458338, 83.0458338,
         90.3354085, 90.3354085, 90.3354085, 90.3354085]
    expected_wgt_data_sat_u = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 14.5678871, 14.5678871,
         14.5678871, 14.5678871, 17.8409212, 17.8409212, 17.8409212, 17.8409212,
         20.1861527, 20.1861527, 20.1861527, 20.1861527, 21.8665844, 21.8665844,
         21.8665844, 21.8665844, 45.283137, 45.283137, 45.283137, 45.283137,
         62.0618301, 62.0618301, 62.0618301, 62.0618301, 74.0842891, 74.0842891,
         74.0842891, 74.0842891, 82.6987574, 82.6987574, 82.6987574, 82.6987574,
         91.5582861, 91.5582861, 91.5582861, 91.5582861, 97.9064158, 97.9064158,
         97.9064158, 97.9064158, 102.4550494, 102.4550494, 102.4550494,
         102.4550494, 105.7142879, 105.7142879, 105.7142879, 105.7142879,
         110.0200954, 110.0200954, 110.0200954, 110.0200954, 113.1053413,
         113.1053413, 113.1053413, 113.1053413]
    expected_wgt_data_no_sat_x = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         65.3267765, 65.3267765, 65.3267765, 65.3267765, 65.3267765, 65.3267765,
         65.3267765, 65.3267765, 65.3267765, 65.3267765, 65.3267765, 65.3267765,
         65.3267765, 65.3267765, 65.3267765, 65.3267765, 65.3267765, 65.3267765,
         65.3267765, 65.3267765, 65.3267765, 65.3267765, 65.3267765, 65.3267765,
         65.3267765, 65.3267765, 65.3267765, 65.3267765, 65.3267765, 65.3267765,
         65.3267765, 65.3267765, 65.3267765, 65.3267765, 65.3267765, 65.3267765,
         65.3267765, 65.3267765, 65.3267765, 65.3267765]
    expected_wgt_data_no_sat_y = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         65.3267765, 65.3267765, 65.3267765, 65.3267765, 65.3267765, 65.3267765,
         65.3267765, 65.3267765, 65.3267765, 65.3267765, 65.3267765, 65.3267765,
         65.3267765, 65.3267765, 65.3267765, 65.3267765, 83.0458338, 83.0458338,
         83.0458338, 83.0458338, 83.0458338, 83.0458338, 83.0458338, 83.0458338,
         83.0458338, 83.0458338, 83.0458338, 83.0458338, 83.0458338, 83.0458338,
         83.0458338, 83.0458338, 83.0458338, 83.0458338, 83.0458338, 83.0458338,
         90.3354085, 90.3354085, 90.3354085, 90.3354085]
    expected_wgt_data_no_sat_u = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 14.5678871, 14.5678871,
         14.5678871, 14.5678871, 17.8409212, 17.8409212, 17.8409212, 17.8409212,
         20.1861527, 20.1861527, 20.1861527, 20.1861527, 21.8665844, 21.8665844,
         21.8665844, 21.8665844, 45.283137, 45.283137, 45.283137, 45.283137,
         62.0618301, 62.0618301, 62.0618301, 62.0618301, 74.0842891, 74.0842891,
         74.0842891, 74.0842891, 82.6987574, 82.6987574, 82.6987574, 82.6987574,
         91.5582861, 91.5582861, 91.5582861, 91.5582861, 97.9064158, 97.9064158,
         97.9064158, 97.9064158, 102.4550494, 102.4550494, 102.4550494,
         102.4550494, 105.7142879, 105.7142879, 105.7142879, 105.7142879,
         110.0200954, 110.0200954, 110.0200954, 110.0200954, 113.1053413,
         113.1053413, 113.1053413, 113.1053413]

    def print_plot(self, learning_rule_cnd, graded_spike_cfg, num_steps,
                   x1_data, x2_data, wgt_data,
                   expected_x1_data, expected_x2_data, expected_wgt_data,
                   loihi_x1_data, loihi_x2_data, loihi_wgt_data):
        list_x1_data = [round(data_pt, 7) for data_pt in x1_data.tolist()]
        list_x2_data = [round(data_pt, 7) for data_pt in x2_data.tolist()]
        list_wgt_data = [round(data_pt, 7) for data_pt in wgt_data.tolist()]

        print("list_x1_data")
        print(list_x1_data)
        print("list_x2_data")
        print(list_x2_data)
        print("list_wgt_data")
        print(list_wgt_data)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.title(
            f"lr_cnd={learning_rule_cnd}/gs_cfg={graded_spike_cfg}/floating-pt/x1")
        plt.step(list(range(num_steps)), x1_data, label="data")
        plt.step(list(range(num_steps)), expected_x1_data,
                 label="expected_data")
        plt.step(list(range(num_steps)), loihi_x1_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title(
            f"lr_cnd={learning_rule_cnd}/gs_cfg={graded_spike_cfg}/floating-pt/x2")
        plt.step(list(range(num_steps)), x2_data, label="data")
        plt.step(list(range(num_steps)), expected_x2_data,
                 label="expected_data")
        plt.step(list(range(num_steps)), loihi_x2_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title(
            f"lr_cnd={learning_rule_cnd}/gs_cfg={graded_spike_cfg}/floating-pt/wgt")
        plt.step(list(range(num_steps)), wgt_data, label="data")
        plt.step(list(range(num_steps)), expected_wgt_data,
                 label="expected_data")
        plt.step(list(range(num_steps)), loihi_wgt_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

    loihi_x1_data_default = \
        [0, 0, 0, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 5, 5, 5, 5, 4, 4, 4,
         4, 16, 16, 16, 16, 12, 12, 12, 12, 9, 9, 9, 9, 7, 7, 7, 7, 19, 19, 19,
         19, 13, 13, 13, 13, 9, 9, 9, 9, 7, 7, 7, 7, 21, 21, 21, 21, 15, 15, 15,
         15]
    loihi_x1_data_overwrite = \
        [0, 0, 0, 0, 26, 26, 26, 18, 18, 18, 18, 12, 12, 12, 12, 8, 8, 8, 8, 6,
         6, 123, 123, 86, 86, 86, 86, 62, 62, 62, 62, 45, 45, 45, 45, 32, 32,
         32, 15, 10, 10, 10, 10, 7, 7, 7, 7, 4, 4, 4, 4, 3, 3, 3, 3, 6, 6, 6, 6,
         5, 5, 5, 5]
    loihi_x1_data_sat = \
        [0, 0, 0, 0, 26, 26, 26, 18, 18, 18, 18, 12, 12, 12, 12, 8, 8, 8, 8, 6,
         6, 127, 127, 90, 90, 90, 90, 64, 64, 64, 64, 46, 46, 46, 46, 33, 33,
         33, 48, 33, 33, 33, 33, 23, 23, 23, 23, 16, 16, 16, 16, 12, 12, 12, 12,
         15, 15, 15, 15, 11, 11, 11, 11]
    loihi_x1_data_no_sat = \
        [0, 0, 0, 0, 26, 26, 26, 19, 19, 19, 19, 13, 13, 13, 13, 9, 9, 9, 9, 7,
         7, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 10, 10, 10,
         10, 7, 7, 7, 7, 4, 4, 4, 4, 3, 3, 3, 3, 8, 8, 8, 8, 6, 6, 6, 6]

    loihi_x2_data = \
        [0, 0, 0, 0, 0, 0, 0, 13, 13, 13, 13, 5, 5, 5, 5, 2, 2, 2, 2, 1, 1, 1,
         1, 15, 15, 15, 15, 7, 7, 7, 7, 4, 4, 4, 4, 2, 2, 2, 2, 20, 20, 20, 20,
         8, 8, 8, 8, 3, 3, 3, 3, 2, 2, 2, 2, 25, 25, 25, 25, 12, 12, 12, 12]
    loihi_x2_data_no_sat = \
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         15, 15, 15, 15, 7, 7, 7, 7, 4, 4, 4, 4, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    loihi_wgt_data_default_x = \
        [10, 10, 10, 10, 10, 10, 10, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
         18, 18, 18, 18, 18, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
         27, 27, 27, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
         37, 47, 47, 47, 47, 47, 47, 47, 47]
    loihi_wgt_data_default_y = \
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
         10, 10, 10, 10, 10, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
         19, 19, 19, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
         28, 28, 28, 28, 28, 36, 36, 36, 36]
    loihi_wgt_data_default_u = \
        [10, 10, 10, 10, 10, 10, 10, 13, 13, 13, 13, 15, 15, 15, 15, 17, 17, 17,
         17, 18, 18, 18, 18, 22, 22, 22, 22, 25, 25, 25, 25, 27, 27, 27, 27, 29,
         29, 29, 29, 33, 33, 33, 33, 36, 36, 36, 36, 38, 38, 38, 38, 40, 40, 40,
         40, 45, 45, 45, 45, 49, 49, 49, 49]
    loihi_wgt_data_overwrite_x = \
        [10, 10, 10, 10, 10, 10, 10, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
         21, 21, 21, 21, 21, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72,
         72, 72, 72, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77,
         77, 80, 80, 80, 80, 80, 80, 80, 80]
    loihi_wgt_data_overwrite_y = \
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
         10, 10, 10, 10, 10, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
         61, 61, 61, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66,
         66, 66, 66, 66, 66, 68, 68, 68, 68]
    loihi_wgt_data_overwrite_u = \
        [10, 10, 10, 10, 10, 10, 10, 14, 14, 14, 14, 17, 17, 17, 17, 19, 19, 19,
         19, 20, 20, 20, 20, 41, 41, 41, 41, 57, 57, 57, 57, 68, 68, 68, 68, 76,
         76, 76, 76, 78, 78, 78, 78, 80, 80, 80, 80, 81, 81, 81, 81, 82, 82, 82,
         82, 83, 83, 83, 83, 84, 84, 84, 84]
    loihi_wgt_data_sat_x = \
        [10, 10, 10, 10, 10, 10, 10, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
         21, 21, 21, 21, 21, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74,
         74, 74, 74, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92,
         92, 99, 99, 99, 99, 99, 99, 99, 99]
    loihi_wgt_data_sat_y = \
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
         10, 10, 10, 10, 10, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
         63, 63, 63, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79,
         79, 79, 79, 79, 79, 85, 85, 85, 85]
    loihi_wgt_data_sat_u = \
        [10, 10, 10, 10, 10, 10, 10, 14, 14, 14, 14, 17, 17, 17, 17, 19, 19, 19,
         19, 20, 20, 20, 20, 42, 42, 42, 42, 58, 58, 58, 58, 69, 69, 69, 69, 77,
         77, 77, 77, 85, 85, 85, 85, 91, 91, 91, 91, 95, 95, 95, 95, 98, 98, 98,
         98, 102, 102, 102, 102, 105, 105, 105, 105]
    loihi_wgt_data_no_sat_x = \
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
         10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
         10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
         10, 10, 10, 10, 10, 10, 10, 10, 10]
    loihi_wgt_data_no_sat_y = \
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
         10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
         10, 10, 10, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
         15, 15, 15, 15, 15, 18, 18, 18, 18]
    loihi_wgt_data_no_sat_u = \
        [10, 10, 10, 10, 10, 10, 10, 15, 15, 15, 15, 19, 19, 19, 19, 22, 22, 22,
         22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
         23, 23, 23, 25, 25, 25, 25, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29,
         29, 31, 31, 31, 31, 32, 32, 32, 32]

    def test_learning_graded_spike_default_floating_pt_x0_condition(self):
        """TODO : WRITE"""

        # TODO : REFACTOR
        expected_x1_data = self.expected_x1_data_default
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_default_x

        loihi_x1_data = self.loihi_x1_data_default
        loihi_x2_data = self.loihi_x2_data
        loihi_wgt_data = self.loihi_wgt_data_default_x

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "x"
        graded_spike_cfg = GradedSpikeCfg.DEFAULT

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

        self.print_plot(learning_rule_cnd, graded_spike_cfg, num_steps,
                        x1_data, x2_data, wgt_data,
                        expected_x1_data, expected_x2_data, expected_wgt_data,
                        loihi_x1_data, loihi_x2_data, loihi_wgt_data)

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_default_floating_pt_y0_condition(self):
        """TODO : WRITE"""

        # TODO : REFACTOR
        expected_x1_data = self.expected_x1_data_default
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_default_y

        loihi_x1_data = self.loihi_x1_data_default
        loihi_x2_data = self.loihi_x2_data
        loihi_wgt_data = self.loihi_wgt_data_default_y

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "y"
        graded_spike_cfg = GradedSpikeCfg.DEFAULT

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

        self.print_plot(learning_rule_cnd, graded_spike_cfg, num_steps,
                        x1_data, x2_data, wgt_data,
                        expected_x1_data, expected_x2_data, expected_wgt_data,
                        loihi_x1_data, loihi_x2_data, loihi_wgt_data)

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_default_floating_pt_u0_condition(self):
        """TODO : WRITE"""

        # TODO : REFACTOR
        expected_x1_data = self.expected_x1_data_default
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_default_u

        loihi_x1_data = self.loihi_x1_data_default
        loihi_x2_data = self.loihi_x2_data
        loihi_wgt_data = self.loihi_wgt_data_default_u

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "u"
        graded_spike_cfg = GradedSpikeCfg.DEFAULT

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

        self.print_plot(learning_rule_cnd, graded_spike_cfg, num_steps,
                        x1_data, x2_data, wgt_data,
                        expected_x1_data, expected_x2_data, expected_wgt_data,
                        loihi_x1_data, loihi_x2_data, loihi_wgt_data)

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_overwrite_floating_pt_x0_condition(self):
        """TODO : WRITE"""

        # TODO : REFACTOR
        expected_x1_data = self.expected_x1_data_overwrite
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_overwrite_x

        loihi_x1_data = self.loihi_x1_data_overwrite
        loihi_x2_data = self.loihi_x2_data
        loihi_wgt_data = self.loihi_wgt_data_overwrite_x

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

        self.print_plot(learning_rule_cnd, graded_spike_cfg, num_steps,
                        x1_data, x2_data, wgt_data,
                        expected_x1_data, expected_x2_data, expected_wgt_data,
                        loihi_x1_data, loihi_x2_data, loihi_wgt_data)

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_overwrite_floating_pt_y0_condition(self):
        """TODO : WRITE"""

        # TODO : REFACTOR
        expected_x1_data = self.expected_x1_data_overwrite
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_overwrite_y

        loihi_x1_data = self.loihi_x1_data_overwrite
        loihi_x2_data = self.loihi_x2_data
        loihi_wgt_data = self.loihi_wgt_data_overwrite_y

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

        self.print_plot(learning_rule_cnd, graded_spike_cfg, num_steps,
                        x1_data, x2_data, wgt_data,
                        expected_x1_data, expected_x2_data, expected_wgt_data,
                        loihi_x1_data, loihi_x2_data, loihi_wgt_data)

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_overwrite_floating_pt_u0_condition(self):
        """TODO : WRITE"""

        # TODO : REFACTOR
        expected_x1_data = self.expected_x1_data_overwrite
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_overwrite_u

        loihi_x1_data = self.loihi_x1_data_overwrite
        loihi_x2_data = self.loihi_x2_data
        loihi_wgt_data = self.loihi_wgt_data_overwrite_u

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

        self.print_plot(learning_rule_cnd, graded_spike_cfg, num_steps,
                        x1_data, x2_data, wgt_data,
                        expected_x1_data, expected_x2_data, expected_wgt_data,
                        loihi_x1_data, loihi_x2_data, loihi_wgt_data)

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_sat_floating_pt_x0_condition(self):
        """TODO : WRITE"""

        # TODO : REFACTOR
        expected_x1_data = self.expected_x1_data_sat
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_sat_x

        loihi_x1_data = self.loihi_x1_data_sat
        loihi_x2_data = self.loihi_x2_data
        loihi_wgt_data = self.loihi_wgt_data_sat_x

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

        self.print_plot(learning_rule_cnd, graded_spike_cfg, num_steps,
                        x1_data, x2_data, wgt_data,
                        expected_x1_data, expected_x2_data, expected_wgt_data,
                        loihi_x1_data, loihi_x2_data, loihi_wgt_data)

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_sat_floating_pt_y0_condition(self):
        """TODO : WRITE"""

        # TODO : REFACTOR
        expected_x1_data = self.expected_x1_data_sat
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_sat_y

        loihi_x1_data = self.loihi_x1_data_sat
        loihi_x2_data = self.loihi_x2_data
        loihi_wgt_data = self.loihi_wgt_data_sat_y

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

        self.print_plot(learning_rule_cnd, graded_spike_cfg, num_steps,
                        x1_data, x2_data, wgt_data,
                        expected_x1_data, expected_x2_data, expected_wgt_data,
                        loihi_x1_data, loihi_x2_data, loihi_wgt_data)

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_sat_floating_pt_u0_condition(self):
        """TODO : WRITE"""

        # TODO : REFACTOR
        expected_x1_data = self.expected_x1_data_sat
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_sat_u

        loihi_x1_data = self.loihi_x1_data_sat
        loihi_x2_data = self.loihi_x2_data
        loihi_wgt_data = self.loihi_wgt_data_sat_u

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

        self.print_plot(learning_rule_cnd, graded_spike_cfg, num_steps,
                        x1_data, x2_data, wgt_data,
                        expected_x1_data, expected_x2_data, expected_wgt_data,
                        loihi_x1_data, loihi_x2_data, loihi_wgt_data)

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_no_sat_floating_pt_x0_condition(self):
        """TODO : WRITE"""

        # TODO : REFACTOR
        expected_x1_data = self.expected_x1_data_no_sat
        expected_x2_data = self.expected_x2_data_no_sat
        expected_wgt_data = self.expected_wgt_data_no_sat_x

        loihi_x1_data = self.loihi_x1_data_no_sat
        loihi_x2_data = self.loihi_x2_data_no_sat
        loihi_wgt_data = self.loihi_wgt_data_no_sat_x

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

        self.print_plot(learning_rule_cnd, graded_spike_cfg, num_steps,
                        x1_data, x2_data, wgt_data,
                        expected_x1_data, expected_x2_data, expected_wgt_data,
                        loihi_x1_data, loihi_x2_data, loihi_wgt_data)

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_no_sat_floating_pt_y0_condition(self):
        """TODO : WRITE"""

        # TODO : REFACTOR
        expected_x1_data = self.expected_x1_data_no_sat
        expected_x2_data = self.expected_x2_data_no_sat
        expected_wgt_data = self.expected_wgt_data_no_sat_y

        loihi_x1_data = self.loihi_x1_data_no_sat
        loihi_x2_data = self.loihi_x2_data_no_sat
        loihi_wgt_data = self.loihi_wgt_data_no_sat_y

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

        self.print_plot(learning_rule_cnd, graded_spike_cfg, num_steps,
                        x1_data, x2_data, wgt_data,
                        expected_x1_data, expected_x2_data, expected_wgt_data,
                        loihi_x1_data, loihi_x2_data, loihi_wgt_data)

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_no_sat_floating_pt_u0_condition(self):
        """TODO : WRITE"""

        # TODO : REFACTOR
        expected_x1_data = self.expected_x1_data_no_sat
        expected_x2_data = self.expected_x2_data_no_sat
        expected_wgt_data = self.expected_wgt_data_no_sat_u

        loihi_x1_data = self.loihi_x1_data_no_sat
        loihi_x2_data = self.loihi_x2_data_no_sat
        loihi_wgt_data = self.loihi_wgt_data_no_sat_u

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

        self.print_plot(learning_rule_cnd, graded_spike_cfg, num_steps,
                        x1_data, x2_data, wgt_data,
                        expected_x1_data, expected_x2_data, expected_wgt_data,
                        loihi_x1_data, loihi_x2_data, loihi_wgt_data)

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)


class TestLearningSimGradedSpikeBitApprox(unittest.TestCase):
    def create_network_single_synapse(self,
                                      num_steps: int,
                                      weights_init: np.ndarray,
                                      learning_rule_cnd: str,
                                      graded_spike_cfg: GradedSpikeCfg) \
            -> ty.Tuple[RingBuffer, LearningDense, RingBuffer]:
        """TODO : WRITE"""
        if learning_rule_cnd == "x" or learning_rule_cnd == "y":
            scaling_exp = -1
        elif learning_rule_cnd == "u":
            scaling_exp = -2
        else:
            scaling_exp = 0

        dw = f"2^{scaling_exp} * {learning_rule_cnd}0 * x1"

        print("dw", dw)

        x1_impulse = 16 if graded_spike_cfg == GradedSpikeCfg.DEFAULT else 0

        learning_rule = Loihi2FLearningRule(dw=dw,
                                            x1_impulse=x1_impulse, x1_tau=12,
                                            x2_impulse=24, x2_tau=5,
                                            y1_impulse=16, y1_tau=10,
                                            y2_impulse=24, y2_tau=5,
                                            t_epoch=4, rng_seed=0)

        spike_raster_pre = np.zeros((1, num_steps))
        spike_raster_pre[0, 4] = 51
        spike_raster_pre[0, 21] = 248
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
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.0, 13.0, 13.0, 13.0, 9.0, 9.0,
         9.0, 9.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 16.0, 16.0, 16.0,
         16.0, 12.0, 12.0, 12.0, 12.0, 10.0, 10.0, 10.0, 10.0, 6.0, 6.0, 6.0,
         6.0, 19.0, 19.0, 19.0, 19.0, 15.0, 15.0, 15.0, 15.0, 11.0, 11.0, 11.0,
         11.0, 7.0, 7.0, 7.0, 7.0, 22.0, 22.0, 22.0, 22.0, 16.0, 16.0, 16.0,
         16.0]
    expected_x1_data_overwrite = \
        [0.0, 0.0, 0.0, 0.0, 25.0, 25.0, 25.0, 17.0, 17.0, 17.0, 17.0, 11.0,
         11.0, 11.0, 11.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 124.0, 124.0, 88.0,
         88.0, 88.0, 88.0, 62.0, 62.0, 62.0, 62.0, 45.0, 45.0, 45.0, 45.0, 32.0,
         32.0, 32.0, 15.0, 11.0, 11.0, 11.0, 11.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
         7.0, 7.0, 3.0, 3.0, 3.0, 3.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]
    expected_x1_data_sat = \
        [0.0, 0.0, 0.0, 0.0, 25.0, 25.0, 25.0, 17.0, 17.0, 17.0, 17.0, 11.0,
         11.0, 11.0, 11.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 127.0, 127.0, 91.0,
         91.0, 91.0, 91.0, 65.0, 65.0, 65.0, 65.0, 48.0, 48.0, 48.0, 48.0, 34.0,
         34.0, 34.0, 49.0, 35.0, 35.0, 35.0, 35.0, 26.0, 26.0, 26.0, 26.0, 20.0,
         20.0, 20.0, 20.0, 13.0, 13.0, 13.0, 13.0, 17.0, 17.0, 17.0, 17.0, 13.0,
         13.0, 13.0, 13.0]
    expected_x1_data_no_sat = \
        [0.0, 0.0, 0.0, 0.0, 25.0, 25.0, 25.0, 17.0, 17.0, 17.0, 17.0, 11.0,
         11.0, 11.0, 11.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 4.0, 4.0, 4.0, 4.0,
         4.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 18.0,
         13.0, 13.0, 13.0, 13.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 5.0,
         5.0, 5.0, 5.0, 12.0, 12.0, 12.0, 12.0, 8.0, 8.0, 8.0, 8.0]

    expected_x2_data = \
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.0, 13.0, 13.0, 13.0, 5.0, 5.0,
         5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 16.0, 16.0, 16.0,
         16.0, 6.0, 6.0, 6.0, 6.0, 4.0, 4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0, 21.0,
         21.0, 21.0, 21.0, 10.0, 10.0, 10.0, 10.0, 6.0, 6.0, 6.0, 6.0, 2.0, 2.0,
         2.0, 2.0, 26.0, 26.0, 26.0, 26.0, 12.0, 12.0, 12.0, 12.0]
    expected_x2_data_no_sat = \
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.0, 16.0, 16.0, 16.0,
         6.0, 6.0, 6.0, 6.0, 4.0, 4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
         2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    expected_wgt_data_default_x = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 18.0, 18.0, 18.0, 18.0, 18.0,
         18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 28.0,
         28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0,
         28.0, 28.0, 28.0, 38.0, 38.0, 38.0, 38.0, 38.0, 38.0, 38.0, 38.0, 38.0,
         38.0, 38.0, 38.0, 38.0, 38.0, 38.0, 38.0, 49.0, 49.0, 49.0, 49.0, 49.0,
         49.0, 49.0, 49.0]
    expected_wgt_data_default_y = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 20.0,
         20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0,
         20.0, 20.0, 20.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0,
         29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 38.0,
         38.0, 38.0, 38.0]
    expected_wgt_data_default_u = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 13.0, 13.0, 13.0, 13.0, 16.0,
         16.0, 16.0, 16.0, 17.0, 17.0, 17.0, 17.0, 18.0, 18.0, 18.0, 18.0, 22.0,
         22.0, 22.0, 22.0, 25.0, 25.0, 25.0, 25.0, 28.0, 28.0, 28.0, 28.0, 29.0,
         29.0, 29.0, 29.0, 34.0, 34.0, 34.0, 34.0, 38.0, 38.0, 38.0, 38.0, 41.0,
         41.0, 41.0, 41.0, 43.0, 43.0, 43.0, 43.0, 49.0, 49.0, 49.0, 49.0, 53.0,
         53.0, 53.0, 53.0]
    expected_wgt_data_overwrite_x = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 21.0, 21.0, 21.0, 21.0, 21.0,
         21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 74.0,
         74.0, 74.0, 74.0, 74.0, 74.0, 74.0, 74.0, 74.0, 74.0, 74.0, 74.0, 74.0,
         74.0, 74.0, 74.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0,
         80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 84.0, 84.0, 84.0, 84.0, 84.0,
         84.0, 84.0, 84.0]
    expected_wgt_data_overwrite_y = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 63.0,
         63.0, 63.0, 63.0, 63.0, 63.0, 63.0, 63.0, 63.0, 63.0, 63.0, 63.0, 63.0,
         63.0, 63.0, 63.0, 68.0, 68.0, 68.0, 68.0, 68.0, 68.0, 68.0, 68.0, 68.0,
         68.0, 68.0, 68.0, 68.0, 68.0, 68.0, 68.0, 68.0, 68.0, 68.0, 68.0, 71.0,
         71.0, 71.0, 71.0]
    expected_wgt_data_overwrite_u = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 14.0, 14.0, 14.0, 14.0, 17.0,
         17.0, 17.0, 17.0, 19.0, 19.0, 19.0, 19.0, 21.0, 21.0, 21.0, 21.0, 43.0,
         43.0, 43.0, 43.0, 59.0, 59.0, 59.0, 59.0, 70.0, 70.0, 70.0, 70.0, 78.0,
         78.0, 78.0, 78.0, 81.0, 81.0, 81.0, 81.0, 83.0, 83.0, 83.0, 83.0, 85.0,
         85.0, 85.0, 85.0, 86.0, 86.0, 86.0, 86.0, 88.0, 88.0, 88.0, 88.0, 90.0,
         90.0, 90.0, 90.0]
    expected_wgt_data_sat_x = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 21.0, 21.0, 21.0, 21.0, 21.0,
         21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 75.0,
         75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0,
         75.0, 75.0, 75.0, 94.0, 94.0, 94.0, 94.0, 94.0, 94.0, 94.0, 94.0, 94.0,
         94.0, 94.0, 94.0, 94.0, 94.0, 94.0, 94.0, 103.0, 103.0, 103.0, 103.0,
         103.0, 103.0, 103.0, 103.0]
    expected_wgt_data_sat_y = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 64.0,
         64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0,
         64.0, 64.0, 64.0, 81.0, 81.0, 81.0, 81.0, 81.0, 81.0, 81.0, 81.0, 81.0,
         81.0, 81.0, 81.0, 81.0, 81.0, 81.0, 81.0, 81.0, 81.0, 81.0, 81.0, 88.0,
         88.0, 88.0, 88.0]
    expected_wgt_data_sat_u = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 14.0, 14.0, 14.0, 14.0, 17.0,
         17.0, 17.0, 17.0, 19.0, 19.0, 19.0, 19.0, 21.0, 21.0, 21.0, 21.0, 44.0,
         44.0, 44.0, 44.0, 61.0, 61.0, 61.0, 61.0, 73.0, 73.0, 73.0, 73.0, 81.0,
         81.0, 81.0, 81.0, 90.0, 90.0, 90.0, 90.0, 97.0, 97.0, 97.0, 97.0,
         102.0, 102.0, 102.0, 102.0, 105.0, 105.0, 105.0, 105.0, 109.0, 109.0,
         109.0, 109.0, 112.0, 112.0, 112.0, 112.0]
    expected_wgt_data_no_sat_x = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 12.0,
         12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0,
         12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0,
         12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0,
         12.0, 12.0, 12.0]
    expected_wgt_data_no_sat_y = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 12.0,
         12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0,
         12.0, 12.0, 12.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0,
         18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 23.0,
         23.0, 23.0, 23.0]
    expected_wgt_data_no_sat_u = \
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 14.0, 14.0, 14.0, 14.0, 17.0,
         17.0, 17.0, 17.0, 19.0, 19.0, 19.0, 19.0, 21.0, 21.0, 21.0, 21.0, 22.0,
         22.0, 22.0, 22.0, 23.0, 23.0, 23.0, 23.0, 24.0, 24.0, 24.0, 24.0, 25.0,
         25.0, 25.0, 25.0, 28.0, 28.0, 28.0, 28.0, 31.0, 31.0, 31.0, 31.0, 33.0,
         33.0, 33.0, 33.0, 34.0, 34.0, 34.0, 34.0, 37.0, 37.0, 37.0, 37.0, 39.0,
         39.0, 39.0, 39.0]

    def print_plot(self, learning_rule_cnd, graded_spike_cfg, num_steps,
                   x1_data, x2_data, wgt_data,
                   expected_x1_data, expected_x2_data, expected_wgt_data,
                   loihi_x1_data, loihi_x2_data, loihi_wgt_data):
        list_x1_data = [round(data_pt, 7) for data_pt in x1_data.tolist()]
        list_x2_data = [round(data_pt, 7) for data_pt in x2_data.tolist()]
        list_wgt_data = [round(data_pt, 7) for data_pt in wgt_data.tolist()]

        print("list_x1_data")
        print(list_x1_data)
        print("list_x2_data")
        print(list_x2_data)
        print("list_wgt_data")
        print(list_wgt_data)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.title(
            f"lr_cnd={learning_rule_cnd}/gs_cfg={graded_spike_cfg}/fixed-pt/x1")
        plt.step(list(range(num_steps)), x1_data, label="data")
        plt.step(list(range(num_steps)), expected_x1_data,
                 label="expected_data")
        plt.step(list(range(num_steps)), loihi_x1_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title(
            f"lr_cnd={learning_rule_cnd}/gs_cfg={graded_spike_cfg}/fixed-pt/x2")
        plt.step(list(range(num_steps)), x2_data, label="data")
        plt.step(list(range(num_steps)), expected_x2_data,
                 label="expected_data")
        plt.step(list(range(num_steps)), loihi_x2_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title(
            f"lr_cnd={learning_rule_cnd}/gs_cfg={graded_spike_cfg}/fixed-pt/wgt")
        plt.step(list(range(num_steps)), wgt_data, label="data")
        plt.step(list(range(num_steps)), expected_wgt_data,
                 label="expected_data")
        plt.step(list(range(num_steps)), loihi_wgt_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

    loihi_x1_data_default = \
        [0, 0, 0, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 5, 5, 5, 5, 4, 4, 4,
         4, 16, 16, 16, 16, 12, 12, 12, 12, 9, 9, 9, 9, 7, 7, 7, 7, 19, 19, 19,
         19, 13, 13, 13, 13, 9, 9, 9, 9, 7, 7, 7, 7, 21, 21, 21, 21, 15, 15, 15,
         15]
    loihi_x1_data_overwrite = \
        [0, 0, 0, 0, 26, 26, 26, 18, 18, 18, 18, 12, 12, 12, 12, 8, 8, 8, 8, 6,
         6, 123, 123, 86, 86, 86, 86, 62, 62, 62, 62, 45, 45, 45, 45, 32, 32,
         32, 15, 10, 10, 10, 10, 7, 7, 7, 7, 4, 4, 4, 4, 3, 3, 3, 3, 6, 6, 6, 6,
         5, 5, 5, 5]
    loihi_x1_data_sat = \
        [0, 0, 0, 0, 26, 26, 26, 18, 18, 18, 18, 12, 12, 12, 12, 8, 8, 8, 8, 6,
         6, 127, 127, 90, 90, 90, 90, 64, 64, 64, 64, 46, 46, 46, 46, 33, 33,
         33, 48, 33, 33, 33, 33, 23, 23, 23, 23, 16, 16, 16, 16, 12, 12, 12, 12,
         15, 15, 15, 15, 11, 11, 11, 11]
    loihi_x1_data_no_sat = \
        [0, 0, 0, 0, 26, 26, 26, 19, 19, 19, 19, 13, 13, 13, 13, 9, 9, 9, 9, 7,
         7, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 10, 10, 10,
         10, 7, 7, 7, 7, 4, 4, 4, 4, 3, 3, 3, 3, 8, 8, 8, 8, 6, 6, 6, 6]

    loihi_x2_data = \
        [0, 0, 0, 0, 0, 0, 0, 13, 13, 13, 13, 5, 5, 5, 5, 2, 2, 2, 2, 1, 1, 1,
         1, 15, 15, 15, 15, 7, 7, 7, 7, 4, 4, 4, 4, 2, 2, 2, 2, 20, 20, 20, 20,
         8, 8, 8, 8, 3, 3, 3, 3, 2, 2, 2, 2, 25, 25, 25, 25, 12, 12, 12, 12]
    loihi_x2_data_no_sat = \
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         15, 15, 15, 15, 7, 7, 7, 7, 4, 4, 4, 4, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    loihi_wgt_data_default_x = \
        [10, 10, 10, 10, 10, 10, 10, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
         18, 18, 18, 18, 18, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
         27, 27, 27, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
         37, 47, 47, 47, 47, 47, 47, 47, 47]
    loihi_wgt_data_default_y = \
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
         10, 10, 10, 10, 10, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
         19, 19, 19, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
         28, 28, 28, 28, 28, 36, 36, 36, 36]
    loihi_wgt_data_default_u = \
        [10, 10, 10, 10, 10, 10, 10, 13, 13, 13, 13, 15, 15, 15, 15, 17, 17, 17,
         17, 18, 18, 18, 18, 22, 22, 22, 22, 25, 25, 25, 25, 27, 27, 27, 27, 29,
         29, 29, 29, 33, 33, 33, 33, 36, 36, 36, 36, 38, 38, 38, 38, 40, 40, 40,
         40, 45, 45, 45, 45, 49, 49, 49, 49]
    loihi_wgt_data_overwrite_x = \
        [10, 10, 10, 10, 10, 10, 10, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
         21, 21, 21, 21, 21, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72,
         72, 72, 72, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77,
         77, 80, 80, 80, 80, 80, 80, 80, 80]
    loihi_wgt_data_overwrite_y = \
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
         10, 10, 10, 10, 10, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
         61, 61, 61, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66,
         66, 66, 66, 66, 66, 68, 68, 68, 68]
    loihi_wgt_data_overwrite_u = \
        [10, 10, 10, 10, 10, 10, 10, 14, 14, 14, 14, 17, 17, 17, 17, 19, 19, 19,
         19, 20, 20, 20, 20, 41, 41, 41, 41, 57, 57, 57, 57, 68, 68, 68, 68, 76,
         76, 76, 76, 78, 78, 78, 78, 80, 80, 80, 80, 81, 81, 81, 81, 82, 82, 82,
         82, 83, 83, 83, 83, 84, 84, 84, 84]
    loihi_wgt_data_sat_x = \
        [10, 10, 10, 10, 10, 10, 10, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
         21, 21, 21, 21, 21, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74,
         74, 74, 74, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92,
         92, 99, 99, 99, 99, 99, 99, 99, 99]
    loihi_wgt_data_sat_y = \
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
         10, 10, 10, 10, 10, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
         63, 63, 63, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79,
         79, 79, 79, 79, 79, 85, 85, 85, 85]
    loihi_wgt_data_sat_u = \
        [10, 10, 10, 10, 10, 10, 10, 14, 14, 14, 14, 17, 17, 17, 17, 19, 19, 19,
         19, 20, 20, 20, 20, 42, 42, 42, 42, 58, 58, 58, 58, 69, 69, 69, 69, 77,
         77, 77, 77, 85, 85, 85, 85, 91, 91, 91, 91, 95, 95, 95, 95, 98, 98, 98,
         98, 102, 102, 102, 102, 105, 105, 105, 105]
    loihi_wgt_data_no_sat_x = \
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
         10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
         10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
         10, 10, 10, 10, 10, 10, 10, 10, 10]
    loihi_wgt_data_no_sat_y = \
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
         10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
         10, 10, 10, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
         15, 15, 15, 15, 15, 18, 18, 18, 18]
    loihi_wgt_data_no_sat_u = \
        [10, 10, 10, 10, 10, 10, 10, 15, 15, 15, 15, 19, 19, 19, 19, 22, 22, 22,
         22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
         23, 23, 23, 25, 25, 25, 25, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29,
         29, 31, 31, 31, 31, 32, 32, 32, 32]

    def test_learning_graded_spike_default_bit_approx_x0_condition(self):
        """TODO : WRITE"""

        # TODO : REFACTOR
        expected_x1_data = self.expected_x1_data_default
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_default_x

        loihi_x1_data = self.loihi_x1_data_default
        loihi_x2_data = self.loihi_x2_data
        loihi_wgt_data = self.loihi_wgt_data_default_x

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "x"
        graded_spike_cfg = GradedSpikeCfg.DEFAULT

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

        self.print_plot(learning_rule_cnd, graded_spike_cfg, num_steps,
                        x1_data, x2_data, wgt_data,
                        expected_x1_data, expected_x2_data, expected_wgt_data,
                        loihi_x1_data, loihi_x2_data, loihi_wgt_data)

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_default_bit_approx_y0_condition(self):
        """TODO : WRITE"""

        # TODO : REFACTOR
        expected_x1_data = self.expected_x1_data_default
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_default_y

        loihi_x1_data = self.loihi_x1_data_default
        loihi_x2_data = self.loihi_x2_data
        loihi_wgt_data = self.loihi_wgt_data_default_y

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "y"
        graded_spike_cfg = GradedSpikeCfg.DEFAULT

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

        self.print_plot(learning_rule_cnd, graded_spike_cfg, num_steps,
                        x1_data, x2_data, wgt_data,
                        expected_x1_data, expected_x2_data, expected_wgt_data,
                        loihi_x1_data, loihi_x2_data, loihi_wgt_data)

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_default_bit_approx_u0_condition(self):
        """TODO : WRITE"""

        # TODO : REFACTOR
        expected_x1_data = self.expected_x1_data_default
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_default_u

        loihi_x1_data = self.loihi_x1_data_default
        loihi_x2_data = self.loihi_x2_data
        loihi_wgt_data = self.loihi_wgt_data_default_u

        num_steps = 63
        size = 1

        weights_init = np.eye(size) * 10
        learning_rule_cnd = "u"
        graded_spike_cfg = GradedSpikeCfg.DEFAULT

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

        self.print_plot(learning_rule_cnd, graded_spike_cfg, num_steps,
                        x1_data, x2_data, wgt_data,
                        expected_x1_data, expected_x2_data, expected_wgt_data,
                        loihi_x1_data, loihi_x2_data, loihi_wgt_data)

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_overwrite_bit_approx_x0_condition(self):
        """TODO : WRITE"""

        # TODO : REFACTOR
        expected_x1_data = self.expected_x1_data_overwrite
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_overwrite_x

        loihi_x1_data = self.loihi_x1_data_overwrite
        loihi_x2_data = self.loihi_x2_data
        loihi_wgt_data = self.loihi_wgt_data_overwrite_x

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

        self.print_plot(learning_rule_cnd, graded_spike_cfg, num_steps,
                        x1_data, x2_data, wgt_data,
                        expected_x1_data, expected_x2_data, expected_wgt_data,
                        loihi_x1_data, loihi_x2_data, loihi_wgt_data)

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_overwrite_bit_approx_y0_condition(self):
        """TODO : WRITE"""

        # TODO : REFACTOR
        expected_x1_data = self.expected_x1_data_overwrite
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_overwrite_y

        loihi_x1_data = self.loihi_x1_data_overwrite
        loihi_x2_data = self.loihi_x2_data
        loihi_wgt_data = self.loihi_wgt_data_overwrite_y

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

        self.print_plot(learning_rule_cnd, graded_spike_cfg, num_steps,
                        x1_data, x2_data, wgt_data,
                        expected_x1_data, expected_x2_data, expected_wgt_data,
                        loihi_x1_data, loihi_x2_data, loihi_wgt_data)

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_overwrite_bit_approx_u0_condition(self):
        """TODO : WRITE"""

        # TODO : REFACTOR
        expected_x1_data = self.expected_x1_data_overwrite
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_overwrite_u

        loihi_x1_data = self.loihi_x1_data_overwrite
        loihi_x2_data = self.loihi_x2_data
        loihi_wgt_data = self.loihi_wgt_data_overwrite_u

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

        self.print_plot(learning_rule_cnd, graded_spike_cfg, num_steps,
                        x1_data, x2_data, wgt_data,
                        expected_x1_data, expected_x2_data, expected_wgt_data,
                        loihi_x1_data, loihi_x2_data, loihi_wgt_data)

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_sat_bit_approx_x0_condition(self):
        """TODO : WRITE"""

        # TODO : REFACTOR
        expected_x1_data = self.expected_x1_data_sat
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_sat_x

        loihi_x1_data = self.loihi_x1_data_sat
        loihi_x2_data = self.loihi_x2_data
        loihi_wgt_data = self.loihi_wgt_data_sat_x

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

        self.print_plot(learning_rule_cnd, graded_spike_cfg, num_steps,
                        x1_data, x2_data, wgt_data,
                        expected_x1_data, expected_x2_data, expected_wgt_data,
                        loihi_x1_data, loihi_x2_data, loihi_wgt_data)

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_sat_bit_approx_y0_condition(self):
        """TODO : WRITE"""

        # TODO : REFACTOR
        expected_x1_data = self.expected_x1_data_sat
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_sat_y

        loihi_x1_data = self.loihi_x1_data_sat
        loihi_x2_data = self.loihi_x2_data
        loihi_wgt_data = self.loihi_wgt_data_sat_y

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

        self.print_plot(learning_rule_cnd, graded_spike_cfg, num_steps,
                        x1_data, x2_data, wgt_data,
                        expected_x1_data, expected_x2_data, expected_wgt_data,
                        loihi_x1_data, loihi_x2_data, loihi_wgt_data)

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_sat_bit_approx_u0_condition(self):
        """TODO : WRITE"""

        # TODO : REFACTOR
        expected_x1_data = self.expected_x1_data_sat
        expected_x2_data = self.expected_x2_data
        expected_wgt_data = self.expected_wgt_data_sat_u

        loihi_x1_data = self.loihi_x1_data_sat
        loihi_x2_data = self.loihi_x2_data
        loihi_wgt_data = self.loihi_wgt_data_sat_u

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

        self.print_plot(learning_rule_cnd, graded_spike_cfg, num_steps,
                        x1_data, x2_data, wgt_data,
                        expected_x1_data, expected_x2_data, expected_wgt_data,
                        loihi_x1_data, loihi_x2_data, loihi_wgt_data)

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_no_sat_bit_approx_x0_condition(self):
        """TODO : WRITE"""

        # TODO : REFACTOR
        expected_x1_data = self.expected_x1_data_no_sat
        expected_x2_data = self.expected_x2_data_no_sat
        expected_wgt_data = self.expected_wgt_data_no_sat_x

        loihi_x1_data = self.loihi_x1_data_no_sat
        loihi_x2_data = self.loihi_x2_data_no_sat
        loihi_wgt_data = self.loihi_wgt_data_no_sat_x

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

        self.print_plot(learning_rule_cnd, graded_spike_cfg, num_steps,
                        x1_data, x2_data, wgt_data,
                        expected_x1_data, expected_x2_data, expected_wgt_data,
                        loihi_x1_data, loihi_x2_data, loihi_wgt_data)

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_no_sat_bit_approx_y0_condition(self):
        """TODO : WRITE"""

        # TODO : REFACTOR
        expected_x1_data = self.expected_x1_data_no_sat
        expected_x2_data = self.expected_x2_data_no_sat
        expected_wgt_data = self.expected_wgt_data_no_sat_y

        loihi_x1_data = self.loihi_x1_data_no_sat
        loihi_x2_data = self.loihi_x2_data_no_sat
        loihi_wgt_data = self.loihi_wgt_data_no_sat_y

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

        self.print_plot(learning_rule_cnd, graded_spike_cfg, num_steps,
                        x1_data, x2_data, wgt_data,
                        expected_x1_data, expected_x2_data, expected_wgt_data,
                        loihi_x1_data, loihi_x2_data, loihi_wgt_data)

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_no_sat_bit_approx_u0_condition(self):
        """TODO : WRITE"""

        # TODO : REFACTOR
        expected_x1_data = self.expected_x1_data_no_sat
        expected_x2_data = self.expected_x2_data_no_sat
        expected_wgt_data = self.expected_wgt_data_no_sat_u

        loihi_x1_data = self.loihi_x1_data_no_sat
        loihi_x2_data = self.loihi_x2_data_no_sat
        loihi_wgt_data = self.loihi_wgt_data_no_sat_u

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

        self.print_plot(learning_rule_cnd, graded_spike_cfg, num_steps,
                        x1_data, x2_data, wgt_data,
                        expected_x1_data, expected_x2_data, expected_wgt_data,
                        loihi_x1_data, loihi_x2_data, loihi_wgt_data)

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)
