# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import unittest

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.proc.event_data.events_to_frame.process import EventsToFrame
from lava.proc.event_data.events_to_frame.models import PyEventsToFramePM

from ..utils import RecvDense, SendSparse


class TestProcessModelEventsToFrame(unittest.TestCase):
    def test_init(self):
        """Tests instantiation of the EventsToFrame ProcessModel."""
        pm = PyEventsToFramePM()
        self.assertIsInstance(pm, PyEventsToFramePM)

    def test_convert_unary_polarity_events_to_frame(self) -> None:
        """Tests whether the EventsToFrame ProcessModel correctly converts
        event-based data with unary polarity to a frame-based
        representation."""
        data = np.array([1, 1, 1, 1, 1, 1])
        xs = [0, 1, 2, 1, 2, 4]
        ys = [0, 2, 1, 5, 7, 7]
        indices = np.ravel_multi_index((xs, ys), (8, 8))

        expected_data = np.zeros((8, 8, 1))
        for x, y, p in zip(xs, ys, data):
            expected_data[x, y, 0] = p

        send_sparse = SendSparse(shape=(10,), data=data, indices=indices)
        to_frame = EventsToFrame(shape_in=(10,),
                                 shape_out=(8, 8, 1))
        recv_dense = RecvDense(shape=(8, 8, 1))

        send_sparse.out_port.connect(to_frame.in_port)
        to_frame.out_port.connect(recv_dense.in_port)

        to_frame.run(condition=RunSteps(num_steps=1),
                     run_cfg=Loihi1SimCfg())

        sent_and_received_data = recv_dense.data.get()

        to_frame.stop()

        np.testing.assert_equal(sent_and_received_data,
                                expected_data)

    def test_convert_binary_polarity_events_to_frame(self) -> None:
        """Tests whether the EventsToFrame ProcessModel correctly converts
        event-based data with binary polarity to a frame-based
        representation."""
        data = np.array([1, 0, 1, 0, 1, 0])
        xs = [0, 1, 2, 1, 2, 4]
        ys = [0, 2, 1, 5, 7, 7]
        indices = np.ravel_multi_index((xs, ys), (8, 8))

        expected_data = np.zeros((8, 8, 2))
        for x, y, p in zip(xs, ys, data):
            expected_data[x, y, p] = 1

        send_sparse = SendSparse(shape=(10,), data=data, indices=indices)
        to_frame = EventsToFrame(shape_in=(10,),
                                 shape_out=(8, 8, 2))
        recv_dense = RecvDense(shape=(8, 8, 2))

        send_sparse.out_port.connect(to_frame.in_port)
        to_frame.out_port.connect(recv_dense.in_port)

        to_frame.run(condition=RunSteps(num_steps=1),
                     run_cfg=Loihi1SimCfg())

        sent_and_received_data = recv_dense.data.get()

        to_frame.stop()

        np.testing.assert_equal(sent_and_received_data,
                                expected_data)


if __name__ == '__main__':
    unittest.main()
