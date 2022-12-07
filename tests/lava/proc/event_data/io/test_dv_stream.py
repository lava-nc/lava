# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from dv import AedatFile
from dv.AedatFile import _AedatFileEventNumpyPacketIterator
import numpy as np
import typing as ty
import unittest

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.proc.event_data.io.dv_stream import DvStream, DvStreamPM


class TestProcessDvStream(unittest.TestCase):
    def test_init(self):
        """
        Tests instantiation of AedatDataLoader.
        """

        stream = DvStream(address="127.0.0.1",
                          port=7777,
                          shape_out=(43200,))

        self.assertIsInstance(stream, DvStream)
        self.assertEqual((43200,), stream.out_port.shape)

    def test_invalid_shape_throws_exception(self):
        """
        Tests whether a shape_out argument with an invalid shape throws an exception.
        """
        with(self.assertRaises(ValueError)):
            DvStream(address="127.0.0.1",
                     port=7777,
                     shape_out=(240, 180))

    def test_negative_size_throws_exception(self):
        """
        Tests whether a shape_out argument with a negative size throws an exception.
        """
        with(self.assertRaises(ValueError)):
            DvStream(address="127.0.0.1",
                     port=7777,
                     shape_out=(-240,))

    def test_negative_port_throws_exception(self):
        """
        Tests whether a port argument with a negative size throws an exception.
        """
        with(self.assertRaises(ValueError)):
            DvStream(address="127.0.0.1",
                     port=-7777,
                     shape_out=(43200,))

    def test_port_out_of_range_throws_exception(self):
        """
             Tests whether a port argument that is out of range throws an error.
        """
        with(self.assertRaises(ValueError)):
            DvStream(address="127.0.0.1",
                     port=7777777,
                     shape_out=(43200,))

    def test_address_empty_string_throws_exception(self):
        with(self.assertRaises(ValueError)):
            DvStream(address="",
                     port=7777,
                     shape_out=(43200,))

class TestProcessModelDvStream(unittest.TestCase):
    def test_init(self):
        """
        Tests instantiation of the DvStream process model.
        """
        proc_params = {
            "address": "127.0.0.1",
            "port": 7777,
            "shape_out": (43200,),
        }

        pm = DvStreamPM(proc_params)

        self.assertIsInstance(pm, DvStreamPM)

    def test_run_spike(self):
        class PacketInput(ty.Protocol):
            def __next__(self):
                ...

        class MockPacketInput:
            def __next__(self):
                return {
                    "x": 35,
                    "y": 35,
                    "polarity": 0,
                }


        proc_params = {
            "address": "127.0.0.1",
            "port": 7777,
            "shape_out": (43200,),
            "_event_stream": MockPacketInput()

        }

        pm = DvStreamPM(proc_params)
        pm.run_spk()




if __name__ == '__main__':
    unittest.main()
