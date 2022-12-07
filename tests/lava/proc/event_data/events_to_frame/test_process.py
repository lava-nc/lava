# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.proc.event_data.events_to_frame.process import EventsToFrame


class TestProcessEventsToFrame(unittest.TestCase):
    def test_init(self):
        """Tests instantiation of SparseToDense for a 3D output."""
        to_frame = EventsToFrame(shape_in=(43200,),
                                 shape_out=(240, 180, 1))

        self.assertIsInstance(to_frame, EventsToFrame)

    def test_invalid_shape_in_throws_exception(self):
        """Tests whether a shape_in argument that isn't (n,) throws
        an exception."""
        with(self.assertRaises(ValueError)):
            EventsToFrame(shape_in=(43200, 1),
                          shape_out=(240, 180, 1))

    def test_invalid_shape_out_throws_exception(self):
        """Tests whether an exception is thrown when a 1d or 4d value
        for the shape_out argument is given."""
        with(self.assertRaises(ValueError)):
            EventsToFrame(shape_in=(43200,),
                          shape_out=(240,))

        with(self.assertRaises(ValueError)):
            EventsToFrame(shape_in=(43200,),
                          shape_out=(240, 180, 3))

        with(self.assertRaises(ValueError)):
            EventsToFrame(shape_in=(43200,),
                          shape_out=(240, 180, 2, 1))

    def test_negative_size_shape_in_throws_exception(self):
        """Tests whether an exception is thrown when a negative integer for
        the shape_in argument is given"""
        with(self.assertRaises(ValueError)):
            EventsToFrame(shape_in=(-43200,),
                          shape_out=(240, 180))

    def test_negative_width_or_height_shape_out_throws_exception(self):
        """Tests whether an exception is thrown when a negative width or height
        for the shape_out argument is given"""
        with(self.assertRaises(ValueError)):
            EventsToFrame(shape_in=(43200,),
                          shape_out=(-240, 180))

        with(self.assertRaises(ValueError)):
            EventsToFrame(shape_in=(43200,),
                          shape_out=(240, -180))


if __name__ == '__main__':
    unittest.main()
