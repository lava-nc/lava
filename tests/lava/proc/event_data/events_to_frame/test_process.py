# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.proc.event_data.events_to_frame.process import EventsToFrame


class TestProcessEventsToFrame(unittest.TestCase):
    def test_init(self) -> None:
        """Tests instantiation of SparseToDense for a 3D output."""
        to_frame = EventsToFrame(shape_in=(43200,),
                                 shape_out=(240, 180, 1))

        self.assertIsInstance(to_frame, EventsToFrame)

    def test_invalid_shape_in_throws_exception(self) -> None:
        """Tests whether an exception is thrown if shape_in is not of the
        form (n,)."""
        invalid_shape_in = (43200, 1)
        with(self.assertRaises(ValueError)):
            EventsToFrame(shape_in=invalid_shape_in,
                          shape_out=(240, 180, 1))

    def test_negative_size_shape_in_throws_exception(self) -> None:
        """Tests whether an exception is thrown when a negative integer for
        the shape_in argument is given."""
        invalid_shape_in = (-43200,)
        with(self.assertRaises(ValueError)):
            EventsToFrame(shape_in=invalid_shape_in,
                          shape_out=(240, 180))

    def test_invalid_shape_out_dimensionality_throws_exception(self) -> None:
        """Tests whether an exception is thrown when the dimensionality of the
        shape_out parameter is not 3."""
        invalid_shape_out = (240, 180)
        with(self.assertRaises(ValueError)):
            EventsToFrame(shape_in=(43200,),
                          shape_out=invalid_shape_out)

    def test_invalid_number_of_channels_throws_exception(self) -> None:
        """Tests whether an exception is thrown when the number of channels is
        larger than 2."""
        invalid_channel_size = 3
        with(self.assertRaises(ValueError)):
            EventsToFrame(shape_in=(43200,),
                          shape_out=(240, 180, invalid_channel_size))

    def test_negative_width_in_shape_out_throws_exception(self) -> None:
        """Tests whether an exception is thrown when a negative width is
        specified for the shape_out parameter."""
        invalid_width = -240
        with(self.assertRaises(ValueError)):
            EventsToFrame(shape_in=(43200,),
                          shape_out=(invalid_width, 180))

    def test_negative_height_in_shape_out_throws_exception(self) -> None:
        """Tests whether an exception is thrown when a negative height is
        specified for the shape_out parameter."""
        invalid_height = -180
        with(self.assertRaises(ValueError)):
            EventsToFrame(shape_in=(43200,),
                          shape_out=(240, invalid_height))


if __name__ == '__main__':
    unittest.main()
