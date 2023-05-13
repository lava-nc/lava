# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
from unittest.mock import MagicMock, patch

from lava.utils import lava_loihi


class TestLavaLoihi(unittest.TestCase):
    def test_on_known_installed_lava_module(self) -> None:
        """Tests whether the method for checking on an installed library
        works in general."""
        self.assertTrue(lava_loihi.is_installed("lava"))

    @patch("importlib.util.find_spec")
    def test_is_installed(self, find_spec) -> None:
        """Tests whether the function detects when Lava-Loihi is installed."""
        find_spec.return_value = \
            MagicMock(spec="importlib.machinery.ModuleSpec")

        self.assertTrue(lava_loihi.is_installed())

    @patch("importlib.util.find_spec")
    def test_is_not_installed(self, find_spec) -> None:
        """Tests whether the function detects when Lava-Loihi is not
        installed."""
        find_spec.return_value = None

        self.assertFalse(lava_loihi.is_installed())
