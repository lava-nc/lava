# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest


class TestFrameworks(unittest.TestCase):
    """Tests for framework import."""

    def test_frameworks_loihi2_import(self):
        """Tests if framework import fails."""
        import lava.frameworks.loihi2 as lv


if __name__ == '__main__':
    unittest.main()
