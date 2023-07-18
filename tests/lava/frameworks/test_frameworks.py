# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np


class TestFrameworks(unittest.TestCase):
    def test_frameworks_loihi2_import(self):
        import lava.frameworks.loihi2 as lv


if __name__ == '__main__':
    unittest.main()
