# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import importlib.util


class TestInitImports(unittest.TestCase):

    def test_lava_init_file_imports_lif_class(self) -> None:
        module_spec = importlib.util.find_spec("lava")
        lava_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(lava_module)

        lif_importable = hasattr(lava_module, "LIF")
        self.assertTrue(lif_importable)
