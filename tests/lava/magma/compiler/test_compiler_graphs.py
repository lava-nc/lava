"""Unit tests for the `compiler_graphs` module.

Currently only includes tests for changes to the process model search process.

"""

import sys
import os
import unittest

from lava.proc.lif.process import LIF
from lava.proc.lif.models import (
    PyLifModelFloat, PyLifModelBitAcc
)

# make sure parent dir is in python path
if os.path.dirname(__file__) not in sys.path:
    sys.path.append(os.path.dirname(__file__))

from lava.magma.compiler.compiler_graphs import ProcGroupDiGraphs

from test_package.proc.test.process import TestProcess
from test_package.proc.test.models_absolute import TestModelAbsolute
from test_package.proc.test.models_relative import TestModelRelative


class TestProcGroupDiGraphs(unittest.TestCase):
    """Testing ProcGroupDiGraphs.
    
    Currently only tests `ProcGroupDiGraphs._find_proc_models()`
    """
    def test_find_proc_models_custom_proc(self):
        """Test process model finding process."""
        proc = TestProcess(name="test")

        proc_models = ProcGroupDiGraphs._find_proc_models(proc)
        expected_models = [TestModelAbsolute, TestModelRelative]

        print(f"{proc_models=}")
        print(f"{expected_models=}")

        self.assertTrue(all(pm in proc_models for pm in expected_models))

    def test_find_proc_models_lava_proc(self):
        """Test process model finding for a standard lava process."""
        lif = LIF(shape=(1,), name='lif')

        proc_models = ProcGroupDiGraphs._find_proc_models(lif)
        expected_models = [PyLifModelFloat, PyLifModelBitAcc]

        self.assertTrue(all(pm in proc_models for pm in expected_models))


if __name__ == "__main__":
    unittest.main()
