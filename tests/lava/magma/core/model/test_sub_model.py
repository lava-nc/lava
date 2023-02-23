# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.model.sub.model import AbstractSubProcessModel


class TestSubProcModel(unittest.TestCase):
    def test_constructor(self):
        """Check SubProcModel creation."""

        # A minimal process
        class Proc(AbstractProcess):
            pass

        # We cannot instantiate the AbstractSubProcessModel directly
        with self.assertRaises(TypeError):
            AbstractSubProcessModel(Proc())

        # But we can instantiate a sub class
        class SubProcModel(AbstractSubProcessModel):
            def __init__(self, _):
                pass

        pm = SubProcModel(Proc())

        self.assertIsInstance(pm, AbstractSubProcessModel)

    def test_find_sub_procs(self):
        """Checks finding of sub processes within a SubProcModel."""

        # A minimal process
        class Proc(AbstractProcess):
            pass

        # A minimal SubProcessModel
        class SubProcModel(AbstractSubProcessModel):
            def __init__(self, _):
                self.proc1 = Proc()
                self.proc2 = Proc()

        # Normally, the Compiler would try to find any sub processes of a
        # SubProcModel
        pm = SubProcModel(Proc())
        sub_procs = pm.find_sub_procs()

        self.assertEqual(list(sub_procs.values()), [pm.proc1, pm.proc2])


if __name__ == "__main__":
    unittest.main()
