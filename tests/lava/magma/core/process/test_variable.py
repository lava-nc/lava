# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var, VarServer


class TestVarInitialization(unittest.TestCase):
    def setUp(self):
        VarServer().reset_server()

    def test_constructor(self):
        """Check initialization of Var."""

        v1 = Var(shape=(1, 2, 3), init=10, shareable=False)
        self.assertIsInstance(v1, Var)

        # Make sure input arguments have been set properly
        self.assertEqual(v1.shape, (1, 2, 3))
        self.assertEqual(v1.name, "Unnamed variable")
        self.assertEqual(v1.id, 0)
        self.assertEqual(v1.init, 10)
        self.assertEqual(v1.shareable, False)

        # Also check that VarServer increments id for each new Var
        v2 = Var(shape=(1,))
        self.assertEqual(v2.id, 1)

        v3 = Var(shape=(1,))
        self.assertEqual(v3.id, 2)

    def test_alias(self):
        """Checks definition of 'alias' relationship between variables.

        Vars of a parent process can alias a Var of a sub process in order to
        expose a sub process variable at the level of the parent process.

        Normally, this happens within a SubProcessModel. Here we
        """

        # Let's create two processes with variables
        class Proc1(AbstractProcess):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.var1 = Var((1,))
                self.var2 = Var((1, 2), shareable=True)

        class Proc2(AbstractProcess):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.var1 = Var((1,))
                self.var2 = Var((1, 2))
                self.var3 = Var((1, 2), shareable=False)

        class Proc3(AbstractProcess):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.var1 = Var((1,))

        # Here we manually make an instance of Proc2 a sub process of Proc1
        p1 = Proc1()
        p2 = Proc2()
        p1.register_sub_procs({p2.name: p2})

        # Assuming that p2 was instantiated as a sub processes of p1, we can
        # define alias relationships between Vars of p1 and Vars of p2.
        # The 'shape' and 'shareable' attribute of both Vars must match
        p1.var1.alias(p2.var1)
        self.assertEqual(p1.var1.aliased_var, p2.var1)
        p1.var2.alias(p2.var2)
        self.assertEqual(p1.var2.aliased_var, p2.var2)

        # If any of them don't match then aliasing must fail
        with self.assertRaises(AssertionError):
            p1.var1.alias(p2.var2)
        with self.assertRaises(AssertionError):
            p1.var2.alias(p2.var3)

        # In addition, it is not possible to alias Vars if there is not sub
        # process relationship between the parent processes of the Vars even
        # if their 'shape' or 'shareable' attributes match
        p3 = Proc3()
        p1.var1.alias(p3.var1)
        with self.assertRaises(AssertionError):
            p1.validate_var_aliases()

    def test_set_get_without_Runtime(self):
        """Check setting of 'Var' value before Runtime has been initialized."""

        # Crete a minimal process with Vars
        class Proc(AbstractProcess):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.var1 = Var((1,), init=10)
                self.var2 = Var((1, 2), init=[20, 30])

        p = Proc()

        # Before a process has been compiled or run, no Runtime has been
        # assigned to a process. Therefore, getting a Var value will only
        # return the initial value:
        self.assertEqual(p.var1.get(), 10)
        self.assertEqual(p.var2.get(), [20, 30])
        # Otherwise it would return the value broadcast to the full shape of
        # the Var from the ProcessModel created at runtime.

        # However, setting a Var before a Runtime has been assigned is not
        # possible because the compiler has already pulled the initial Var
        # value to create the ProcessBuilder which would now ignore any new
        # values assigned to the initial value of the Var.
        import numpy as np
        with self.assertRaises(ValueError):
            p.var1.set(np.ones((1,)))
        # In the future we could upgrade the ProcessBuilder to get the latest
        # Var.init values before deployment to reflect any last minute
        # changes to Var values by the user.


if __name__ == "__main__":
    unittest.main()
