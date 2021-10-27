# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest

from lava.magma.core.decorator import implements, requires
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.model.model import AbstractProcessModel
from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.magma.core.resources import CPU, Loihi1NeuroCore, ECPU


class Decorators(unittest.TestCase):
    def test_implements(self):
        """Checks 'implements' decorator."""

        # Define minimal Process to be implemented
        class TestProc(AbstractProcess):
            pass

        # Define minimal Protocol to be implemented
        class TestProtocol(AbstractSyncProtocol):
            pass

        # Define minimal ProcModel that implements 'TestProc'
        @implements(proc=TestProc, protocol=TestProtocol)
        class TestModel(AbstractProcessModel):
            def run(self):
                pass

        # The 'implements' decorator adds class variables that allows
        # ProcessModels to be filtered by the compiler
        self.assertEqual(TestModel.implements_process, TestProc)
        self.assertEqual(TestModel.implements_protocol, TestProtocol)

    def test_implements_failing(self):
        """Checks failing 'implements' usage."""

        # Define minimal Process to be implemented
        class TestProc(AbstractProcess):
            pass

        # Define minimal Protocol to be implemented
        class TestProtocol(AbstractSyncProtocol):
            pass

        # We must pass a class, not an instance or anything else
        with self.assertRaises(TypeError):
            @implements(proc=TestProc(), protocol=TestProtocol)  # type: ignore
            class TestModel(AbstractProcessModel):  # type: ignore
                def run(self):
                    pass

        # Same for 'protocol'
        with self.assertRaises(TypeError):
            @implements(proc=TestProc, protocol=TestProtocol())  # type: ignore
            class TestModel2(AbstractProcessModel):
                def run(self):
                    pass

        # And we can only decorate a subclass of 'AbstractProcessModel'
        with self.assertRaises(AssertionError):
            @implements(proc=TestProc, protocol=TestProtocol)
            class TestProcess2(AbstractProcess):
                pass

    def test_implements_subclassing(self):
        """Check that 'implements' can also only be called on sub classes."""

        # Define minimal Process to be implemented
        class TestProc(AbstractProcess):
            pass

        # Define two minimal SyncProtocol
        class TestProtocol(AbstractSyncProtocol):
            pass

        # A process model that serves as a base class may only specify 'proc'
        # or 'protocol'
        @implements(proc=TestProc)
        class TestModel(AbstractProcessModel):
            def run(self):
                pass

        # In this case, 'implements_process' will be set but
        # 'implements_protocol' will not be set
        self.assertEqual(TestModel.implements_process, TestProc)
        self.assertEqual(TestModel.implements_protocol, None)

        @implements(protocol=TestProtocol)
        class SubTestModel(TestModel):
            pass

        # Finally both class attributes will be set
        self.assertEqual(SubTestModel.implements_process, TestProc)
        self.assertEqual(SubTestModel.implements_protocol, TestProtocol)

        # ...but attributes of parent class have not changed
        self.assertEqual(TestModel.implements_process, TestProc)
        self.assertEqual(TestModel.implements_protocol, None)

    def test_implements_subclassing_due_to_overwrite(self):
        """Check that we cannot overwrite an already set Process or
        SyncProtocol class."""

        # Define minimal Process to be implemented
        class TestProc(AbstractProcess):
            pass

        # Define two minimal SyncProtocol
        class TestProtocol1(AbstractSyncProtocol):
            pass

        class TestProtocol2(AbstractSyncProtocol):
            pass

        # A new ProcessModel might implement TestProc with TestProtocol1
        @implements(proc=TestProc, protocol=TestProtocol1)
        class TestModel(AbstractProcessModel):
            def run(self):
                pass

        # Attempting to overwrite either of the 'proc' or 'protocol'
        # attributes must fail
        with self.assertRaises(AssertionError):
            @implements(protocol=TestProtocol2)
            class SubTestModel(TestModel):
                pass

    def test_requires(self):
        """Checks 'requires' decorator."""

        # Define minimal ProcModel that requires a single 'AbstractResource'
        @requires(CPU)
        class TestModel1(AbstractProcessModel):
            def run(self):
                pass

        # The 'requires' decorator adds a class variable to informs the
        # Compiler which compute resources are required
        self.assertEqual(TestModel1.required_resources, [CPU])

        # There can be multiple requirements...
        @requires(CPU, Loihi1NeuroCore)
        class TestModel2(AbstractProcessModel):
            def run(self):
                pass

        # ... in which case we expect a list of supported devices
        self.assertEqual(TestModel2.required_resources, [CPU, Loihi1NeuroCore])

        # We can also require a CPU and one of the elements in the list
        @requires([Loihi1NeuroCore, ECPU], CPU)
        class TestModel3(AbstractProcessModel):
            def run(self):
                pass

        self.assertEqual(
            TestModel3.required_resources, [[Loihi1NeuroCore, ECPU], CPU]
        )

    def test_requires_subclassing(self):
        """Checks that requirements can be added for sub classes."""

        # Define minimal ProcModel that requires a single 'AbstractResource'
        @requires(CPU)
        class TestModel(AbstractProcessModel):
            def run(self):
                pass

        # This adds CPU to the list of required resources
        self.assertEqual(TestModel.required_resources, [CPU])

        # Sub classes can add further requirements
        @requires(Loihi1NeuroCore)
        class SubTestModel(TestModel):
            pass

        # This adds Loihi1NeuroCore to the list of requirements...
        self.assertEqual(SubTestModel.required_resources,
                         [CPU, Loihi1NeuroCore])

        # ...but does not modify requirements of parent class
        self.assertEqual(TestModel.required_resources, [CPU])

    def test_requires_failing(self):
        """Checks failing 'requires' usage."""

        # We must decorate a ProcessModel and nothing else:
        with self.assertRaises(AssertionError):
            @requires(CPU)
            class Something(AbstractProcess):
                pass

        # We must decorate a ProcessModel with an 'AbstractResource' class
        # and nothing else
        with self.assertRaises(TypeError):

            @requires(CPU())  # type: ignore
            class TestModel(AbstractProcessModel):
                def run(self):
                    pass


if __name__ == "__main__":
    unittest.main()
