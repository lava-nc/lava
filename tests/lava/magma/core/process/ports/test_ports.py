# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.magma.core.process.ports.exceptions import (
    ReshapeError,
    DuplicateConnectionError,
    ConcatShapeError,
    ConcatIndexError,
    TransposeShapeError,
    TransposeIndexError,
    VarNotSharableError,
)
from lava.magma.core.process.ports.ports import (
    InPort,
    OutPort,
    RefPort,
    VarPort,
    ConcatPort,
    TransposePort,
)
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var


class TestPortInitialization(unittest.TestCase):
    def test_constructor(self):
        """Check initialization of Ports."""

        in_port = InPort(shape=(1, 2, 3))
        out_port = OutPort(shape=(3, 2, 1))
        ref_port = RefPort(shape=(1, 1, 1))
        self.assertIsInstance(in_port, InPort)
        self.assertIsInstance(out_port, OutPort)
        self.assertIsInstance(ref_port, RefPort)

        self.assertEqual(in_port.shape, (1, 2, 3))
        self.assertEqual(out_port.shape, (3, 2, 1))
        self.assertEqual(ref_port.shape, (1, 1, 1))


class TestIOPorts(unittest.TestCase):
    """Normally ports will only ever be created and used within a parent
    process. However, the tests around establishing connections between ports
    in this TestCase do not require the parent process to exist."""

    def test_connect_OutPort_to_InPort(self):
        """Check connecting OutPort directly to InPort."""

        op = OutPort((1, 2, 3))
        ip = InPort((1, 2, 3))

        # Normally an OutPort connects to another InPort
        op.connect(ip)

        # This registers ip as an 'out_connection' of 'op' and 'op' as an
        # 'in_connection' of 'ip'.
        self.assertEqual(op.out_connections, [ip])
        self.assertEqual(ip.in_connections, [op])

    def test_connect_OutPort_to_InPort_with_different_shape(self):
        """Check that connecting ports with different shape fails."""

        # Create two ports with incompatible shapes
        op = OutPort((1, 2, 3))
        ip = InPort((3, 2, 1))

        # In this case, connecting them must fail
        with self.assertRaises(AssertionError):
            op.connect(ip)

    def test_connect_OutPort_to_many_InPorts(self):
        """Check connecting OutPort directly to multiple InPorts."""

        op1 = OutPort((1, 2, 3))
        op2 = OutPort((1, 2, 3))
        ip1 = InPort((1, 2, 3))
        ip2 = InPort((1, 2, 3))

        # An OutPort can also connect to multiple InPorts
        # Either at once...
        op1.connect([ip1, ip2])

        # ... or consecutively
        op2.connect(ip1)
        op2.connect(ip2)

        # The established connections are the same in both cases
        self.assertEqual(op1.out_connections, [ip1, ip2])
        self.assertEqual(op2.out_connections, [ip1, ip2])
        self.assertEqual(ip1.in_connections, [op1, op2])
        self.assertEqual(ip2.in_connections, [op1, op2])

    def test_duplicate_connections(self):
        """Check that connecting the same ports more than once fails."""

        op = OutPort((1, 2, 3))
        ip1 = InPort((1, 2, 3))
        ip2 = InPort((1, 2, 3))

        # We can connect ports once
        op.connect([ip1, ip2])

        # But attempting to connect the same ports twice must fail
        with self.assertRaises(DuplicateConnectionError):
            op.connect(ip2)

    def test_legal_chain_of_connections(self):
        """Check that OutPort to OutPort and InPort to InPort connections can
        can be chained together as would happen in hierarchical processes."""

        # In hierarchical processes, OutPorts of a nested process might
        # connect to OutPorts of parent process. Similarly InPorts of a
        # parent process might connect to InPorts of nested processes.
        op1 = OutPort((1, 2, 3))
        op2 = OutPort((1, 2, 3))
        ip1 = InPort((1, 2, 3))
        ip2 = InPort((1, 2, 3))
        # (op1/ip1 might belong to a nested process of the parent process of
        # op2/ip2)

        # Connect ports in sequence: op1 -> op2 -> ip2 -> ip1
        op1.connect(op2)
        op2.connect(ip2)
        ip2.connect(ip1)

        # Each port on the right is an output to the port on the left
        self.assertEqual(op1.out_connections, [op2])
        self.assertEqual(op2.out_connections, [ip2])
        self.assertEqual(ip2.out_connections, [ip1])

        # Vice versa, each port on the left is an input to the port on the right
        self.assertEqual(op2.in_connections, [op1])
        self.assertEqual(ip2.in_connections, [op2])
        self.assertEqual(ip1.in_connections, [ip2])

    def test_illegal_chain_of_connections(self):
        """Checks that InPorts cannot connect to other OutPorts, i.e. in a
        chain of connections."""

        op = OutPort((1, 2, 3))
        ip1 = InPort((1, 2, 3))
        ip2 = InPort((1, 2, 3))

        # An InPort may connect to a (nested) InPort
        ip1.connect(ip2)

        # But it must not connect to another OutPort
        with self.assertRaises(AssertionError):
            ip1.connect(op)  # type: ignore

    def test_legal_chain_of_connections_with_connect_from(self):
        """Checks that 'connect_from(..)' behaves analogous to 'connect(..)'."""

        # Create a bunch of ports
        op1 = OutPort((1, 2, 3))
        op2 = OutPort((1, 2, 3))
        ip1 = InPort((1, 2, 3))
        ip2 = InPort((1, 2, 3))
        # (op1/ip1 might belong to a nested process of the parent process of
        # op2/ip2)

        # In contrast to the implied left-to-right directionality of the
        # 'connect(..)' method, the 'connect_from(..)' method assumes an
        # inverse left-from-right or analogously right-to-left directionality.
        # Connect ports in sequence: op1 -> op2 -> ip2 -> ip1
        op2.connect_from(op1)
        ip2.connect_from(op2)
        ip1.connect_from(ip2)

        # In turn, each port on the right is an input to the port on the left
        self.assertEqual(op2.in_connections, [op1])
        self.assertEqual(ip2.in_connections, [op2])
        self.assertEqual(ip1.in_connections, [ip2])

        # Vice versa, each port on the left is an output to the port on the
        # right
        self.assertEqual(op1.out_connections, [op2])
        self.assertEqual(op2.out_connections, [ip2])
        self.assertEqual(ip2.out_connections, [ip1])

    def test_getting_src_and_dst_connections(self):
        """Check that source and destination ports can be retrieved from a
        chain of connections."""

        # Create a bunch of ports
        op1 = OutPort((1, 2, 3))
        op2 = OutPort((1, 2, 3))
        op3 = OutPort((1, 2, 3))
        ip1 = InPort((1, 2, 3))
        ip2 = InPort((1, 2, 3))
        ip3 = InPort((1, 2, 3))

        # Connect them in a way that each source has multiple destinations
        # and each destination port as a multiple input connections
        op3.connect_from([op1, op2])
        op3.connect(ip3)
        ip3.connect([ip1, ip2])

        # op1 connects ultimately to ip1 and ip2
        self.assertEqual(op1.get_dst_ports(), [ip1, ip2])
        # ip2 ultimately receives inputs from op1 and op2
        self.assertEqual(ip2.get_src_ports(), [op1, op2])


class TestRVPorts(unittest.TestCase):
    """RefPorts and VarPorts enable shared memory access from a RefPort's
    parent process to the VarPort's parent process. They both inherit most of
    their implementation from AbstractPort."""

    def test_connect_RefPort_to_VarPort(self):
        """Checks connecting RefPort explicitly to Var via VarPort"""

        # In order to expose a Var explicitly in a Process, one should
        # explicitly create a VarPort from a Var within the Process constructor.
        v = Var((1, 2, 3))
        vp = VarPort(v)
        # (This is normally good practice to make users aware of potential
        # side effects of shared memory interaction.)

        # Next, a RefPort can connect to the exposed VarPort
        rp = RefPort((1, 2, 3))
        rp.connect(vp)

        # This makes the VarPort the destination port of the RefPort...
        self.assertEqual(rp.get_dst_ports(), [vp])
        # ... and the VarPort holds a reference to the wrapped Var
        self.assertEqual(vp.var, v)
        # The destination Var can also be obtained directly
        self.assertEqual(rp.get_dst_vars(), [v])

    def test_connect_RefPort_to_Var(self):
        """Checks connecting RefPort implicitly to Var."""

        # One can also create a Var and RefPort...
        v = Var((1, 2, 3))
        rp = RefPort((1, 2, 3))

        # ...but then connect them directly via connect_var(..)
        rp.connect_var(v)

        # This has the same effect as connecting a RefPort explicitly via a
        # VarPort to a Var...
        self.assertEqual(rp.get_dst_vars(), [v])
        # ... but still creates a VarPort implicitly
        vp = rp.get_dst_ports()[0]
        self.assertIsInstance(vp, VarPort)
        # ... which wraps the original Var
        self.assertEqual(vp.var, v)

        # In this case, the VarPort inherits its name and parent process from
        # the Var it wraps
        self.assertEqual(vp.name, "_" + v.name + "_implicit_port")
        # (We can't check for the same parent process here because it has not
        # been assigned to the Var yet)

    def test_connect_RefPort_to_Var_process(self):
        """Checks connecting RefPort implicitly to Var, with registered
        processes."""

        # Create a mock parent process
        class VarProcess(AbstractProcess):
            ...

        # Create a Var and RefPort...
        v = Var((1, 2, 3))
        rp = RefPort((1, 2, 3))

        # ...register a process for the Var
        v.process = VarProcess()

        # ...then connect them directly via connect_var(..)
        rp.connect_var(v)

        # This has the same effect as connecting a RefPort explicitly via a
        # VarPort to a Var...
        self.assertEqual(rp.get_dst_vars(), [v])
        # ... but still creates a VarPort implicitly
        vp = rp.get_dst_ports()[0]
        self.assertIsInstance(vp, VarPort)
        # ... which wraps the original Var
        self.assertEqual(vp.var, v)

        # In this case, the VarPort inherits its name and parent process from
        # the Var it wraps
        self.assertEqual(vp.name, "_" + v.name + "_implicit_port")
        self.assertEqual(vp.process, v.process)

    def test_connect_RefPort_to_Var_process_conflict(self):
        """Checks connecting RefPort implicitly to Var, with registered
        processes and conflicting names. -> adding _k with k=1,2,3... to
        the name."""

        # Create a mock parent process
        class VarProcess(AbstractProcess):
            # Attribute is named like our implicit VarPort after creation
            _existing_attr_implicit_port = None

        # Create a Var and RefPort...
        v = Var((1, 2, 3))
        rp = RefPort((1, 2, 3))

        # ...register a process for the Var and name it so it conflicts with
        # the attribute of VarProcess (very unlikely to happen)
        v.process = VarProcess()
        v.name = "existing_attr"

        # ... and connect it directly via connect_var(..)
        rp.connect_var(v)

        # This has the same effect as connecting a RefPort explicitly via a
        # VarPort to a Var...
        self.assertEqual(rp.get_dst_vars(), [v])
        # ... but still creates a VarPort implicitly
        vp = rp.get_dst_ports()[0]
        self.assertIsInstance(vp, VarPort)
        # ... which wraps the original Var
        self.assertEqual(vp.var, v)

        # In this case, the VarPort inherits its name and parent process from
        # the Var it wraps + _implicit_port + _k with k=1,2,3...
        self.assertEqual(vp.name, "_" + v.name + "_implicit_port" + "_1")
        self.assertEqual(vp.process, v.process)

    @unittest.skip("Currently not supported")
    def test_connect_RefPort_to_many_Vars(self):
        """Checks that RefPort can be connected to many Vars."""

        # We can have multiple Vars...
        v1 = Var((1, 2, 3))
        v2 = Var((1, 2, 3))
        # ...and connect a RefPort to them
        rp = RefPort((1, 2, 3))
        rp.connect_var([v1, v2])

        # The RefPort will have two destination Vars...
        self.assertEqual(rp.get_dst_vars(), [v1, v2])
        # ...and two unique implicit VarPorts
        vps = rp.get_dst_ports()
        self.assertIsInstance(vps[0], VarPort)
        self.assertIsInstance(vps[1], VarPort)
        self.assertNotEqual(vps[0], vps[1])

    def test_connect_RefPort_to_Var_with_incompatible_shape(self):
        """Checks that shapes must be compatible when connecting ports."""

        # There can be variables with different shapes...
        v1 = Var((1, 2, 3))
        v2 = Var((3, 2, 1))
        # ...but a RefPort can only connect if the shapes are compatible
        rp = RefPort((1, 1, 1))
        with self.assertRaises(AssertionError):
            rp.connect_var([v1, v2])

    def test_connect_RefPort_to_non_sharable_Var(self):
        """Check that RefPorts can only cannot to shareable Vars."""

        # By default, Vars are 'sharable', i.e. allow shared-memory access.
        # But this can also be disabled
        v = Var((1, 2, 3), shareable=False)

        # In this case, connecting a RefPort (or generating a VarPort)...
        rp = RefPort((1, 2, 3))
        # ...to a Var must fail
        with self.assertRaises(VarNotSharableError):
            rp.connect_var(v)

    def test_connect_RefPort_to_InPort_OutPort(self):
        """Checks connecting RefPort to an InPort or OutPort. -> TypeError"""

        # Create an InPort, OutPort, RefPort...
        ip = InPort((1, 2, 3))
        op = OutPort((1, 2, 3))
        rp = RefPort((1, 2, 3))

        # ... and connect them via connect(..)
        # The type conflict should raise an TypeError
        with self.assertRaises(TypeError):
            rp.connect(ip)

        with self.assertRaises(TypeError):
            rp.connect(op)

        # Connect them via connect_from(..)
        # The type conflict should raise an TypeError
        with self.assertRaises(TypeError):
            rp.connect_from(ip)

        with self.assertRaises(TypeError):
            rp.connect_from(op)

    def test_connect_VarPort_to_InPort_OutPort_RefPort(self):
        """Checks connecting VarPort to an InPort, OutPort or RefPort.
        -> TypeError (RefPort can only be connected via connect_from(..) to
        VarPort."""

        # Create an InPort, OutPort, RefPort, Var with VarPort...
        ip = InPort((1, 2, 3))
        op = OutPort((1, 2, 3))
        rp = RefPort((1, 2, 3))
        v = Var((1, 2, 3))
        vp = VarPort(v)

        # ... and connect them via connect(..)
        # The type conflict should raise an TypeError
        with self.assertRaises(TypeError):
            vp.connect(ip)

        with self.assertRaises(TypeError):
            vp.connect(op)

        with self.assertRaises(TypeError):
            vp.connect(rp)

        # Connect them via connect_from(..)
        # The type conflict should raise an TypeError
        with self.assertRaises(TypeError):
            vp.connect_from(ip)

        with self.assertRaises(TypeError):
            vp.connect_from(op)

        # Connect RefPort via connect_from(..) raises no error
        vp.connect_from(rp)


class TestVirtualPorts(unittest.TestCase):
    """Contains unit tests around virtual ports. Virtual ports are derived
    ports that are not directly created by the developer as part of process
    definition but which serve to somehow transform the properties of a
    developer-defined port."""

    def test_reshape(self):
        """Checks reshaping of a port."""

        # Create some ports
        op = OutPort((1, 2, 3))
        ip = InPort((3, 2, 1))

        # Using reshape(..), ports with different shape can be connected as
        # long as total number of elements does not change
        op.reshape((3, 2, 1)).connect(ip)

        # We can still find destination and source connection even with
        # virtual ports in the chain
        self.assertEqual(op.get_dst_ports(), [ip])
        self.assertEqual(ip.get_src_ports(), [op])

    def test_reshape_with_wrong_number_of_elements_raises_exception(self):
        """Checks whether an exception is raised when the number of elements
        in the specified shape is different from the number of elements in
        the source shape."""

        with self.assertRaises(ReshapeError):
            OutPort((1, 2, 3)).reshape((1, 2, 2))

    def test_flatten(self):
        """Checks flattening of a port."""

        op = OutPort((1, 2, 3))
        ip = InPort((6,))

        # Flatten the shape of the port.
        fp = op.flatten()
        self.assertEqual(fp.shape, (6,))

        # This enables connecting to an input port with a flattened shape.
        fp.connect(ip)

        # We can still find destination and source connection even with
        # virtual ports in the chain
        self.assertEqual(op.get_dst_ports(), [ip])
        self.assertEqual(ip.get_src_ports(), [op])

    def test_concat(self):
        """Checks concatenation of ports."""

        # Create a bunch of ports to be concatenated
        op1 = OutPort((2, 3, 1))
        op2 = OutPort((2, 3, 1))
        op3 = OutPort((2, 3, 1))
        ip1 = InPort((6, 3, 1))
        ip2 = InPort((2, 6, 1))

        # concat_with(..) concatenates calling port (op1) with other ports
        # (op2, op3) along given axis
        cp = op1.concat_with([op2, op3], axis=0)
        # The return value is a virtual ConcatPort ...
        self.assertIsInstance(cp, ConcatPort)
        # ... which needs to have the same dimensions ...
        self.assertEqual(cp.shape, (6, 3, 1))
        # ... as the port we want to connect the concatenated ports with
        cp.connect(ip1)
        # Finally, the virtual ConcatPort is the input connection of the ip1
        self.assertEqual(cp, ip1.in_connections[0])

        # Again, we can still find destination and source ports through a
        # chain of ports containing virtual ports
        self.assertEqual(op1.get_dst_ports(), [ip1])
        self.assertEqual(ip1.get_src_ports(), [op1, op2, op3])

        # We can also concat just op1 and op2 along axis 1 and compactly
        # connect to ip2 in one line because the ConcatPort instance is not
        # needed
        op1.concat_with(op2, axis=1).connect(ip2)
        # (2, 3, 1) + (2, 3, 1) concatenated along axis 1 results in (2, 6, 1)
        self.assertEqual(ip2.in_connections[0].shape, (2, 6, 1))

    def test_concat_with_incompatible_shapes_raises_exception(self):
        """Checks that incompatible ports cannot be concatenated."""

        # Create ports with incompatible shapes
        op1 = OutPort((2, 3, 1))
        op2 = OutPort((2, 4, 1))

        # Ports with incompatible shape outside of concatenation axis cannot
        # be concatenated because (3, 1) and (4, 1) don't match
        with self.assertRaises(ConcatShapeError):
            op1.concat_with(op2, axis=0)

    def test_concat_with_incompatible_type_raises_exception(self):
        """Checks that incompatible port types raise an exception."""

        op = OutPort((2, 3, 1))
        ip = InPort((2, 3, 1))
        # This will fail because concatenated ports must be of same type
        with self.assertRaises(AssertionError):
            op.concat_with(ip, axis=0)

    def test_concat_with_axis_out_of_bounds_raises_exception(self):
        """Checks whether an exception is raised when the specified axis is
        out of bounds."""

        op1 = OutPort((2, 3, 1))
        op2 = OutPort((2, 3, 1))
        with self.assertRaises(ConcatIndexError):
            op1.concat_with(op2, axis=3)

    def test_transpose(self):
        """Checks transposing of ports."""

        op = OutPort((1, 2, 3))
        ip = InPort((2, 1, 3))

        tp = op.transpose(axes=(1, 0, 2))
        # The return value is a virtual TransposePort ...
        self.assertIsInstance(tp, TransposePort)
        # ... which needs to have the same dimensions ...
        self.assertEqual(tp.shape, (2, 1, 3))
        # ... as the port we want to connect to.
        tp.connect(ip)
        # Finally, the virtual TransposePort is the input connection of the ip
        self.assertEqual(tp, ip.in_connections[0])

        # Again, we can still find destination and source ports through a
        # chain of ports containing virtual ports
        self.assertEqual(op.get_dst_ports(), [ip])
        self.assertEqual(ip.get_src_ports(), [op])

    def test_transpose_without_specified_axes(self):
        """Checks whether transpose reverses the shape-elements when no
        'axes' argument is given."""

        op = OutPort((1, 2, 3))
        tp = op.transpose()
        self.assertEqual(tp.shape, (3, 2, 1))

    def test_transpose_incompatible_axes_length_raises_exception(self):
        """Checks whether an exception is raised when the number of elements
        in the specified 'axes' argument differs from the number of elements
        of the parent port."""

        op = OutPort((1, 2, 3))
        with self.assertRaises(TransposeShapeError):
            op.transpose(axes=(0, 0, 1, 2))

    def test_transpose_incompatible_axes_indices_raises_exception(self):
        """Checks whether an exception is raised when the indices specified
        in the 'axes' argument are out of bounds for the parent port."""

        op = OutPort((1, 2, 3))
        with self.assertRaises(TransposeIndexError):
            op.transpose(axes=(0, 1, 3))


if __name__ == "__main__":
    unittest.main()
