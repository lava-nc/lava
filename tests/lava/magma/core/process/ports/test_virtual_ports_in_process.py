# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import unittest
import numpy as np
import functools as ft

from lava.magma.core.decorator import requires, tag, implements
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.variable import Var
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import CPU
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.model.py.ports import (
    PyInPort,
    PyOutPort,
    PyRefPort,
    PyVarPort
)
from lava.magma.core.process.ports.ports import (
    AbstractPort,
    AbstractVirtualPort,
    InPort,
    OutPort,
    RefPort,
    VarPort
)


np.random.seed(7739)


class MockVirtualPort(AbstractVirtualPort, AbstractPort):
    """A mock-up of a virtual port that reshapes the input."""
    def __init__(self, new_shape: ty.Tuple):
        AbstractPort.__init__(self, new_shape)

    def get_transform_func(self) -> ft.partial:
        return ft.partial(np.reshape, newshape=self.shape)


class TestVirtualPortNetworkTopologies(unittest.TestCase):
    """Tests different network topologies that include virtual ports using a
    dummy virtual port as a stand-in for all types of virtual ports."""

    def setUp(self) -> None:
        self.num_steps = 1
        self.shape = (4, 3, 2)
        self.new_shape = (12, 2)
        self.input_data = np.random.randint(256, size=self.shape)

    def test_outport_to_inport_in_hierarchical_processes(self) -> None:
        """Tests a virtual port between an OutPort of a hierarchical Process
        and an InPort of another hierarchical Process."""

        source = HOutPortProcess(data=self.input_data)
        sink = HInPortProcess(shape=self.new_shape)

        virtual_port = MockVirtualPort(new_shape=self.new_shape)

        source.out_port._connect_forward(
            [virtual_port], AbstractPort, assert_same_shape=False
        )
        virtual_port.connect(sink.in_port)

        sink.run(condition=RunSteps(num_steps=self.num_steps),
                 run_cfg=Loihi1SimCfg(select_tag='floating_pt'))
        output = sink.data.get()
        sink.stop()

        expected = self.input_data.reshape(self.new_shape)
        self.assertTrue(
            np.all(output == expected),
            f'Input and output do not match.\n'
            f'{output[output!=expected]=}\n'
            f'{expected[output!=expected] =}\n'
        )

    def test_inport_to_inport_in_a_hierarchical_process(self) -> None:
        """Tests a virtual port between an InPort of a hierarchical Process
        and an InPort in a nested Process.

        The data comes from an OutPortProcess and enters the hierarchical
        Process (HVPInPortProcess) via its InPort. That InPort is connected
        via a virtual port to the InPort of the nested Process. The data is
        from there written into a Var 'data' of the nested Process, for which
        the Var 's_data' of the hierarchical Process is an alias.
        """

        out_port_process = OutPortProcess(data=self.input_data)
        h_proc = HVPInPortProcess(h_shape=self.shape, s_shape=self.new_shape)

        out_port_process.out_port.connect(h_proc.in_port)

        h_proc.run(condition=RunSteps(num_steps=self.num_steps),
                   run_cfg=Loihi1SimCfg(select_tag='floating_pt'))
        output = h_proc.s_data.get()
        h_proc.stop()

        expected = self.input_data.reshape(self.new_shape)
        self.assertTrue(
            np.all(output == expected),
            f'Input and output do not match.\n'
            f'{output[output!=expected]=}\n'
            f'{expected[output!=expected] =}\n'
        )

    def test_outport_to_outport_in_a_hierarchical_process(self) -> None:
        """Tests a virtual port between an OutPort of a child Process and an
        OutPort of the corresponding hierarchical parent Process.

        The data comes from the child Process, is passed from its OutPort
        through a virtual port (where it is reshaped) to the OutPort of the
        hierarchical Process (HVPOutPortProcess). From there it is passed to
        the InPort of the InPortProcess and written into a Var 'data'.
        """

        h_proc = HVPOutPortProcess(h_shape=self.new_shape, data=self.input_data)
        in_port_process = InPortProcess(shape=self.new_shape)

        h_proc.out_port.connect(in_port_process.in_port)

        h_proc.run(condition=RunSteps(num_steps=self.num_steps),
                   run_cfg=Loihi1SimCfg(select_tag='floating_pt'))
        output = in_port_process.data.get()
        h_proc.stop()

        expected = self.input_data.reshape(self.new_shape)
        self.assertTrue(
            np.all(output == expected),
            f'Input and output do not match.\n'
            f'{output[output!=expected]=}\n'
            f'{expected[output!=expected] =}\n'
        )

    def test_refport_write_to_varport(self) -> None:
        """Tests a virtual port between a RefPort and a VarPort,
        where the RefPort writes to the VarPort."""

        source = RefPortWriteProcess(data=self.input_data)
        sink = VarPortProcess(data=np.zeros(self.new_shape))

        virtual_port = MockVirtualPort(new_shape=self.new_shape)

        source.ref_port._connect_forward(
            [virtual_port], AbstractPort, assert_same_shape=False
        )
        virtual_port.connect(sink.var_port)

        try:
            sink.run(condition=RunSteps(num_steps=self.num_steps),
                     run_cfg=Loihi1SimCfg(select_tag='floating_pt'))
            output = sink.data.get()
        finally:
            sink.stop()

        expected = self.input_data.reshape(self.new_shape)
        self.assertTrue(
            np.all(output == expected),
            f'Input and output do not match.\n'
            f'{output[output!=expected]=}\n'
            f'{expected[output!=expected] =}\n'
        )

    def test_chaining_multiple_virtual_ports(self) -> None:
        """Tests whether two virtual ReshapePorts can be chained through the
        flatten() method."""

        source = OutPortProcess(data=self.input_data)
        shape_final = (int(np.prod(self.shape)),)
        sink = InPortProcess(shape=shape_final)

        virtual_port1 = MockVirtualPort(new_shape=self.new_shape)
        virtual_port2 = MockVirtualPort(new_shape=shape_final)

        source.out_port._connect_forward(
            [virtual_port1], AbstractPort, assert_same_shape=False
        )
        virtual_port1._connect_forward(
            [virtual_port2], AbstractPort, assert_same_shape=False
        )
        virtual_port2.connect(sink.in_port)

        sink.run(condition=RunSteps(num_steps=self.num_steps),
                 run_cfg=Loihi1SimCfg(select_tag='floating_pt'))
        output = sink.data.get()
        sink.stop()

        expected = self.input_data.ravel()
        self.assertTrue(
            np.all(output == expected),
            f'Input and output do not match.\n'
            f'{output[output!=expected]=}\n'
            f'{expected[output!=expected] =}\n'
        )

    def test_joining_virtual_ports_throws_exception(self) -> None:
        """Tests whether joining two virtual ports throws an exception."""

        source1 = OutPortProcess(data=self.input_data)
        source2 = OutPortProcess(data=self.input_data)
        sink = InPortProcess(shape=self.new_shape)

        virtual_port1 = MockVirtualPort(new_shape=self.new_shape)
        virtual_port2 = MockVirtualPort(new_shape=self.new_shape)

        source1.out_port._connect_forward(
            [virtual_port1], AbstractPort, assert_same_shape=False
        )
        source2.out_port._connect_forward(
            [virtual_port2], AbstractPort, assert_same_shape=False
        )

        virtual_port1.connect(sink.in_port)
        virtual_port2.connect(sink.in_port)

        with self.assertRaises(NotImplementedError):
            sink.run(condition=RunSteps(num_steps=self.num_steps),
                     run_cfg=Loihi1SimCfg(select_tag='floating_pt'))


class TestTransposePort(unittest.TestCase):
    """Tests virtual TransposePorts on Processes that are executed."""

    def setUp(self) -> None:
        self.num_steps = 1
        self.axes = (2, 0, 1)
        self.axes_reverse = list(self.axes)
        for idx, ax in enumerate(self.axes):
            self.axes_reverse[ax] = idx
        self.axes_reverse = tuple(self.axes_reverse)
        self.shape = (4, 3, 2)
        self.shape_transposed = tuple(self.shape[i] for i in self.axes)
        self.shape_transposed_reverse = \
            tuple(self.shape[i] for i in self.axes_reverse)
        self.input_data = np.random.randint(256, size=self.shape)

    def test_transpose_outport_to_inport(self) -> None:
        """Tests a virtual TransposePort between an OutPort and an InPort."""

        source = OutPortProcess(data=self.input_data)
        sink = InPortProcess(shape=self.shape_transposed)

        source.out_port.transpose(axes=self.axes).connect(sink.in_port)

        sink.run(condition=RunSteps(num_steps=self.num_steps),
                 run_cfg=Loihi1SimCfg(select_tag='floating_pt'))
        output = sink.data.get()
        sink.stop()

        expected = self.input_data.transpose(self.axes)
        self.assertTrue(
            np.all(output == expected),
            f'Input and output do not match.\n'
            f'{output[output!=expected]=}\n'
            f'{expected[output!=expected] =}\n'
        )


class TestReshapePort(unittest.TestCase):
    """Tests virtual ReshapePorts on Processes that are executed."""

    def setUp(self) -> None:
        self.num_steps = 1
        self.shape = (4, 3, 2)
        self.shape_reshaped = (12, 2)
        self.input_data = np.random.randint(256, size=self.shape)

    def test_reshape_outport_to_inport(self) -> None:
        """Tests a virtual ReshapePort between an OutPort and an InPort."""

        source = OutPortProcess(data=self.input_data)
        sink = InPortProcess(shape=self.shape_reshaped)

        source.out_port.reshape(new_shape=self.shape_reshaped).connect(
            sink.in_port)

        sink.run(condition=RunSteps(num_steps=self.num_steps),
                 run_cfg=Loihi1SimCfg(select_tag='floating_pt'))
        output = sink.data.get()
        sink.stop()

        expected = self.input_data.reshape(self.shape_reshaped)
        self.assertTrue(
            np.all(output == expected),
            f'Input and output do not match.\n'
            f'{output[output!=expected]=}\n'
            f'{expected[output!=expected] =}\n'
        )


class TestFlattenPort(unittest.TestCase):
    """Tests virtual ReshapePorts, created by the flatten() method,
    on Processes that are executed."""

    def setUp(self) -> None:
        self.num_steps = 1
        self.shape = (4, 3, 2)
        self.shape_reshaped = (24,)
        self.input_data = np.random.randint(256, size=self.shape)

    def test_flatten_outport_to_inport(self) -> None:
        """Tests a virtual ReshapePort with flatten() between an OutPort and an
        InPort."""

        source = OutPortProcess(data=self.input_data)
        sink = InPortProcess(shape=self.shape_reshaped)

        source.out_port.flatten().connect(sink.in_port)

        sink.run(condition=RunSteps(num_steps=self.num_steps),
                 run_cfg=Loihi1SimCfg(select_tag='floating_pt'))
        output = sink.data.get()
        sink.stop()

        expected = self.input_data.ravel()
        self.assertTrue(
            np.all(output == expected),
            f'Input and output do not match.\n'
            f'{output[output!=expected]=}\n'
            f'{expected[output!=expected] =}\n'
        )


# A minimal Process with an OutPort
class OutPortProcess(AbstractProcess):
    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data=data)
        self.data = Var(shape=data.shape, init=data)
        self.out_port = OutPort(shape=data.shape)


# A minimal PyProcModel implementing OutPortProcess
@implements(proc=OutPortProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyOutPortProcessModelFloat(PyLoihiProcessModel):
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32)
    data: np.ndarray = LavaPyType(np.ndarray, np.int32)

    def run_spk(self):
        self.out_port.send(self.data)
        self.log.info("Sent output data of OutPortProcess: ", str(self.data))


# A minimal Process with an InPort
class InPortProcess(AbstractProcess):
    def __init__(self, shape: ty.Tuple[int, ...]) -> None:
        super().__init__(shape=shape)
        self.data = Var(shape=shape, init=np.zeros(shape))
        self.in_port = InPort(shape=shape)


# A minimal PyProcModel implementing InPortProcess
@implements(proc=InPortProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyInPortProcessModelFloat(PyLoihiProcessModel):
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32)
    data: np.ndarray = LavaPyType(np.ndarray, np.int32)

    def run_spk(self):
        self.data[:] = self.in_port.recv()
        self.log.info("Received input data for InPortProcess: ", str(self.data))


# A minimal hierarchical Process with an OutPort
class HOutPortProcess(AbstractProcess):
    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data=data)
        self.out_port = OutPort(shape=data.shape)
        self.proc_params['data'] = data


# A minimal hierarchical ProcModel with a nested OutPortProcess
@implements(proc=HOutPortProcess)
class SubHOutPortProcModel(AbstractSubProcessModel):
    def __init__(self, proc):
        self.out_proc = OutPortProcess(data=proc.proc_params['data'])
        self.out_proc.out_port.connect(proc.out_port)


# A minimal hierarchical Process with an InPort and a Var
class HInPortProcess(AbstractProcess):
    def __init__(self, shape: ty.Tuple[int, ...]) -> None:
        super().__init__(shape=shape)
        self.data = Var(shape=shape, init=np.zeros(shape))
        self.in_port = InPort(shape=shape)
        self.proc_params['shape'] = shape


# A minimal hierarchical ProcModel with a nested InPortProcess and an aliased
# Var
@implements(proc=HInPortProcess)
class SubHInPortProcModel(AbstractSubProcessModel):
    def __init__(self, proc):
        self.in_proc = InPortProcess(shape=proc.proc_params['shape'])
        proc.in_port.connect(self.in_proc.in_port)
        proc.data.alias(self.in_proc.data)


# A minimal hierarchical Process with an InPort, where the SubProcessModel
# connects the InPort to another InPort via a virtual port. The data that
# comes in through the InPort will eventually be accessible through the
# 's_data' Var.
class HVPInPortProcess(AbstractProcess):
    def __init__(self,
                 h_shape: ty.Tuple[int, ...],
                 s_shape: ty.Tuple[int, ...]) -> None:
        super().__init__()
        self.s_data = Var(shape=s_shape, init=np.zeros(s_shape))
        self.in_port = InPort(shape=h_shape)
        self.proc_params['s_shape'] = s_shape


# A minimal hierarchical ProcModel with a nested InPortProcess and an aliased
# Var
@implements(proc=HVPInPortProcess)
class SubHVPInPortProcModel(AbstractSubProcessModel):
    def __init__(self, proc):
        self.in_proc = InPortProcess(shape=proc.proc_params['s_shape'])

        virtual_port = MockVirtualPort(new_shape=proc.proc_params['s_shape'])
        proc.in_port._connect_forward(
            [virtual_port], AbstractPort, assert_same_shape=False
        )
        virtual_port.connect(self.in_proc.in_port)

        proc.s_data.alias(self.in_proc.data)


# A minimal hierarchical Process with an OutPort, where the data that is
# given as an argument may have a different shape than the OutPort of the
# Process.
class HVPOutPortProcess(AbstractProcess):
    def __init__(self,
                 h_shape: ty.Tuple[int, ...],
                 data: np.ndarray) -> None:
        super().__init__(h_shape=h_shape, data=data)
        self.out_port = OutPort(shape=h_shape)
        self.proc_params['data'] = data
        self.proc_params['h_shape'] = h_shape


# A minimal hierarchical ProcModel with a nested OutPortProcess
@implements(proc=HVPOutPortProcess)
class SubHVPOutPortProcModel(AbstractSubProcessModel):
    def __init__(self, proc):
        self.out_proc = OutPortProcess(data=proc.proc_params['data'])

        virtual_port = MockVirtualPort(new_shape=proc.proc_params['h_shape'])
        self.out_proc.out_port._connect_forward(
            [virtual_port], AbstractPort, assert_same_shape=False
        )
        virtual_port.connect(proc.out_port)


# A minimal Process with a RefPort that writes
class RefPortWriteProcess(AbstractProcess):
    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data=data)
        self.data = Var(shape=data.shape, init=data)
        self.ref_port = RefPort(shape=data.shape)


# A minimal PyProcModel implementing RefPortWriteProcess
@implements(proc=RefPortWriteProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyRefPortWriteProcessModelFloat(PyLoihiProcessModel):
    ref_port: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, np.int32)
    data: np.ndarray = LavaPyType(np.ndarray, np.int32)

    def post_guard(self):
        return True

    def run_post_mgmt(self):
        self.ref_port.write(self.data)
        self.log.info("Sent output data of RefPortWriteProcess: ",
                      str(self.data))


# A minimal Process with a Var and a VarPort
class VarPortProcess(AbstractProcess):
    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data=data)
        self.data = Var(shape=data.shape, init=data)
        self.var_port = VarPort(self.data)


# A minimal PyProcModel implementing VarPortProcess
@implements(proc=VarPortProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyVarPortProcessModelFloat(PyLoihiProcessModel):
    var_port: PyInPort = LavaPyType(PyVarPort.VEC_DENSE, np.int32)
    data: np.ndarray = LavaPyType(np.ndarray, np.int32)


if __name__ == '__main__':
    unittest.main()
