# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import unittest
import numpy as np

from lava.magma.core.decorator import requires, tag, implements
from lava.magma.core.model.py.model import PyLoihiProcessModel
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
    InPort,
    OutPort,
    RefPort,
    VarPort
)


np.random.seed(7739)


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

    def test_transpose_inport_to_inport(self) -> None:
        """Tests a virtual TransposePort between an InPort and another InPort.
        In a real implementation, the source InPort would be in a
        hierarchical Process and the sink InPort would be in a SubProcess of
        that hierarchical Process."""

        out_port_process = OutPortProcess(data=self.input_data)
        source = InPortProcess(shape=self.shape)
        sink = InPortProcess(shape=self.shape_transposed)

        out_port_process.out_port.connect(source.in_port)
        source.in_port.transpose(axes=self.axes).connect(sink.in_port)

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

    def test_transpose_refport_write_to_varport(self) -> None:
        """Tests a virtual TransposePort between a RefPort and a VarPort,
        where the RefPort writes to the VarPort."""

        source = RefPortWriteProcess(data=self.input_data)
        sink = VarPortProcess(data=np.zeros(self.shape_transposed))
        source.ref_port.transpose(axes=self.axes).connect(sink.var_port)

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

    def test_transpose_refport_read_from_varport(self) -> None:
        """Tests a virtual TransposePort between a RefPort and a VarPort,
        where the RefPort reads from the VarPort."""

        source = VarPortProcess(data=self.input_data)
        sink = RefPortReadProcess(data=np.zeros(self.shape_transposed_reverse))
        sink.ref_port.transpose(axes=self.axes).connect(source.var_port)

        sink.run(condition=RunSteps(num_steps=self.num_steps),
                 run_cfg=Loihi1SimCfg(select_tag='floating_pt'))
        output = sink.data.get()
        sink.stop()

        expected = self.input_data.transpose(self.axes_reverse)
        self.assertTrue(
            np.all(output == expected),
            f'Input and output do not match.\n'
            f'{output[output!=expected]=}\n'
            f'{expected[output!=expected] =}\n'
        )

    @unittest.skip("RefPort to RefPort not yet implemented")
    def test_transpose_refport_write_to_refport(self) -> None:
        """Tests a virtual TransposePort between a RefPort and another
        RefPort, where the first RefPort writes to the second. In a real
        implementation, the source RefPort would be in a
        hierarchical Process and the sink RefPort would be in a SubProcess of
        that hierarchical Process."""

        source = RefPortWriteProcess(data=self.input_data)
        sink = RefPortReadProcess(data=np.zeros(self.shape_transposed))
        var_port_process = VarPortProcess(data=np.zeros(self.shape_transposed))

        source.ref_port.transpose(axes=self.axes).connect(sink.ref_port)
        sink.ref_port.connect(var_port_process.var_port)

        sink.run(condition=RunSteps(num_steps=self.num_steps),
                 run_cfg=Loihi1SimCfg(select_tag='floating_pt'))
        output = var_port_process.data.get()
        sink.stop()

        expected = self.input_data.transpose(self.axes)
        self.assertTrue(
            np.all(output == expected),
            f'Input and output do not match.\n'
            f'{output[output!=expected]=}\n'
            f'{expected[output!=expected] =}\n'
        )

    @unittest.skip("RefPort to RefPort not yet implemented")
    def test_transpose_refport_read_from_refport(self) -> None:
        """Tests a virtual TransposePort between a RefPort and another
        RefPort, where the first RefPort reads from the second. In a real
        implementation, the source RefPort would be in a
        hierarchical Process and the sink RefPort would be in a SubProcess of
        that hierarchical Process."""

        source = RefPortReadProcess(
            data=np.zeros(self.shape_transposed_reverse)
        )
        sink = RefPortWriteProcess(data=self.input_data)

        sink.ref_port.transpose(axes=self.axes).connect(source.ref_port)

        sink.run(condition=RunSteps(num_steps=self.num_steps),
                 run_cfg=Loihi1SimCfg(select_tag='floating_pt'))
        output = sink.data.get()
        sink.stop()

        expected = self.input_data.transpose(self.axes_reverse)
        self.assertTrue(
            np.all(output == expected),
            f'Input and output do not match.\n'
            f'{output[output!=expected]=}\n'
            f'{expected[output!=expected] =}\n'
        )

    @unittest.skip("VarPort to VarPort not yet implemented")
    def test_transpose_varport_write_to_varport(self) -> None:
        """Tests a virtual TransposePort between a VarPort and another
        VarPort, where the first VarPort writes to the second. In a
        real implementation, the source VarPort would be in a
        hierarchical Process and the sink VarPort would be in a SubProcess of
        that hierarchical Process."""

        source = VarPortProcess(data=self.input_data)
        sink = VarPortProcess(data=np.zeros(self.shape_transposed))

        source.var_port.transpose(axes=self.axes).connect(sink.var_port)

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

    @unittest.skip("VarPort to VarPort not yet implemented")
    def test_transpose_varport_read_from_varport(self) -> None:
        """Tests a virtual TransposePort between a VarPort and another
        VarPort, where the first VarPort reads from the second. In a real
        implementation, the source VarPort would be in a
        hierarchical Process and the sink VarPort would be in a SubProcess of
        that hierarchical Process."""

        sink = VarPortProcess(data=np.zeros(self.shape_transposed_reverse))
        source = VarPortProcess(data=self.input_data)

        sink.var_port.transpose(axes=self.axes).connect(source.var_port)

        sink.run(condition=RunSteps(num_steps=self.num_steps),
                 run_cfg=Loihi1SimCfg(select_tag='floating_pt'))
        output = sink.data.get()
        sink.stop()

        expected = self.input_data.transpose(self.axes_reverse)
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

    def test_reshape_inport_to_inport(self) -> None:
        """Tests a virtual ReshapePort between an InPort and another InPort.
        In a real implementation, the source InPort would be in a
        hierarchical Process and the sink InPort would be in a SubProcess of
        that hierarchical Process."""

        out_port_process = OutPortProcess(data=self.input_data)
        source = InPortProcess(shape=self.shape)
        sink = InPortProcess(shape=self.shape_reshaped)

        out_port_process.out_port.connect(source.in_port)
        source.in_port.reshape(new_shape=self.shape_reshaped).connect(
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

    def test_reshape_refport_write_to_varport(self) -> None:
        """Tests a virtual ReshapePort between a RefPort and a VarPort,
        where the RefPort writes to the VarPort."""

        source = RefPortWriteProcess(data=self.input_data)
        sink = VarPortProcess(data=np.zeros(self.shape_reshaped))
        source.ref_port.reshape(new_shape=self.shape_reshaped).connect(
            sink.var_port)

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

    def test_reshape_refport_read_from_varport(self) -> None:
        """Tests a virtual ReshapePort between a RefPort and a VarPort,
        where the RefPort reads from the VarPort."""

        source = VarPortProcess(data=self.input_data)
        sink = RefPortReadProcess(data=np.zeros(self.shape_reshaped))
        sink.ref_port.reshape(new_shape=self.shape_reshaped).connect(
            source.var_port)

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
    """Tests virtual FlattenPorts on Processes that are executed."""

    def setUp(self) -> None:
        self.num_steps = 1
        self.shape = (4, 3, 2)
        self.shape_reshaped = (24,)
        self.input_data = np.random.randint(256, size=self.shape)

    def test_flatten_outport_to_inport(self) -> None:
        """Tests a virtual FlattenPort between an OutPort and an InPort."""

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

    def test_flatten_inport_to_inport(self) -> None:
        """Tests a virtual FlattenPort between an InPort and another InPort.
        In a real implementation, the source InPort would be in a
        hierarchical Process and the sink InPort would be in a SubProcess of
        that hierarchical Process."""

        out_port_process = OutPortProcess(data=self.input_data)
        source = InPortProcess(shape=self.shape)
        sink = InPortProcess(shape=self.shape_reshaped)

        out_port_process.out_port.connect(source.in_port)
        source.in_port.flatten().connect(sink.in_port)

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

    def test_flatten_refport_write_to_varport(self) -> None:
        """Tests a virtual FlattenPort between a RefPort and a VarPort,
        where the RefPort writes to the VarPort."""

        source = RefPortWriteProcess(data=self.input_data)
        sink = VarPortProcess(data=np.zeros(self.shape_reshaped))
        source.ref_port.flatten().connect(sink.var_port)

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

    def test_flatten_refport_read_from_varport(self) -> None:
        """Tests a virtual FlattenPort between a RefPort and a VarPort,
        where the RefPort reads from the VarPort."""

        source = VarPortProcess(data=self.input_data)
        sink = RefPortReadProcess(data=np.zeros(self.shape_reshaped))
        sink.ref_port.flatten().connect(source.var_port)

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


class TestConcatPort(unittest.TestCase):
    """Tests virtual ConcatPorts on Processes that are executed."""

    def setUp(self) -> None:
        self.num_steps = 1
        self.shape = (3, 1, 4)
        self.shape_concat = (3, 3, 4)
        self.input_data = np.random.randint(256, size=self.shape)

    def test_concat_outport_to_inport(self) -> None:
        """Tests a virtual ConcatPort between an OutPort and an InPort."""

        source_1 = OutPortProcess(data=self.input_data)
        source_2 = OutPortProcess(data=self.input_data)
        source_3 = OutPortProcess(data=self.input_data)
        sink = InPortProcess(shape=self.shape_concat)

        source_1.out_port.concat_with([
            source_2.out_port,
            source_3.out_port], axis=1).connect(sink.in_port)

        sink.run(condition=RunSteps(num_steps=self.num_steps),
                 run_cfg=Loihi1SimCfg(select_tag='floating_pt'))
        output = sink.data.get()
        sink.stop()

        expected = np.concatenate([self.input_data] * 3, axis=1)
        self.assertTrue(
            np.all(output == expected),
            f'Input and output do not match.\n'
            f'{output[output!=expected]=}\n'
            f'{expected[output!=expected] =}\n'
        )


# minimal process with an OutPort
class OutPortProcess(AbstractProcess):
    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data=data)
        self.data = Var(shape=data.shape, init=data)
        self.out_port = OutPort(shape=data.shape)


# minimal process with an InPort
class InPortProcess(AbstractProcess):
    def __init__(self, shape: ty.Tuple[int, ...]) -> None:
        super().__init__(shape=shape)
        self.data = Var(shape=shape, init=np.zeros(shape))
        self.in_port = InPort(shape=shape)


# A minimal process with a RefPort that writes
class RefPortWriteProcess(AbstractProcess):
    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data=data)
        self.data = Var(shape=data.shape, init=data)
        self.ref_port = RefPort(shape=data.shape)


# A minimal process with a RefPort that reads
class RefPortReadProcess(AbstractProcess):
    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data=data)
        self.data = Var(shape=data.shape, init=data)
        self.ref_port = RefPort(shape=data.shape)


# A minimal process with a Var and a VarPort
class VarPortProcess(AbstractProcess):
    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data=data)
        self.data = Var(shape=data.shape, init=data)
        self.var_port = VarPort(self.data)


# A minimal PyProcModel implementing OutPortProcess
@implements(proc=OutPortProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyOutPortProcessModelFloat(PyLoihiProcessModel):
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32)
    data: np.ndarray = LavaPyType(np.ndarray, np.int32)

    def run_spk(self):
        self.out_port.send(self.data)
        print("Sent output data of OutPortProcess: ", str(self.data))


# A minimal PyProcModel implementing InPortProcess
@implements(proc=InPortProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyInPortProcessModelFloat(PyLoihiProcessModel):
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32)
    data: np.ndarray = LavaPyType(np.ndarray, np.int32)

    def run_spk(self):
        self.data[:] = self.in_port.recv()
        print("Received input data for InPortProcess: ", str(self.data))


# A minimal PyProcModel implementing RefPortWriteProcess
@implements(proc=RefPortWriteProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyRefPortWriteProcessModelFloat(PyLoihiProcessModel):
    ref_port: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, np.int32)
    data: np.ndarray = LavaPyType(np.ndarray, np.int32)

    def pre_guard(self):
        return True

    def run_pre_mgmt(self):
        self.ref_port.write(self.data)
        print("Sent output data of RefPortWriteProcess: ", str(self.data))


# A minimal PyProcModel implementing RefPortReadProcess
@implements(proc=RefPortReadProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyRefPortReadProcessModelFloat(PyLoihiProcessModel):
    ref_port: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, np.int32)
    data: np.ndarray = LavaPyType(np.ndarray, np.int32)

    def pre_guard(self):
        return True

    def run_pre_mgmt(self):
        self.data = self.ref_port.read()
        print("Received input data for RefPortReadProcess: ", str(self.data))


# A minimal PyProcModel implementing VarPortProcess
@implements(proc=VarPortProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyVarPortProcessModelFloat(PyLoihiProcessModel):
    var_port: PyInPort = LavaPyType(PyVarPort.VEC_DENSE, np.int32)
    data: np.ndarray = LavaPyType(np.ndarray, np.int32)


if __name__ == '__main__':
    unittest.main()
