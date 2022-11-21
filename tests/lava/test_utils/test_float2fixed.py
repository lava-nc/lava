# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
import matplotlib
import unittest

from lava.utils.float2fixed import Float2FixedConverter
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.precision import Precision
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyOutPort, PyInPort, PyRefPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort, RefPort
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.decorator import implements, requires, tag
from lava.proc.monitor.process import Monitor
from lava.proc.io.sink import RingBuffer
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.magma.compiler import compiler_graphs
from lava.magma.core.run_configs import Loihi1SimCfg
from collections import OrderedDict


class Proc(AbstractProcess):
    """A dummy Process with two Vars. Behavior will be implemented with various
    ProcessModels to check converter."""

    def __init__(self, shape=(1,), **kwargs):
        u = kwargs.get('u', 0)
        v = kwargs.get('v', 5)
        super().__init__(**kwargs)
        self.u = Var(shape=shape, init=u)
        self.v = Var(shape=shape, init=v)
        self.inport = InPort(shape=shape)
        self.outport = OutPort(shape=shape)
        self.shape = shape


class ProcDense(AbstractProcess):
    """A dummy dense Process with only a weight."""

    def __init__(self, **kwargs):
        w = kwargs.get('w', 0)
        super().__init__(**kwargs)
        self.w = Var(shape=(1,), init=w)
        self.outport = OutPort(shape=(1,))
        self.inport = InPort(shape=(1,))


class HierProc(AbstractProcess):
    """A dummy hierarchical Process with In- and OutPorts. Will wrap around a
    Proc (see above) Process."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inport = InPort(shape=(1,))
        self.outport = OutPort(shape=(1,))


class ProcRef(AbstractProcess):
    """A dummy process with RefPort."""
    def __init__(self, **kwargs):
        u = kwargs.get('u', 0)
        v = kwargs.get('v', 5)
        w = kwargs.get('w', 5)
        super().__init__(**kwargs)
        self.refport = RefPort(shape=(1,))
        self.v = Var(shape=(1,), init=v)
        self.u = Var(shape=(1,), init=u)
        self.w = Var(shape=(1,), init=w)


class ProcVar(AbstractProcess):
    """A dummy process to which RefProc connects.."""
    def __init__(self, **kwargs):
        v = kwargs.get('v', 5)
        super().__init__(**kwargs)
        self.v = Var(shape=(1,), init=v)


@implements(proc=Proc, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt_pytype_target_err')
class ProcPyProcModelTargetErr(PyLoihiProcessModel):
    """A minimal fixed-point PyProcModel implementing Proc."""
    u: np.ndarray = LavaPyType(np.ndarray, np.int32, domain=None,
                               precision=17, constant=True, num_bits_exp=4,
                               meta_parameter=False)
    v: np.ndarray = LavaPyType(np.ndarray, np.int32,
                               precision=Precision(is_signed=False,
                                                   num_bits=24,
                                                   implicit_shift=0),
                               domain=None, constant=False,
                               num_bits_exp=None, meta_parameter=False)
    inport: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int16,
                                  precision="u:16:3")
    outport: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int16,
                                    precision="u:16:0")


@implements(proc=Proc, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt_pytype_var_err')
class ProcPyProcModelVarErr(PyLoihiProcessModel):
    """A minimal fixed-point PyProcModel implementing Proc."""
    u: np.ndarray = LavaPyType(np.ndarray, np.int32,
                               precision=Precision(is_signed=True,
                                                   num_bits=24,
                                                   implicit_shift=0),
                               domain=None, constant=True, num_bits_exp=4,
                               meta_parameter=False)
    v: np.ndarray = LavaPyType(np.ndarray, np.int32,
                               precision=Precision(is_signed=False,
                                                   num_bits=24,
                                                   implicit_shift=0),
                               domain=None, constant=False,
                               num_bits_exp=None, meta_parameter=False)
    inport: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int16,
                                  precision=Precision(is_signed=False,
                                                      num_bits=16,
                                                      implicit_shift=3.5))
    outport: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int16,
                                    precision=Precision(is_signed=False,
                                                        num_bits=16,
                                                        implicit_shift=0))


@implements(proc=Proc, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt_1')
class ProcPyProcModel1(PyLoihiProcessModel):
    """A minimal fixed-point PyProcModel implementing Proc."""

    u: np.ndarray = LavaPyType(np.ndarray, np.int32,
                               precision=Precision(is_signed=True,
                                                   num_bits=17,
                                                   implicit_shift=6),
                               domain=None, constant=True, num_bits_exp=4,
                               exp_var='exp_var', meta_parameter=False)
    v: np.ndarray = LavaPyType(np.ndarray, np.int32,
                               precision=Precision(is_signed=False,
                                                   num_bits=24,
                                                   implicit_shift=0),
                               domain=None, constant=False,
                               num_bits_exp=None, meta_parameter=False)
    inport: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int16,
                                  precision=Precision(is_signed=False,
                                                      num_bits=16,
                                                      implicit_shift=3))
    outport: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int16,
                                    precision=Precision(is_signed=False,
                                                        num_bits=16,
                                                        implicit_shift=0))


@implements(proc=Proc, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt_2')
class ProcPyProcModel2(PyLoihiProcessModel):
    """A minimal fixed-point PyProcModel implementing Proc."""

    u: np.ndarray = LavaPyType(np.ndarray, np.int32,
                               precision=Precision(is_signed=True,
                                                   num_bits=17,
                                                   implicit_shift=6),
                               domain=None, constant=True, num_bits_exp=4,
                               meta_parameter=True)
    v: np.ndarray = LavaPyType(np.ndarray, np.int32,
                               precision=Precision(is_signed=False,
                                                   num_bits=24,
                                                   implicit_shift=0),
                               domain=None, constant=False,
                               num_bits_exp=None, meta_parameter=False)
    inport: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int16,
                                  precision=Precision(is_signed=False,
                                                      num_bits=6,
                                                      implicit_shift=6))
    outport: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int16,
                                    precision=Precision(is_signed=False,
                                                        num_bits=16,
                                                        implicit_shift=0))


@implements(proc=Proc, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt_3')
class ProcPyProcModel3(PyLoihiProcessModel):
    """A minimal fixed-point PyProcModel implementing Proc."""

    u: np.ndarray = LavaPyType(np.ndarray, np.int32,
                               precision=Precision(is_signed=True,
                                                   num_bits=17,
                                                   implicit_shift=6),
                               domain=[0, 1])
    v: np.ndarray = LavaPyType(np.ndarray, np.int32,
                               precision=Precision(is_signed=False,
                                                   num_bits=24,
                                                   implicit_shift=0))
    inport: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int16,
                                  precision=Precision(is_signed=False,
                                                      num_bits=16,
                                                      implicit_shift=6))
    outport: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int16,
                                    precision=Precision(is_signed=False,
                                                        num_bits=16,
                                                        implicit_shift=0))


@implements(proc=Proc, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt_4')
class ProcPyProcModel4(PyLoihiProcessModel):
    """A minimal fixed-point PyProcModel implementing Proc."""

    u: np.ndarray = LavaPyType(np.ndarray, np.int32,
                               precision=Precision(is_signed=True,
                                                   num_bits=17,
                                                   implicit_shift=6))
    v: np.ndarray = LavaPyType(np.ndarray, np.int32,
                               precision=Precision(is_signed=False,
                                                   num_bits=24,
                                                   implicit_shift=0))
    inport: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int16,
                                  precision=Precision(is_signed=False,
                                                      num_bits=16,
                                                      implicit_shift=0))
    outport: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int16,
                                    precision=Precision(is_signed=False,
                                                        num_bits=16,
                                                        implicit_shift=0))


@implements(proc=Proc, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt_5')
class ProcPyProcModel5(PyLoihiProcessModel):
    """A minimal fixed-point PyProcModel implementing Proc."""

    u: np.ndarray = LavaPyType(np.ndarray, np.int32,
                               precision=Precision(is_signed=True,
                                                   num_bits=17,
                                                   implicit_shift=6),
                               domain=None, constant=True, num_bits_exp=4,
                               meta_parameter=False, scale_domain=1,
                               exp_var='exp_var_u')
    v: np.ndarray = LavaPyType(np.ndarray, np.int32,
                               precision=Precision(is_signed=False,
                                                   num_bits=24,
                                                   implicit_shift=0),
                               domain=None, constant=False, scale_domain=0,
                               num_bits_exp=None)
    inport: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int16,
                                  precision=Precision(is_signed=False,
                                                      num_bits=16,
                                                      implicit_shift=6))
    outport: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int16,
                                    precision=Precision(is_signed=False,
                                                        num_bits=16,
                                                        implicit_shift=0))


@implements(proc=Proc, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt_6')
class ProcPyProcModel6(PyLoihiProcessModel):
    """A minimal fixed-point PyProcModel implementing Proc."""

    u: np.ndarray = LavaPyType(np.ndarray, np.int32,
                               precision=Precision(is_signed=True,
                                                   num_bits=17,
                                                   implicit_shift=6),
                               domain=None, constant=True, num_bits_exp=4,
                               meta_parameter=True)
    v: np.ndarray = LavaPyType(np.ndarray, np.int32,
                               precision=Precision(is_signed=False,
                                                   num_bits=24,
                                                   implicit_shift=0),
                               domain=None, constant=False,
                               num_bits_exp=None, meta_parameter=False)
    inport: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int16,
                                  precision=Precision(is_signed=False,
                                                      num_bits=16,
                                                      implicit_shift=0))
    outport: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int16,
                                    precision=Precision(is_signed=False,
                                                        num_bits=16,
                                                        implicit_shift=0))


@implements(proc=Proc, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt_1')
class ProcPyProcModelFloat1(PyLoihiProcessModel):
    """A minimal floating-point PyProcModel implementing Proc."""

    u: np.ndarray = LavaPyType(np.ndarray , float)
    v: np.ndarray = LavaPyType(np.ndarray, float)
    inport: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    outport: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self):
        self.u += 1
        self.v += 2


@implements(proc=Proc, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt_2')
class ProcPyProcModelFloat2(PyLoihiProcessModel):
    """A minimal floating-point PyProcModel implementing Proc."""

    u: np.ndarray = LavaPyType(np.ndarray , float)
    v: np.ndarray = LavaPyType(np.ndarray, float)
    inport: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    outport: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self):
        self.v += 2


@implements(proc=HierProc, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt_1')
class SubProcPyProcModelHierProc(AbstractSubProcessModel):
    """A minimal floating-point PyProcModel implementing HierProc."""

    def __init__(self, proc):

        self.proc_in = Proc()

        self.proc_in.outport.connect(proc.outport)
        proc.inport.connect(self.proc_in.inport)


@implements(proc=ProcDense, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt_1')
class ProcDensePyProcModel1(PyLoihiProcessModel):
    """A minimal fixed-point PyProcModel implementing ProcDense."""

    w: np.ndarray = LavaPyType(np.ndarray, np.int32,
                               precision=Precision(is_signed=False,
                                                   num_bits=16,
                                                   implicit_shift=0),
                               constant=True)
    inport: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int16,
                                  precision=Precision(is_signed=False,
                                                      num_bits=16,
                                                      implicit_shift=0))
    outport: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int16,
                                    precision=Precision(is_signed=False,
                                                        num_bits=16,
                                                        implicit_shift=0))


@implements(proc=ProcDense, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt_2')
class ProcDensePyProcModel2(PyLoihiProcessModel):
    """A minimal fixed-point PyProcModel implementing ProcDense."""

    w: np.ndarray = LavaPyType(np.ndarray, np.int32,
                               precision=Precision(is_signed=True,
                                                   num_bits=16,
                                                   implicit_shift=3),
                               constant=True)
    inport: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int16,
                                  precision=Precision(is_signed=True,
                                                      num_bits=16,
                                                      implicit_shift=0))
    outport: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int16,
                                    precision=Precision(is_signed=True,
                                                        num_bits=16,
                                                        implicit_shift=0))


@implements(proc=ProcDense, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt_3')
class ProcDensePyProcModel3(PyLoihiProcessModel):
    """A minimal fixed-point PyProcModel implementing ProcDense."""

    w: np.ndarray = LavaPyType(np.ndarray, np.int32,
                               precision=Precision(is_signed=True,
                                                   num_bits=16,
                                                   implicit_shift=0),
                               constant=True)
    inport: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int16,
                                  precision=Precision(is_signed=False,
                                                      num_bits=16,
                                                      implicit_shift=0))
    outport: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int16,
                                    precision=Precision(is_signed=False,
                                                        num_bits=15,
                                                        implicit_shift=0))


@implements(proc=ProcDense, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt_5')
class ProcDensePyProcModel5(PyLoihiProcessModel):
    """A minimal fixed-point PyProcModel implementing ProcDense."""

    w: np.ndarray = LavaPyType(np.ndarray, np.int32,
                               precision=Precision(is_signed=True,
                                                   num_bits=16,
                                                   implicit_shift=0),
                               constant=True, scale_domain=0)
    inport: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int16,
                                  precision=Precision(is_signed=False,
                                                      num_bits=16,
                                                      implicit_shift=0))
    outport: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int16,
                                    precision=Precision(is_signed=False,
                                                        num_bits=15,
                                                        implicit_shift=0))


@implements(proc=ProcDense, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt_6')
class ProcDensePyProcModel6(PyLoihiProcessModel):
    """A minimal fixed-point PyProcModel implementing ProcDense."""

    w: np.ndarray = LavaPyType(np.ndarray, np.int32,
                               precision=Precision(is_signed=True,
                                                   num_bits=16,
                                                   implicit_shift=1),
                               constant=True, num_bits_exp=2, exp_var="exp_w")
    inport: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int16,
                                  precision=Precision(is_signed=False,
                                                      num_bits=16,
                                                      implicit_shift=0))
    outport: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int16,
                                    precision=Precision(is_signed=False,
                                                        num_bits=15,
                                                        implicit_shift=0))


@implements(proc=ProcDense, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt_1', 'floating_pt_2')
class ProcDensePyProcModelFloat2(PyLoihiProcessModel):
    """A minimal fixed-point PyProcModel implementing ProcDense."""

    w: np.ndarray = LavaPyType(np.ndarray, float, constant=True)
    inport: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    outport: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)


@implements(proc=ProcRef, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class ProcRefPyProcModelFixed(PyLoihiProcessModel):
    """A minimal fixed-point PyProcModel implementing ProcRef."""

    refport : PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, int)
    v: np.ndarray = LavaPyType(np.ndarray, np.int32,
                               precision=Precision(is_signed=True,
                                                   num_bits=16,
                                                   implicit_shift=0),
                               constant=False, scale_domain=0)
    u: np.ndarray = LavaPyType(np.ndarray, np.int32,
                               precision=Precision(is_signed=True,
                                                   num_bits=16,
                                                   implicit_shift=3),
                               constant=False, scale_domain=0)
    w: np.ndarray = LavaPyType(np.ndarray, np.int32,
                               precision=Precision(is_signed=False,
                                                   num_bits=12,
                                                   implicit_shift=0),
                               constant=True, scale_domain=1)


@implements(proc=ProcVar, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class ProcVarPyProcModelFixed(PyLoihiProcessModel):
    """A minimal fixed-point PyProcModel implementing ProcVar."""

    v: np.ndarray = LavaPyType(np.ndarray, np.int32,
                               precision=Precision(is_signed=True,
                                                   num_bits=16,
                                                   implicit_shift=3),
                               constant=False, scale_domain=0)


class TestFloat2FixedConverter(unittest.TestCase):

    def test_converter_constructor(self):
        """Check if Float2Fixed Converter is correctly instantiated."""
        converter = Float2FixedConverter()

        # Check if correct instance is created.
        self.assertIsInstance(converter, Float2FixedConverter)

    def test_set_run_cfg(self):
        """Check if set_run_cfg correctly sets run configs."""
        converter = Float2FixedConverter()

        floating_pt_run_cfg = Loihi1SimCfg(select_tag='floating_pt_1')
        fixed_pt_run_cfg = Loihi1SimCfg(select_tag='fixed_pt')

        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)

        # Check if run configs are set correctly.
        self.assertEqual(converter.floating_pt_run_cfg, floating_pt_run_cfg)
        self.assertEqual(converter.fixed_pt_run_cfg, fixed_pt_run_cfg)

    def test_set_run_cfg_raise_error(self):
        """Check if set_run_cfg method raises error provided wrong parameters,
        here a string instead of a run config object."""
        converter = Float2FixedConverter()
        run_cfg = Loihi1SimCfg()

        # Check if TypeError is raised for first argument.
        with self.assertRaises(TypeError):
            converter.set_run_cfg(floating_pt_run_cfg='floating_point_1',
                                  fixed_pt_run_cfg=run_cfg)

        # Check if TypeError is raised for second argument.
        with self.assertRaises(TypeError):
            converter.set_run_cfg(floating_pt_run_cfg=run_cfg,
                                  fixed_pt_run_cfg='fixed_point')

    def test_set_procs_no_find_connected_procs(self):
        """Check if _set_procs sets processes correctly when passing single
        Process and disable find_connected_procs. Output should be only passed
        Process."""
        proc1 = LIF(shape=(1,))
        proc2 = Dense(weights=np.array([[1]]))
        proc3 = LIF(shape=(1,))

        proc1.s_out.connect(proc2.s_in)
        proc2.a_out.connect(proc3.a_in)

        find_connected_procs = False

        true_procs = {proc1.id: proc1}
        converter = Float2FixedConverter()

        floating_pt_run_cfg = Loihi1SimCfg()
        fixed_pt_run_cfg = Loihi1SimCfg(select_tag='fixed_pt')

        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)

        converter._set_procs(proc=proc1,
                             find_connected_procs=find_connected_procs)
        procs = converter.procs

        # Check if set proc equals true proc.
        self.assertDictEqual(procs, true_procs)

    def test_set_procs_find_connected_procs(self):
        """Check if _set_procs sets processes correctly when passing single
        Process and enabled find_connected_procs. Output should be a dictionary
        constructed from Processes connected to passed Processes."""
        proc1 = LIF(shape=(1,))
        proc2 = Dense(weights=np.array([[1]]))
        proc3 = LIF(shape=(1,))

        proc1.s_out.connect(proc2.s_in)
        proc2.a_out.connect(proc3.a_in)

        find_connected_procs = True

        true_procs = {proc1.id: proc1,
                      proc2.id: proc2,
                      proc3.id: proc3}
        converter = Float2FixedConverter()

        floating_pt_run_cfg = Loihi1SimCfg()
        fixed_pt_run_cfg = Loihi1SimCfg(select_tag='fixed_pt')

        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)

        converter._set_procs(proc=proc1,
                             find_connected_procs=find_connected_procs)
        procs = converter.procs

        # Check if set proc equals true proc.
        self.assertSetEqual(set(procs), set(true_procs))

    def test_set_procs_procs_list_find_connected_procs(self):
        """Check if _set_procs sets processes correctly when passing list of
        Processes. 'find_connected_procs' should be ignored here and the output
        should be dictionary constructed from passed Processes."""
        proc1 = LIF(shape=(1,))
        proc2 = Dense(weights=np.array([[1]]))
        proc3 = LIF(shape=(1,))

        proc1.s_out.connect(proc2.s_in)
        proc2.a_out.connect(proc3.a_in)

        find_connected_procs = True

        proc_list = [proc1, proc2, proc3]
        true_procs = {proc1.id: proc1,
                      proc2.id: proc2,
                      proc3.id: proc3}

        converter = Float2FixedConverter()

        floating_pt_run_cfg = Loihi1SimCfg()
        fixed_pt_run_cfg = Loihi1SimCfg(select_tag='fixed_pt')

        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)

        converter._set_procs(proc=proc_list,
                             find_connected_procs=find_connected_procs)
        procs = converter.procs

        # Check if set proc equals true proc.
        self.assertDictEqual(procs, true_procs)

    def test_set_procs_procs_list_no_find_connected_procs(self):
        """Check if _set_procs sets processes correctly when passing list of
        Processes. Output should be a dictionary constructed from passed
        Processes."""
        proc1 = LIF(shape=(1,))
        proc2 = Dense(weights=np.array([[1]]))
        proc3 = LIF(shape=(1,))

        proc1.s_out.connect(proc2.s_in)
        proc2.a_out.connect(proc3.a_in)

        find_connected_procs = False

        proc_list = [proc1, proc2, proc3]
        true_procs = {proc1.id : proc1,
                      proc2.id: proc2,
                      proc3.id: proc3}
        converter = Float2FixedConverter()

        floating_pt_run_cfg = Loihi1SimCfg()
        fixed_pt_run_cfg = Loihi1SimCfg(select_tag='fixed_pt')

        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)

        converter._set_procs(proc=proc_list,
                             find_connected_procs=find_connected_procs)
        procs = converter.procs

        # Check if set proc equals true proc.
        self.assertDictEqual(procs, true_procs)

    def test_set_procs_procs_raise_error_wrong_argument(self):
        """Check if _set_procs raises an error when passed argument is not a
        list and not a Process."""
        proc = LIF(shape=(1,))
        converter = Float2FixedConverter()

        # Check if TypeError is raised when neither Procs of List of Procs is
        # passed.
        with self.assertRaises(TypeError):
            converter._set_procs(proc=proc.id)

    def test_set_procs_procs_raise_error_wrong_list_member(self):
        """Check if _set_procs raises error when list passed is not list of
        Processes."""
        proc = LIF(shape=(1,))
        converter = Float2FixedConverter()

        # Check if TypeError raised when list not containing Procs is passed.
        with self.assertRaises(TypeError):
            converter._set_procs(proc=[proc.id])

    def test_set_procs_hierarchical_process(self):
        """Check if _set_procs sets procs and hierarchical_procs correctly when
        passing an hierarchical Process, that is, the hierarchical Process
        should be stored in hierarchical_procs, the Subprocesses on the proc.
        """
        proc1 = HierProc()
        proc2 = ProcDense()

        proc1.outport.connect(proc2.inport)

        true_procs_type = {Proc, ProcDense}
        true_hierarchical_procs = {proc1.id: proc1}

        converter = Float2FixedConverter()

        floating_pt_run_cfg = Loihi1SimCfg(select_tag='floating_pt_1',
                                           select_sub_proc_model=True)
        fixed_pt_run_cfg = Loihi1SimCfg()

        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)

        converter._set_procs(proc=[proc1, proc2])

        procs = converter.procs
        procs_type = set([type(p) for p in procs.values()])
        hierarchical_procs = converter.hierarchical_procs

        # Check if split in procs and hierarchical_procs resulted in corrected
        # type for procs.
        self.assertSetEqual(procs_type, true_procs_type)

        #  Check if hierarchical_procs gets filled correctly.
        self.assertDictEqual(hierarchical_procs, true_hierarchical_procs)

    def test_set_procs_ignores_monitor(self):
        """Check if a Monitor Process is ignored by _set_procs, i.e. the
        Process does not appear in procs."""
        proc1 = Monitor()

        # True procs is empty becasue Monitors Processes are disregarded.
        true_procs = {}

        converter = Float2FixedConverter()

        converter.set_run_cfg(floating_pt_run_cfg=Loihi1SimCfg(),
                              fixed_pt_run_cfg=Loihi1SimCfg())

        converter._set_procs(proc=[proc1])

        procs = converter.procs

        self.assertDictEqual(procs, true_procs)

    def test_set_procs_ignores_ringbuffer(self):
        """Check if a sink RingBuffer Process is ignored by _set_procs, i.e.
        the Process does not appear in procs."""
        proc1 = RingBuffer(shape=(1,), buffer=0)
        proc2 = Proc()

        # True procs only contains proc2 because sink RingBuffers are
        # disregarded.
        true_procs = {proc2.id: proc2}

        converter = Float2FixedConverter()

        converter.set_run_cfg(floating_pt_run_cfg=Loihi1SimCfg(),
                              fixed_pt_run_cfg=Loihi1SimCfg())

        converter._set_procs(proc=[proc1, proc2])

        procs = converter.procs

        self.assertDictEqual(procs, true_procs)

    def test_get_var_ports_outports(self):
        """Check if Float2Fixed Converter correctly identifies Vars and
        SourcePorts needed for conversion, that is writing all Variables and
        Ports which inherits from AbstractSrcPort in a nested dictionary
        structured by the Process ID and the type (Var, Port) and the
        respective Name."""
        proc1 = Proc()
        proc2 = ProcDense()

        true_var_ports = {}
        true_var_ports[proc1.id] = {'Var': {'v', 'u'},
                                    'SourcePort': {'outport'}}
        true_var_ports[proc2.id] = {'Var': {'w'},
                                    'SourcePort': {'outport'}}

        converter = Float2FixedConverter()

        floating_pt_run_cfg = Loihi1SimCfg()
        fixed_pt_run_cfg = Loihi1SimCfg(select_tag='fixed_pt_1')

        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)

        converter._set_procs([proc1, proc2])

        var_ports = converter._get_var_ports()

        # Check if dicts are equal.
        self.assertDictEqual(var_ports, true_var_ports)

    def test_get_var_ports_refports(self):
        """Check if Float2Fixed Converter correctly identifies Vars and
        RefPorts needed for conversion."""
        proc1 = ProcRef()

        true_var_ports = {}
        true_var_ports[proc1.id] = {'Var': {'v', 'w', 'u'},
                                    'SourcePort': {'refport'}}

        converter = Float2FixedConverter()

        floating_pt_run_cfg = Loihi1SimCfg()
        fixed_pt_run_cfg = Loihi1SimCfg(select_tag='fixed_pt')

        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)

        converter._set_procs([proc1])

        var_ports = converter._get_var_ports()

        # Check if dicts of primitives are equal.
        self.assertDictEqual(var_ports, true_var_ports)

    def test_update_target_ports(self):
        """Check if _update_target_ports correctly traces the connection to a
        hierarchical Process to the relevant Subprocesses."""
        proc1 = HierProc()
        proc2 = HierProc()
        proc3 = ProcDense()

        proc1.outport.connect(proc2.inport)
        proc1.outport.connect(proc3.inport)

        true_target_ports_types = {Proc, ProcDense}

        converter = Float2FixedConverter()

        floating_pt_run_cfg = Loihi1SimCfg(select_tag='floating_pt_1',
                                           select_sub_proc_model=True)
        fixed_pt_run_cfg = Loihi1SimCfg()

        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)

        converter._set_procs([proc1, proc2, proc3])
        converter.var_ports = converter._get_var_ports()

        out_port = converter.hierarchical_procs[proc1.id].outport
        target_ports = out_port.out_connections
        target_ports = converter._update_target_ports(target_ports)

        target_port_types = set([type(port.process) for port in target_ports])

        self.assertSetEqual(target_port_types, true_target_ports_types)

    def test_explicate_hierarchical_procs(self):
        """Check if _explicate_hierarchical_procs correctly sets up an
        equivalent model with only first-order Processes, i.e. if an
        hierarchical Process consists of only Proc Process, instantiate a Proc
        process instead. If the hierarchical Process contains multiple
        Subprocesses which are not hierarchical Processes, all of them need to
        be instantiated and connected correctly."""
        proc1 = HierProc()
        proc2 = ProcDense()
        proc3 = HierProc()

        proc1.outport.connect(proc2.inport)
        proc2.outport.connect(proc1.inport)

        true_proc_types = {Proc, ProcDense}
        true_proc_list = [Proc, Proc, ProcDense]
        true_proc1_out_type = [ProcDense]
        true_proc2_out_type = [Proc]

        converter = Float2FixedConverter()

        floating_pt_run_cfg = Loihi1SimCfg(select_tag='floating_pt_1',
                                           select_sub_proc_model=True)
        fixed_pt_run_cfg = Loihi1SimCfg()

        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)

        converter._set_procs([proc1, proc2, proc3])
        converter.var_ports = converter._get_var_ports()
        converter._explicate_hierarchical_procs()

        procs = converter.procs

        proc_types = set([type(p) for p in procs.values()])
        proc_list = [p for p in procs.values()]
        proc1_out_type = [type(o.process) for o in
                          proc_list[0].outport.out_connections]
        proc2_out_type = [type(o.process) for o in
                          proc_list[1].outport.out_connections]

        # Check if setting up of equivalent model worked correctly.
        self.assertSetEqual(proc_types, true_proc_types)
        self.assertEqual(len(proc_list), len(true_proc_list))
        self.assertListEqual(true_proc1_out_type, proc1_out_type)
        self.assertListEqual(true_proc2_out_type, proc2_out_type)

    def test_get_fixed_pt_proc_models(self):
        """Check if Float2Fixed Converter correctly fetches fixed-point
        ProcModels correctly based on the set fixed-point run configuration."""
        proc1 = Proc()
        proc2 = ProcDense()

        # Expected fixed point ProcModels.
        fixed_proc_model_1 = ProcPyProcModel1(proc1.proc_params)
        fixed_proc_model_2 = ProcDensePyProcModel1(proc2.proc_params)
        true_fixed_pt_proc_models = {}
        true_fixed_pt_proc_models[proc1.id] = fixed_proc_model_1
        true_fixed_pt_proc_models[proc2.id] = fixed_proc_model_2

        # Set up converter.
        floating_pt_run_cfg = Loihi1SimCfg()
        fixed_pt_run_cfg = Loihi1SimCfg(select_tag='fixed_pt_1')

        converter = Float2FixedConverter()
        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)
        converter._set_procs([proc1, proc2])
        fixed_pt_proc_models = converter._get_fixed_pt_proc_models()

        # Check if dictionaries and ProcModels haven been set up correctly.
        for proc_id, proc_model in fixed_pt_proc_models.items():
            self.assertEqual(type(proc_model),
                             type(true_fixed_pt_proc_models[proc_id]))

    def test_get_conv_data(self):
        """Check if Float2Fixed Converter correctly fetches conversion data
        needed for conversion of all Processes that need to be converterd and
        stores them in nested dictionary structured by Process ID and Variable
        name."""
        proc1 = Proc()

        true_conv_data = {proc1.id: {}}
        true_conv_data[proc1.id]['u'] = {'is_signed': True,
                                         'num_bits': 17,
                                         'implicit_shift': 6,
                                         'scale_domain': 0,
                                         'domain': 0,
                                         'constant': True,
                                         'num_bits_exp': 4,
                                         'exp_var': 'exp_var'}
        true_conv_data[proc1.id]['v'] = {'is_signed': False,
                                         'num_bits': 24,
                                         'implicit_shift': 0,
                                         'scale_domain': 0,
                                         'domain': None,
                                         'constant': False,
                                         'num_bits_exp': None,
                                         'exp_var': None}

        # Set up Float2FixedPoint Converter.
        floating_pt_run_cfg = Loihi1SimCfg()
        fixed_pt_run_cfg = Loihi1SimCfg(select_tag='fixed_pt_1')

        converter = Float2FixedConverter()
        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)
        converter._set_procs([proc1])
        converter.var_ports = converter._get_var_ports()
        converter.fixed_pt_proc_models = converter._get_fixed_pt_proc_models()

        conv_data = converter._get_conv_data()

        # Check if dicts of precision are equal.
        # Test array with entries separately and remove it from dicts.
        np.testing.assert_array_equal(
            conv_data[proc1.id]['v'].pop('domain'),
            true_conv_data[proc1.id]['v'].pop('domain')
        )
        # Test remaining dict.
        self.assertDictEqual(conv_data, true_conv_data)

    def test_get_conv_data_target_port_error(self):
        """Check if error is caught correctly in function getting the
        conversion data if the target port of a Process has wrong precision."""
        proc1 = Proc()
        proc2 = Proc()

        proc1.outport.connect(proc2.inport)

        # Set up Float2FixedPoint Converter.
        floating_pt_run_cfg = Loihi1SimCfg()
        fixed_pt_run_cfg = Loihi1SimCfg(select_tag='fixed_pt_pytype_target_err')

        converter = Float2FixedConverter()
        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)
        converter._set_procs([proc1, proc2])
        converter.var_ports = converter._get_var_ports()
        converter.fixed_pt_proc_models = converter._get_fixed_pt_proc_models()

        # Check if ValueError raised when target port has wrong precision
        # information.
        with self.assertRaises(ValueError):
            conv_data = converter._get_conv_data()

    def test_get_conv_data_var_error(self):
        """Check if error is caught correctly in function getting the
        conversion data if the Variable of a Process has wrong precision."""
        proc1 = Proc()
        proc2 = Proc()

        proc1.outport.connect(proc2.inport)

        # Set up Float2FixedPoint Converter.
        floating_pt_run_cfg = Loihi1SimCfg()
        fixed_pt_run_cfg = Loihi1SimCfg(select_tag='fixed_pt_pytype_var_err')

        converter = Float2FixedConverter()
        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)
        converter._set_procs([proc1, proc2])
        converter.var_ports = converter._get_var_ports()
        converter.fixed_pt_proc_models = converter._get_fixed_pt_proc_models()

        # Check if ValueError raised when target port has wrong precision
        # information.
        with self.assertRaises(ValueError):
            conv_data = converter._get_conv_data()

    def test_get_conv_data_skip_meta_parameter(self):
        """Check if Float2Fixed Converter correctly fetches conversion data for
        conversion. Variable u is meta_parameter here and should be ignored."""
        proc1 = Proc()

        true_conv_data = {proc1.id: {}}
        true_conv_data[proc1.id]['v'] = {'is_signed': False,
                                         'num_bits': 24,
                                         'implicit_shift': 0,
                                         'scale_domain': 0,
                                         'domain': None,
                                         'constant': False,
                                         'num_bits_exp': None,
                                         'exp_var': None}

        # Set up Float2FixedPoint Converter.
        # In tagged fixed-point ProcessModel Var u is meta parameter and should
        # be skipped bet _get_conv_data.
        floating_pt_run_cfg = Loihi1SimCfg()
        fixed_pt_run_cfg = Loihi1SimCfg(select_tag='fixed_pt_2')

        converter = Float2FixedConverter()
        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)
        converter._set_procs([proc1])
        converter.var_ports = converter._get_var_ports()
        converter.fixed_pt_proc_models = converter._get_fixed_pt_proc_models()

        conv_data = converter._get_conv_data()

        # Check if dicts of precision are equal.
        self.assertDictEqual(conv_data, true_conv_data)

    def test_update_implicit_shift_conv_data_outport(self):
        """Check if Float2Fixed Converter correctly fetches updates implicit
        shift due to out connections via OutPort. Variable u in proc1 is a meta
        parameters and thus does not appear in the conversion data. Implicit
        shift of w in proc2 is increased du to connection from proc2 to proc1
        via OutPort."""
        proc1 = Proc()
        proc2 = ProcDense()

        proc2.outport.connect(proc1.inport)

        true_conv_data = {proc1.id: {},
                          proc2.id: {}}
        true_conv_data[proc1.id]['v'] = {'is_signed': False,
                                         'num_bits': 24,
                                         'implicit_shift': 0,
                                         'scale_domain': 0,
                                         'domain': None,
                                         'constant': False,
                                         'num_bits_exp': None,
                                         'exp_var': None}

        true_conv_data[proc2.id]['w'] = {'is_signed': True,
                                         'num_bits': 16,
                                         'implicit_shift': 9,
                                         'scale_domain': 0,
                                         'domain': 0,
                                         'constant': True,
                                         'num_bits_exp': None,
                                         'exp_var': None}

        # Set up Float2FixedPoint Converter.
        # In tagged fixed-point ProcessModel Var u is meta parameter and should
        # be skipped bet _get_conv_data.
        floating_pt_run_cfg = Loihi1SimCfg()
        fixed_pt_run_cfg = Loihi1SimCfg(select_tag='fixed_pt_2')

        converter = Float2FixedConverter()
        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)
        converter._set_procs([proc1, proc2])
        converter.var_ports = converter._get_var_ports()
        converter.fixed_pt_proc_models = converter._get_fixed_pt_proc_models()

        conv_data = converter._get_conv_data()

        # Check if dicts of precision are equal.
        self.assertDictEqual(conv_data, true_conv_data)

    def test_update_implicit_shift_conv_data_refport(self):
        """Check if Float2Fixed Converter correctly fetches updates implicit
        shift due to out connections via RefPort. Variables u and v in proc1
        need to be updated since the inherit the implicit scaling from Variable
        v of Process proc2. Variables w of proc1 needs to stay the same since
        it is on other scale domain.."""
        proc1 = ProcRef(u=1, v=2, w=3)
        proc2 = ProcVar(v=4)

        proc1.refport.connect_var(proc2.v)

        true_conv_data = {proc1.id: {},
                          proc2.id: {}}
        true_conv_data[proc1.id]['v'] = {'is_signed': True,
                                         'num_bits': 16,
                                         'implicit_shift': 3,
                                         'scale_domain': 0,
                                         'domain': None,
                                         'constant': False,
                                         'num_bits_exp': None,
                                         'exp_var': None}
        true_conv_data[proc1.id]['u'] = {'is_signed': True,
                                         'num_bits': 16,
                                         'implicit_shift': 6,
                                         'scale_domain': 0,
                                         'domain': None,
                                         'constant': False,
                                         'num_bits_exp': None,
                                         'exp_var': None}
        true_conv_data[proc1.id]['w'] = {'is_signed': False,
                                         'num_bits': 12,
                                         'implicit_shift': 0,
                                         'scale_domain': 1,
                                         'domain': 3,
                                         'constant': True,
                                         'num_bits_exp': None,
                                         'exp_var': None}
        true_conv_data[proc2.id]['v'] = {'is_signed': True,
                                         'num_bits': 16,
                                         'implicit_shift': 3,
                                         'scale_domain': 0,
                                         'domain': None,
                                         'constant': False,
                                         'num_bits_exp': None,
                                         'exp_var': None}

        # Set up Float2FixedPoint Converter.
        floating_pt_run_cfg = Loihi1SimCfg()
        fixed_pt_run_cfg = Loihi1SimCfg(select_tag='fixed_pt')

        converter = Float2FixedConverter()
        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)
        converter._set_procs([proc1, proc2])
        converter.var_ports = converter._get_var_ports()
        converter.fixed_pt_proc_models = converter._get_fixed_pt_proc_models()

        conv_data = converter._get_conv_data()
        # Check if dicts of precision are equal.
        self.assertDictEqual(conv_data, true_conv_data)

    def test_check_implicit_shift_conv_data_consistency(self):
        """Check if Float2Fixed Converter correctly throws error if implicit
        shift updates due to out connections are inconsistent."""
        proc1 = Proc()
        proc2 = ProcDense()
        proc3 = ProcDense()

        # Proc1 and Proc3 result in inconsistent implicit shift update.
        # Inport of proc1 scales up by 2^6.
        proc2.outport.connect(proc1.inport)
        proc2.outport.connect(proc3.inport)

        # Set up Float2FixedPoint Converter.
        # In tagged ProcModel u is meta parameter.
        floating_pt_run_cfg = Loihi1SimCfg()
        fixed_pt_run_cfg = Loihi1SimCfg(select_tag='fixed_pt_2')

        converter = Float2FixedConverter()
        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)
        converter._set_procs([proc1, proc2, proc3])
        converter.var_ports = converter._get_var_ports()
        converter.fixed_pt_proc_models = converter._get_fixed_pt_proc_models()

        with self.assertRaises(ValueError):
            conv_data = converter._get_conv_data()

    def test_monitoring_infrastructure_procs_num(self):
        """Check if Float2Fixed Converter correctly sets up monitoring
        infrastructure for recording dynamic range of variables. Here we check
        if no other Processes than monitors and the correct number get
        instantiated."""
        proc1 = Proc()
        proc2 = Proc()
        proc_list = [proc1, proc2]
        proc1.outport.connect(proc2.inport)

        # Create dummy monitor.
        m = Monitor()

        # Create list with true types that ought to be created by function
        # building monitoring infrastructure:
        # proc1 -> variales u and v
        # proc2 -> variales u and v
        true_proc_type_list = [type(proc1), type(proc2), type(m), type(m),
                               type(m), type(m)]

        # Set up Float2FixedPoint Converter.
        # In tagged ProcModel u and v are both dynamic variables.
        floating_pt_run_cfg = Loihi1SimCfg()
        fixed_pt_run_cfg = Loihi1SimCfg(select_tag='fixed_pt_4')

        converter = Float2FixedConverter()
        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)
        converter._set_procs(proc_list)
        converter.var_ports = converter._get_var_ports()
        converter.fixed_pt_proc_models = converter._get_fixed_pt_proc_models()

        converter.conv_data = converter._get_conv_data()

        monitors = converter._create_monitoring_infrastructure()

        proc_type_list = [type(p) for p in proc_list]

        for p in proc_list:
            # Fetch dictionary storing {TargetVarName: monitor} for Process.
            monitor_dict = monitors[p.id]
            for var, monitor in monitor_dict.items():
                proc_type_list.append(type(monitor))

        # Test that correct number of monitors got created.
        self.assertEqual(len(true_proc_type_list), len(proc_type_list))

        # Test that correct processes got instantiated.
        self.assertSetEqual(set(true_proc_type_list), set(proc_type_list))

    def test_monitoring_infrastructure_connections(self):
        """Check if Float2Fixed Converter correctly sets up monitoring
        infrastructure for recording dynamic range of variables. Here we check
        if the Processes stored in the constructed dictionary are Monitors and
        connect to the correct Vars."""
        proc1 = Proc()
        proc2 = Proc()
        proc_list = [proc1, proc2]
        proc1.outport.connect(proc2.inport)

        # Set up Float2FixedPoint Converter.
        # In tagged ProcModel u and v are both dynamic variables.
        floating_pt_run_cfg = Loihi1SimCfg()
        fixed_pt_run_cfg = Loihi1SimCfg(select_tag='fixed_pt_4')

        converter = Float2FixedConverter()
        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)
        converter._set_procs(proc_list)
        converter.var_ports = converter._get_var_ports()
        converter.fixed_pt_proc_models = converter._get_fixed_pt_proc_models()

        converter.conv_data = converter._get_conv_data()

        monitors = converter._create_monitoring_infrastructure()

        proc_type_list = [type(p) for p in proc_list]

        # Check if monitors target correct Vars.
        for p in proc_list:
            # Fetch dictionary storing {TargetVarName: monitor} for Process.
            monitor_dict = monitors[p.id]
            for var, monitor in monitor_dict.items():
                # Check if Process stored in dictionary is monitor.
                self.assertIsInstance(monitor, Monitor)
                # Get true target of monitor
                true_target = p.__getattribute__(var)
                # Check if monitor target is correct
                self.assertEqual(monitor.ref_port_0.out_connections[0].var,
                                 true_target)

    def test_monitoring_infrastructure_w_constant(self):
        """Check if Float2Fixed Converter correctly sets up monitoring
        infrastructure for recording dynamic range of variables. Variable u is
        a constant. For this Variable no Monitor should be instantiated."""
        proc1 = Proc()

        true_monitors = {proc1.id: {'v': None}}

        # Set up Float2FixedPoint Converter.
        # In tagged ProcModel u is constant.
        floating_pt_run_cfg = Loihi1SimCfg()
        fixed_pt_run_cfg = Loihi1SimCfg(select_tag='fixed_pt_1')

        converter = Float2FixedConverter()
        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)
        converter._set_procs([proc1])
        converter.var_ports = converter._get_var_ports()
        converter.fixed_pt_proc_models = converter._get_fixed_pt_proc_models()

        converter.conv_data = converter._get_conv_data()

        monitors = converter._create_monitoring_infrastructure()

        # Only one additional proc connected to proc1, namely internally
        # generated monitor.
        procs = compiler_graphs.find_processes(proc1)
        for proc in procs:
            if isinstance(proc, Monitor):
                true_monitors[proc1.id]['v'] = proc

        # Check if correct number of processes get created.
        self.assertDictEqual(true_monitors, monitors)

    def test_monitoring_infrastructure_w_domain(self):
        """Check if Float2Fixed Converter correctly sets up monitoring
        infrastructure for recording dynamic range of variables. Varaible u has
        pre-defined domain and thus no Monitor should be instantiated for this
        variable."""
        proc1 = Proc()

        true_monitors = {proc1.id: {'v': None}}

        # Set up Float2FixedPoint Converter.
        # In tagged ProcModel u has predefined domain.
        floating_pt_run_cfg = Loihi1SimCfg()
        fixed_pt_run_cfg = Loihi1SimCfg(select_tag='fixed_pt_3')

        converter = Float2FixedConverter()
        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)
        converter._set_procs([proc1])
        converter.var_ports = converter._get_var_ports()
        converter.fixed_pt_proc_models = converter._get_fixed_pt_proc_models()

        converter.conv_data = converter._get_conv_data()

        monitors = converter._create_monitoring_infrastructure()

        # Only one additional proc connected to proc1, namely internally
        # generated monitor.
        procs = compiler_graphs.find_processes(proc1)
        for proc in procs:
            if isinstance(proc, Monitor):
                true_monitors[proc1.id]['v'] = proc

        # Check if correct number of processes get created.
        self.assertDictEqual(true_monitors, monitors)

    def test_run_procs(self):
        """Check if Processes are run for correct number of time steps and the
        correct output is stored in nested dictionary with keys Process ID,
        Variable names."""
        proc1 = Proc(u=0, v=0)
        proc2 = Proc(u=1, v=2)
        proc_list = [proc1, proc2]
        proc1.outport.connect(proc2.inport)

        num_steps = 2

        var_names = ['u', 'v']

        true_domains = {}
        true_domains[proc1.id] = {'u': np.array([1, 2]),
                                  'v': np.array([2, 4])}
        true_domains[proc2.id] = {'u': np.array([2, 3]),
                                  'v': np.array([4, 6])}

        # Set up Float2FixedPoint Converter.
        # In tagged ProcModel u and v are both dynamic variables.
        floating_pt_run_cfg = Loihi1SimCfg(select_tag='floating_pt_1')
        fixed_pt_run_cfg = Loihi1SimCfg(select_tag='fixed_pt_4')

        converter = Float2FixedConverter()
        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)
        converter._set_procs(proc_list)
        converter.var_ports = converter._get_var_ports()
        converter.fixed_pt_proc_models = converter._get_fixed_pt_proc_models()
        converter.conv_data = converter._get_conv_data()
        converter.num_steps = num_steps
        converter.monitors = converter._create_monitoring_infrastructure()

        converter._run_procs()

        for p_id, conv_data_p_id in converter.conv_data.items():
            for var in var_names:
                np.testing.assert_array_equal(true_domains[p_id][var],
                                              conv_data_p_id[var]['domain'])

    def test_get_scale_domains(self):
        """Check if scale domain dictionaries are created correctly. Here
        Variable u of proc1 has own scale domain and must stored seperately."""
        proc1 = Proc(u=1, v=0)
        proc2 = ProcDense(w=0)
        proc_list = [proc1, proc2]

        proc2.outport.connect(proc1.inport)

        true_scale_domains = {0: {proc1.id: {},
                                  proc2.id: {}},
                              1: {proc1.id: {}}}

        true_scale_domains[0][proc1.id]['v'] = {'is_signed': False,
                                                'num_bits': 24,
                                                'implicit_shift': 0,
                                                'domain': np.array([2, 4]),
                                                'num_bits_exp': 0}

        true_scale_domains[0][proc2.id]['w'] = {'is_signed': True,
                                                'num_bits': 16,
                                                'implicit_shift': 6,
                                                'domain': 0,
                                                'num_bits_exp': 0}

        true_scale_domains[1][proc1.id]['u'] = {'is_signed': True,
                                                'num_bits': 17,
                                                'implicit_shift': 6,
                                                'domain': 1,
                                                'num_bits_exp': 4}

        # Set up Float2FixedPoint Converter.
        # In tagged ProcModel only v of proc1 is a dynamic variables and
        # belongs to the global scale domain, together with w of proc2.
        # The variable u of proc1 belongs to a different scale domain.
        floating_pt_run_cfg = Loihi1SimCfg(select_tag='floating_pt_2')
        fixed_pt_run_cfg = Loihi1SimCfg(select_tag='fixed_pt_5')

        num_steps = 2

        converter = Float2FixedConverter()
        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)
        converter._set_procs(proc_list)
        converter.var_ports = converter._get_var_ports()
        converter.fixed_pt_proc_models = converter._get_fixed_pt_proc_models()
        converter.conv_data = converter._get_conv_data()
        converter.num_steps = num_steps
        converter.monitors = converter._create_monitoring_infrastructure()
        converter._run_procs()
        scale_domains = converter._update_scale_domains()

        # Test np.array domain for proc1 separately.
        np.testing.assert_array_equal(
            scale_domains[0][proc1.id]['v'].pop('domain'),
            true_scale_domains[0][proc1.id]['v'].pop('domain')
        )

        # Test remaining scale domain dictionary.
        self.assertDictEqual(true_scale_domains, scale_domains)

    def test_find_scaling_factors(self):
        """Check if scaling functions are determined correctly for the scale
        domains from the conversion data and the dynamic range."""

        # Construct scale domains to test finding of scaling functions.
        scale_domains = {0: {1: {},
                             3: {}},
                         1: {1: {},
                             3 : {}}}

        scale_domains[0][1]['v'] = {'is_signed': False,
                                    'num_bits': 24,
                                    'implicit_shift': 0,
                                    'domain': np.array([2, 4]),
                                    'num_bits_exp': 0}
        scale_domains[0][3]['w'] = {'is_signed': True,
                                    'num_bits': 16,
                                    'implicit_shift': 6,
                                    'domain': 3,
                                    'num_bits_exp': 0}
        scale_domains[1][1]['u'] = {'is_signed': True,
                                    'num_bits': 17,
                                    'implicit_shift': 0,
                                    'domain': 2.5,
                                    'num_bits_exp': 2}
        scale_domains[1][1]['x'] = {'is_signed': False,
                                    'num_bits': 6,
                                    'implicit_shift': 1,
                                    'domain': 0.5,
                                    'num_bits_exp': 0}
        scale_domains[1][3]['w'] = {'is_signed': True,
                                    'num_bits': 16,
                                    'implicit_shift': 2,
                                    'domain': 30,
                                    'num_bits_exp': 2}

        # Calculate the allowed factors manually and select the smaller one.
        factor_0 = np.min([(2 ** 24 - 1) / 4,
                           (2 ** 15 - 1) * 2 ** 6 / 3])
        factor_1_1 = np.min([(2 ** 16 - 1) * 2 ** 3 / 2.5,
                             (2 ** 6 - 1) * 2 / 0.5])
        factor_1_3 = (2 ** 15 - 1) * 2 ** 2 * 2 ** 3 / 30

        true_scaling_factors = {0: factor_0,
                                1: {1: factor_1_1,
                                    3: factor_1_3}}

        # Need to set up dummy monitor dictionary for use in tested function.
        monitors = {1: {},
                    3: {}}

        converter = Float2FixedConverter()
        converter.monitors = monitors
        converter.quantiles = [0, 1]
        converter.scale_domains = scale_domains
        scaling_factors = converter._find_scaling_factors()

        self.assertDictEqual(scaling_factors, true_scaling_factors)

    def test_find_scaling_factors_quantiles(self):
        """Check if scaling functions are determined correctly if dynamic range
        is determined via quantiles from domain."""

        # Construct scale domains to test finding of scaling functions.
        scale_domains = {0: {1: {},
                             3: {}},
                         1: {1: {},
                             3 : {}}}

        scale_domains[0][1]['v'] = {'is_signed': False,
                                    'num_bits': 24,
                                    'implicit_shift': 0,
                                    'domain': np.array([0, 1, 2, 3, 4, 5]),
                                    'num_bits_exp': 0}
        scale_domains[0][3]['w'] = {'is_signed': True,
                                    'num_bits': 16,
                                    'implicit_shift': 6,
                                    'domain': 0.1,
                                    'num_bits_exp': 0}
        scale_domains[1][1]['u'] = {'is_signed': True,
                                    'num_bits': 17,
                                    'implicit_shift': 0,
                                    'domain': 2.5,
                                    'num_bits_exp': 2}
        scale_domains[1][1]['x'] = {'is_signed': False,
                                    'num_bits': 6,
                                    'implicit_shift': 1,
                                    'domain': 0.5,
                                    'num_bits_exp': 0}
        scale_domains[1][3]['w'] = {'is_signed': True,
                                    'num_bits': 16,
                                    'implicit_shift': 2,
                                    'domain': 30,
                                    'num_bits_exp': 2}

        # Calculate the allowed factors manually and select the smaller one.
        # Choose 4 for division because 0.8-quantile of [0,1,2,3,4,5] is 4.
        factor_0 = np.min([(2 ** 24 - 1) / 4,
                           (2 ** 15 - 1) * 2 ** 6 / 0.1])
        factor_1_1 = np.min([(2 ** 16 - 1) * 2 ** 3 / 2.5,
                             (2 ** 6 - 1) * 2 / 0.5])
        factor_1_3 = (2 ** 15 - 1) * 2 ** 2 * 2 ** 3 / 30

        true_scaling_factors = {0: factor_0,
                                1: {1: factor_1_1,
                                    3: factor_1_3}}

        # Need to set up dummy monitor dictionary for use in tested function.
        # Want to apply quantile only to 'v'.
        monitors = {1: {'v': {}},
                    3: {}}

        converter = Float2FixedConverter()
        # Set quantiles so that the higher values of variable 'v' in dynamic
        # range will be 4.
        converter.quantiles = [0, 0.8]
        converter.monitors = monitors
        converter.scale_domains = scale_domains
        scaling_factors = converter._find_scaling_factors()

        self.assertDictEqual(scaling_factors, true_scaling_factors)

    def test_scale_parameters(self):
        """Test scaling and storing of the parameters of the Processes that
        need to be converted from floating- to fixed-point representation.
        """
        proc1 = Proc(u=1, v=1)
        proc2 = ProcDense(w=3)
        proc_list = [proc1, proc2]

        proc2.outport.connect(proc1.inport)

        scale_domains = {0: {proc1.id: {},
                             proc2.id: {}}}

        scale_domains[0][proc1.id]['v'] = {'is_signed': False,
                                           'num_bits': 24,
                                           'implicit_shift': 0,
                                           'domain': np.array([3, 5]),
                                           'num_bits_exp': 0}

        scale_domains[0][proc2.id]['w'] = {'is_signed': True,
                                           'num_bits': 16,
                                           'implicit_shift': 9,
                                           'domain': 3,
                                           'num_bits_exp': 0}

        scaling_factor = np.min([(2 ** 24 - 1) / 5,
                                 (2 ** 15 - 1) * 2 ** 9 / 3])

        true_scaled_params = {proc1.id: {},
                              proc2.id: {}}

        # For v we have to choose the initial value for the parameter mapping.
        true_scaled_params[proc1.id]['v'] = np.round(scaling_factor * 1)
        true_scaled_params[proc1.id]['u'] = 1
        true_scaled_params[proc2.id]['w'] = np.round(scaling_factor
                                                     * 3 / 2 ** 9).astype(int)

        # Set up Float2FixedPoint Converter.
        # In tagged ProcModel only v of proc1 is a dynamic variables and
        # belongs to the global scale domain, together with w of proc2.
        # The variable u is a meta-parameter.
        floating_pt_run_cfg = Loihi1SimCfg(select_tag='floating_pt_2')
        fixed_pt_run_cfg = Loihi1SimCfg(select_tag='fixed_pt_2')

        converter = Float2FixedConverter()
        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)
        converter._set_procs(proc_list)
        converter.var_ports = converter._get_var_ports()
        converter.fixed_pt_proc_models = converter._get_fixed_pt_proc_models()
        converter.conv_data = converter._get_conv_data()
        converter.monitors = converter._create_monitoring_infrastructure()
        converter.scale_domains = scale_domains
        converter.quantiles = [0, 1]
        converter.scaling_factors = converter._find_scaling_factors()
        scaled_params = converter._scale_parameters()

        # Test scaled parameter dictionary.
        self.assertDictEqual(scaled_params, true_scaled_params)

    def test_scale_parameters_exp_var_exp_val_zero(self):
        """Test scaling and storing of the parameters of the Processes that
        need to be converted from floating- to fixed-point representation. In
        this test a variable with a split in exponent and mantissa is zero,
        which triggers a separates case in the scale_parameters function.
        """
        proc1 = Proc(u=20, v=1)
        proc2 = ProcDense(w=0)
        proc_list = [proc1, proc2]

        proc2.outport.connect(proc1.inport)

        scale_domains = {0: {proc1.id: {},
                             proc2.id: {}}}

        scale_domains[0][proc1.id]['v'] = {'is_signed': False,
                                           'num_bits': 24,
                                           'implicit_shift': 0,
                                           'domain': np.array([1]),
                                           'num_bits_exp': 0}

        scale_domains[0][proc2.id]['w'] = {'is_signed': True,
                                           'num_bits': 16,
                                           'implicit_shift': 1,
                                           'domain': 0,
                                           'num_bits_exp': 2}

        scaling_factor = (2 ** 24 - 1) / 1

        true_scaled_params = {proc1.id: {},
                              proc2.id: {}}

        # For v we have to choose the initial value for the parameter mapping.
        true_scaled_params[proc1.id]['v'] = int(np.round(scaling_factor * 1))
        true_scaled_params[proc1.id]['u'] = 20

        # Split w in mantissa and exponent, both are zero and determined as
        # special case.
        true_scaled_params[proc2.id]['w'] = 0
        true_scaled_params[proc2.id]['exp_w'] = 0

        # Set up Float2FixedPoint Converter.
        # In tagged ProcModel only v of proc1 is a dynamic variables and
        # belongs to the global scale domain, together with w of proc2.
        # The variable u is a meta-parameter.
        floating_pt_run_cfg = Loihi1SimCfg(select_tag='floating_pt_2')
        fixed_pt_run_cfg = Loihi1SimCfg(select_tag='fixed_pt_6')

        converter = Float2FixedConverter()
        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)
        converter._set_procs(proc_list)
        converter.var_ports = converter._get_var_ports()
        converter.fixed_pt_proc_models = converter._get_fixed_pt_proc_models()
        converter.conv_data = converter._get_conv_data()
        converter.monitors = converter._create_monitoring_infrastructure()
        converter.scale_domains = scale_domains
        converter.quantiles = [0, 1]
        converter.scaling_factors = converter._find_scaling_factors()
        scaled_params = converter._scale_parameters()

        # Test scaled parameter dictionary.
        self.assertDictEqual(scaled_params, true_scaled_params)

    def test_converter_no_exp_var(self):
        """Test scaling and storing of the parameters of the Processes that
        need to be converted from floating- to fixed-point representation.
        Here, no Variable has a split in mantissa and exponent.
        """
        proc1 = Proc(u=20, v=1)
        proc2 = ProcDense(w=3)
        proc_list = [proc1, proc2]

        proc2.outport.connect(proc1.inport)

        scale_domains = {0: {proc1.id: {},
                             proc2.id: {}}}

        scale_domains[0][proc1.id]['v'] = {'is_signed': False,
                                           'num_bits': 24,
                                           'implicit_shift': 0,
                                           'domain': np.array([3, 5]),
                                           'num_bits_exp': 0}

        scale_domains[0][proc2.id]['w'] = {'is_signed': True,
                                           'num_bits': 16,
                                           'implicit_shift': 9,
                                           'domain': 3,
                                           'num_bits_exp': 0}

        scaling_factor = np.min([(2 ** 24 - 1) / 5,
                                 (2 ** 15 - 1) * 2 ** 9 / 3])

        true_scaled_params = {proc1.id: {},
                              proc2.id: {}}

        # For v we have to choose the inital value for the parameter mapping.
        true_scaled_params[proc1.id]['v'] = np.round(scaling_factor * 1)
        true_scaled_params[proc1.id]['u'] = 20
        true_scaled_params[proc2.id]['w'] = np.round(scaling_factor
                                                     * 3 / 2 ** 9).astype(int)

        # Set up Float2FixedPoint Converter.
        # In tagged ProcModel only v of proc1 is a dynamic variables and
        # belongs to the global scale domain, together with w of proc2.
        # The variable u is a meta-parameter.
        floating_pt_run_cfg = Loihi1SimCfg(select_tag='floating_pt_2')
        fixed_pt_run_cfg = Loihi1SimCfg(select_tag='fixed_pt_2')

        num_steps = 2

        converter = Float2FixedConverter()
        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)
        converter.convert(proc_list, num_steps=num_steps)
        scaled_params = converter.scaled_params

        # Test scaled parameter dictionary.
        self.assertDictEqual(scaled_params, true_scaled_params)

    def test_converter_exp_var(self):
        """Test scaling and storing of the parameters of the Processes that
        need to be converted from floating- to fixed-point representation. Here
        a Variable in split in mantissa and exponent (Variable w of proc2).
        """
        proc1 = Proc(u=20, v=1)
        proc2 = ProcDense(w=3)
        proc_list = [proc1, proc2]

        proc2.outport.connect(proc1.inport)

        scale_domains = {0: {proc1.id: {},
                             proc2.id: {}}}

        scale_domains[0][proc1.id]['v'] = {'is_signed': False,
                                           'num_bits': 24,
                                           'implicit_shift': 0,
                                           'domain': np.array([3, 5]),
                                           'num_bits_exp': 0}

        scale_domains[0][proc2.id]['w'] = {'is_signed': True,
                                           'num_bits': 16,
                                           'implicit_shift': 1,
                                           'domain': 3,
                                           'num_bits_exp': 2}

        scaling_factor = np.min([(2 ** 24 - 1) / 5,
                                 (2 ** 15 - 1) * 2 ** 1 * 2 ** 3 / 3])

        true_scaled_params = {proc1.id: {},
                              proc2.id: {}}

        # For v we have to choose the initial value for the parameter mapping.
        true_scaled_params[proc1.id]['v'] = int(np.round(scaling_factor * 1))
        true_scaled_params[proc1.id]['u'] = 20

        # Split w in mantissa and exponent.
        w_scaled = np.round((scaling_factor * 3)).astype(int)
        w_scaled = np.right_shift(w_scaled, 1)  # Due to implicit shift.
        w_exp = (np.round(np.log2(w_scaled)) - 16 + 1).astype(int)
        w_mant = np.right_shift(w_scaled, w_exp)
        true_scaled_params[proc2.id]['w'] = w_mant
        true_scaled_params[proc2.id]['exp_w'] = w_exp

        # Set up Float2FixedPoint Converter.
        # In tagged ProcModel only v of proc1 is a dynamic variables and
        # belongs to the global scale domain, together with w of proc2.
        # The variable u is a meta-parameter.
        floating_pt_run_cfg = Loihi1SimCfg(select_tag='floating_pt_2')
        fixed_pt_run_cfg = Loihi1SimCfg(select_tag='fixed_pt_6')

        num_steps = 2

        converter = Float2FixedConverter()
        converter.set_run_cfg(floating_pt_run_cfg=floating_pt_run_cfg,
                              fixed_pt_run_cfg=fixed_pt_run_cfg)
        converter.convert(proc_list, num_steps=num_steps)
        scaled_params = converter.scaled_params
        # Test scaled parameter dictionary.
        self.assertDictEqual(scaled_params, true_scaled_params)

    def test_plot_var_create_figure(self):
        """Test whether `plot_var' correctly creates a matplotlib.figure.Figure
        object if none is passed.
        """
        # Create dummy ProcessID, Variable and domain
        p_id = 0
        var = 'v'
        domain = np.random.normal(0, 1, 5)

        conv_data = {p_id: {var: {'domain': domain}}}

        # Instantiate and set up converter.
        converter = Float2FixedConverter()
        converter.conv_data = conv_data
        converter.quantiles = (0, 1)

        fig = converter.plot_var(p_id, var)

        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_plot_var_reuse_figure(self):
        """Test whether `plot_var' correctly reuses a matplotlib.figure.Figure
        object if one is passed.
        """
        # Create matplotlib.figure.Figure object
        fig = matplotlib.pyplot.figure()

        # Create dummy ProcessID, Variable and domain
        p_id = 0
        var = 'v'
        domain = np.random.normal(0, 1, 5)

        conv_data = {p_id: {var: {'domain': domain}}}

        # Instantiate and set up converter.
        converter = Float2FixedConverter()
        converter.conv_data = conv_data
        converter.quantiles = (0, 1)

        fig_return = converter.plot_var(p_id, var, fig=fig)

        self.assertEqual(fig, fig_return)


if __name__ == '__main__':
    unittest.main()
