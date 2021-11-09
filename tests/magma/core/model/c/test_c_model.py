from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration,get_info

from importlib import import_module,invalidate_caches

import numpy as np
from typing import List,Callable

from lava.magma.core.model.model import AbstractProcessModel
from lava.magma.core.model.py.model import AbstractPyProcessModel

import unittest
import typing as ty

import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.decorator import implements, requires
from lava.magma.core.resources import CPU
from lava.magma.core.model.py.model import AbstractPyProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort

from lava.magma.compiler.utils import VarInitializer, PortInitializer
from lava.magma.compiler.builder import PyProcessBuilder
from lava.magma.compiler.channels.interfaces import AbstractCspPort

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import RunConfig
from lava.proc.dense.process import Dense
from lava.proc.lif.process import LIF



def configuration(parent_package='', top_path=None):
    config = Configuration('',parent_package,top_path)
    config.add_extension('custom',['custom.c'],extra_info=get_info('npymath'))
    return config

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
setup(configuration=configuration, script_args=['build_ext','--inplace'])
invalidate_caches()
module = import_module("custom")
#from custom import Custom

class Proc(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.in_port = InPort((1, 1))
        self.out_port = OutPort((1, 1))

@implements(proc=Proc)
@requires(CPU)
class ProcModel(AbstractPyProcessModel):
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int, 8)
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int, 8)

class SimpleRunConfig(RunConfig):
    def __init__(self, **kwargs):
        sync_domains = kwargs.pop("sync_domains")
        super().__init__(custom_sync_domains=sync_domains)
        self.model = None
        if "model" in kwargs:
            self.model = kwargs.pop("model")

    def select(self, process, proc_models):
        if self.model is not None:
            if self.model == "sub" and isinstance(process, AbstractProcess):
                return proc_models[1]
        return proc_models[0]


class TestCProcessModel(unittest.TestCase):
    def test_callback(self):
        class A:
            x:int=0
            def recv(self):
                self.x+=1
                return x
            def send(self,x:int):
                self.x+=1
                
        a = A()
        c = module.Custom(a)
        c.run()
        self.assertEqual(a.x,2)
        
    '''
    def test_full(self):
        p1=Proc()
        p2=Proc()
        p1.out_port.connect(p2.in_port)
        p1.run(condition=RunSteps(num_steps=10),run_cfg=SimpleRunConfig(sync_domains=[]))
        p1.stop()
    '''

if __name__=="__main__":
    unittest.main()

