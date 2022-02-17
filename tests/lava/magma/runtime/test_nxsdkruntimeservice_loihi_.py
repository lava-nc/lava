import logging
import unittest

from tests.lava.test_utils.utils import Utils

from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.nc.type import LavaNcType
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import Loihi2NeuroCore, Loihi1NeuroCore
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import RunConfig
from lava.magma.core.sync.domain import SyncDomain
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.nc.model import NcProcessModel


class SimpleProcess(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(loglevel=logging.WARNING, **kwargs)
        shape = kwargs["shape"]
        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)


class SimpleRunConfig(RunConfig):
    def __init__(self, **kwargs):
        sync_domains = kwargs.pop("sync_domains")
        super().__init__(custom_sync_domains=sync_domains,
                         loglevel=logging.WARNING)
        self.model = None
        if "model" in kwargs:
            self.model = kwargs.pop("model")

    def select(self, process, proc_models):
        if self.model is not None:
            if self.model == "sub" and isinstance(process, SimpleProcess):
                return proc_models[1]

        return proc_models[0]


@implements(proc=SimpleProcess, protocol=LoihiProtocol)
@requires(Loihi1NeuroCore)
class SimpleProcessModel1(NcProcessModel):
    u = LavaNcType(int, int)
    v = LavaNcType(int, int)

    def post_guard(self):
        return False

    def pre_guard(self):
        return False

    def lrn_guard(self):
        return False


@implements(proc=SimpleProcess, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class SimpleProcessModel2(NcProcessModel):
    u = LavaNcType(int, int)
    v = LavaNcType(int, int)

    def post_guard(self):
        return False

    def pre_guard(self):
        return False

    def lrn_guard(self):
        return False


@unittest.skip
class TestProcessLoihi1(unittest.TestCase):
    # Run Loihi Tests using example command below:
    #
    # SLURM=1 LOIHI_GEN=N3B3 BOARD=ncl-og-05 PARTITION=oheogulch
    # RUN_LOIHI_TESTS=1 python -m unittest
    # tests/lava/magma/runtime/test_nxsdkruntimeservice_loihi_.py

    run_loihi_tests: bool = Utils.get_env_test_setting("RUN_LOIHI_TESTS")

    @unittest.skipUnless(run_loihi_tests,
                         "runtime_to_runtimeservice_to_nxcore_to_loihi")
    def test_synchronization_single_process_model(self):
        process = SimpleProcess(shape=(2, 2))
        simple_sync_domain = SyncDomain("simple", LoihiProtocol(), [process])
        run_config = SimpleRunConfig(sync_domains=[simple_sync_domain])
        process.run(condition=RunSteps(num_steps=10), run_cfg=run_config)
        process.run(condition=RunSteps(num_steps=5), run_cfg=run_config)
        process.stop()


class TestProcessLoihi2(unittest.TestCase):
    run_loihi_tests: bool = Utils.get_env_test_setting("RUN_LOIHI_TESTS")

    @unittest.skipUnless(run_loihi_tests,
                         "runtime_to_runtimeservice_to_nxcore_to_loihi")
    def test_synchronization_single_process_model(self):
        process = SimpleProcess(shape=(2, 2))
        simple_sync_domain = SyncDomain("simple", LoihiProtocol(), [process])
        run_config = SimpleRunConfig(sync_domains=[simple_sync_domain])
        process.run(condition=RunSteps(num_steps=10), run_cfg=run_config)
        process.run(condition=RunSteps(num_steps=5), run_cfg=run_config)
        process.stop()


if __name__ == "__main__":
    unittest.main()
