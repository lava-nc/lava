"""Test process for testing process model search.

"""

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.model.model import AbstractProcessModel
from lava.magma.core.decorator import implements
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol


class TestProcess(AbstractProcess):
    """Test process for proc-model search"""


@implements(proc=TestProcess, protocol=LoihiProtocol)
class TestModelSameFile(AbstractProcessModel):
    """Test process model in same file as process for proc-model search."""
