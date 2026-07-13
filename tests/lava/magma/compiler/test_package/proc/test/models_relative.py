"""Process models using absolute imports.

"""

from lava.magma.core.model.model import AbstractProcessModel
from lava.magma.core.decorator import implements
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from .process import TestProcess


@implements(proc=TestProcess, protocol=LoihiProtocol)
class TestModelRelative(AbstractProcessModel):
    """Process model defined using relative import of Process."""

