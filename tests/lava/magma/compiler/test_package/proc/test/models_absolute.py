"""Process models using absolute imports.

"""

import os
import sys
sys.path.append(os.path.abspath("../../../"))

from lava.magma.core.model.model import AbstractProcessModel
from lava.magma.core.decorator import implements
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from test_package.proc.test.process import TestProcess


@implements(proc=TestProcess, protocol=LoihiProtocol)
class TestModelAbsolute(AbstractProcessModel):
    """Process model defined using absolute import of Process."""

