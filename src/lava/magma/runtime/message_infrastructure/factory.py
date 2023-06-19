# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/
from lava.magma.runtime.message_infrastructure.message_interface_enum \
    import ActorType
from lava.magma.runtime.message_infrastructure import PURE_PYTHON_VERSION



class MessageInfrastructureFactory:
    """Creates the message infrastructure instance based on type"""

    @staticmethod
    def create(factory_type: ActorType):
        if PURE_PYTHON_VERSION:
            factory_type = ActorType.PyMultiProcessing
        """type of actor framework being chosen"""
        if factory_type == ActorType.MultiProcessing:
            from lava.magma.runtime.message_infrastructure.multiprocessing \
                import MultiProcessing
            return MultiProcessing()
        elif factory_type == ActorType.PyMultiProcessing:
            from lava.magma.runtime.message_infrastructure.py_multiprocessing \
                import MultiProcessing
            return MultiProcessing()
        else:
            raise Exception("Unsupported factory_type")
