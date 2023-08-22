# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

from lava.magma.core.process.message_interface_enum import ActorType
from lava.magma.runtime.message_infrastructure.multiprocessing import \
    MultiProcessing


class MessageInfrastructureFactory:
    """Factory class to create the messaging infrastructure"""

    @staticmethod
    def create(factory_type: ActorType):
        """Creates the message infrastructure instance based on type
        of actor framework being chosen."""
        if factory_type == ActorType.MultiProcessing:
            return MultiProcessing()
        else:
            raise Exception("Unsupported factory_type")
