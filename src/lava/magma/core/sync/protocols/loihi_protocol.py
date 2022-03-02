# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from collections import namedtuple
from dataclasses import dataclass

from lava.magma.core.resources import (
    CPU,
    NeuroCore,
    Loihi1NeuroCore,
    Loihi2NeuroCore
)
from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.magma.runtime.mgmt_token_enums import enum_to_np
from lava.magma.runtime.runtime_services.runtime_service import (
    LoihiPyRuntimeService,
    LoihiCRuntimeService,
    NxSdkRuntimeService
)

Proc_Function_With_Guard = namedtuple("Proc_Function_With_Guard", "guard func")


class Phase:
    SPK = enum_to_np(1)
    PRE_MGMT = enum_to_np(2)
    LRN = enum_to_np(3)
    POST_MGMT = enum_to_np(4)
    HOST = enum_to_np(5)


@dataclass
class LoihiProtocol(AbstractSyncProtocol):
    # The phases of Loihi protocol
    phases = [Phase.SPK, Phase.PRE_MGMT,
              Phase.LRN, Phase.POST_MGMT, Phase.HOST]
    # Methods that processes implementing protocol may provide
    proc_functions = [
        Proc_Function_With_Guard("pre_guard", "run_pre_mgmt"),
        Proc_Function_With_Guard("lrn_guard", "run_lrn"),
        Proc_Function_With_Guard("post_guard", "run_post_mgmt"),
        Proc_Function_With_Guard("host_guard", "run_host_mgmt"),
        Proc_Function_With_Guard(None, "run_spk"),
    ]
    # Synchronizer classes that implement protocol in a domain
    # TODO: Convert this to a dictionary runtime_service = {Node: Class}
    # runtime_service = {CPU: LoihiPyRuntimeService, Loihi1:
    # LoihiCRuntimeService}

    @property
    def runtime_service(self):
        return {CPU: LoihiPyRuntimeService,
                NeuroCore: LoihiCRuntimeService,
                Loihi1NeuroCore: NxSdkRuntimeService,
                Loihi2NeuroCore: NxSdkRuntimeService}
