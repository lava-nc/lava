# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from collections import namedtuple
from dataclasses import dataclass

from lava.magma.core.resources import (CPU, LMT, Loihi1NeuroCore,
                                       Loihi2NeuroCore, NeuroCore)
from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.magma.runtime.mgmt_token_enums import enum_to_np
from lava.magma.runtime.runtime_services.runtime_service import \
    LoihiPyRuntimeService

try:
    from lava.magma.runtime.runtime_services.nxsdk_runtime_service import \
        NxSdkRuntimeService
except ImportError:
    class NxSdkRuntimeService:
        pass

Proc_Function_With_Guard = namedtuple("Proc_Function_With_Guard", "guard func")


class Phase:
    SPK = enum_to_np(1)
    PRE_MGMT = enum_to_np(2)
    LRN = enum_to_np(3)
    POST_MGMT = enum_to_np(4)
    HOST = enum_to_np(5)


@dataclass
class LoihiProtocol(AbstractSyncProtocol):
    """
    A ProcessModel implementing this synchronization protocol adheres to the
    phases of execution of the neuromorphic processor Loihi.

    Each phase is implemented by a dedicated method. Loihi's phases of
    execution are listed below in order of execution. A ProcessModel adhering
    to this protocol must implement each phase in a dedicated method.
    Additionally, for each phase (except the spiking phase), a "guard" method
    must be implemented that determines whether its respective phase is
    executed in the given time step.

    For example:
    To execute the post management phase ('run_post_mgmt()'), the function
    `post_guard()` must return True.

        >>> def post_guard(self) -> bool:
        >>>     return True

        >>> def run_post_mgmt(self) -> None:
        >>>     # i.e. read out a variable from a neurocore

    *Phases*
    The phases are executed in this order:

    1. Spiking phase:
        Synaptic input is served, neuron states are updated and
        output spikes are generated and delivered. This method is always
        executed and does not have a corresponding guard method.

        Method:
            'run_spk()' -> None

        Guard:
            None

    2. Pre management phase (`run_pre_mgmt`):
        Memory is consolidated before the learning phase.
        In order to jump into this phase, 'pre_guard' and `learn_guard`
        must return True.

        Method:
            'run_pre_mgmt()' -> None

        Guard:
            `learn_guard()` -> None, and `pre_guard()` -> None

    3. Learning phase:
        Activity traces are calculated, learning rules are applied and
        parameters (weights, thresholds, delays, tags, etc) are updated.
        In order to jump into this phase, 'lrn_guard' must return True.

        Method:
            'run_lrn()' -> None

        Guard:
            `learn_guard()` -> None

    4. Post management phase:
        Memory is consolidated after learning phase. Read and write access
        to neuron states are safe now.
        In order to jump into this phase 'post_guard' must return True.

        Method:
            'run_post_mgmt()' -> None

          Guard:
            `learn_guard()` -> None

    5. Host phase:
        Memory of the host system is consolidated.
        In order to jump into this phase 'host_guard' must return True.

        Method:
            'run_host_mgmt()' -> None

        Guard:
            `host_guard()` -> None

    """
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
    @property
    def runtime_service(self):
        """Return RuntimeService."""
        return {CPU: LoihiPyRuntimeService,
                LMT: NxSdkRuntimeService,
                NeuroCore: NxSdkRuntimeService,
                Loihi1NeuroCore: NxSdkRuntimeService,
                Loihi2NeuroCore: NxSdkRuntimeService}
