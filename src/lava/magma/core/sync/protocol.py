# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from abc import abstractmethod


class AbstractSyncProtocol:
    """
    Base class for `SyncProtocols`.

    A `SyncProtocol` defines how and when the Processes in a `SyncDomain` are
    synchronized and communication is possible. SyncProtocols need to implement
    the `runtime_service()' method which returns a map between hardware
    resources and the corresponding `RuntimeServices`.

    For example:

    >>> @property
    >>> def runtime_service(self) -> ty.Dict[Resource, AbstractRuntimeService]:
    >>>     return {CPU: LoihiPyRuntimeService,
    >>>             LMT: NxSdkRuntimeService,
    >>>             NeuroCore: NxSdkRuntimeService,
    >>>             Loihi1NeuroCore: NxSdkRuntimeService,
    >>>             Loihi2NeuroCore: NxSdkRuntimeService}

    The phases of execution and synchronizations points are implemented in
    the specific `RuntimeService`.
    """
    @property
    @abstractmethod
    def runtime_service(self):
        raise NotImplementedError()
