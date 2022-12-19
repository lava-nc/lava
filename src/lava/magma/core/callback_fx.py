# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from abc import ABC, abstractmethod
try:
    from nxcore.arch.base.nxboard import NxBoard
except ImportError:
    class NxBoard:
        pass


class CallbackFx(ABC):
    """Base class for callback functions which are executed before
    and after a run in the runtime service. The base class provides
    the infrastructure to communicate information from runtime to
    runtime service and vice versa as well as the abstract pre- and
    post-callback methods, which needs to be overrwritten by the
    special CallbackFx classes for each compute ressource.

    TODO: implement runtime <-> runtime_service channel communication.
    """

    def get_data_from_runtime(self) -> np.ndarray:
        pass

    def send_data_to_runtime(self, data):
        pass

    def get_data_from_runtime_service(self) -> np.ndarray:
        pass

    def send_data_to_runtime_service(self, data):
        pass


class NxSdkCallbackFx(CallbackFx):
    """Abstract class for callback functions processed in the
    NxSdkRuntimeSercice pre- and post run.

    TODO: implement runtime <-> runtime_service channel communication.
    """

    @abstractmethod
    def pre_run_callback(self,
                         board: NxBoard = None,
                         var_id_to_var_model_map: dict = None):
        pass

    @abstractmethod
    def post_run_callback(self,
                          board: NxBoard = None,
                          var_id_to_var_model_map: dict = None):
        pass
