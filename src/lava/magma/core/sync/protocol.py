# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from abc import abstractmethod


class AbstractSyncProtocol:
    @property
    @abstractmethod
    def runtime_service(self):
        pass
