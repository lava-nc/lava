# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from abc import abstractmethod
import typing as ty


class AbstractSyncProtocol:
    @property
    @abstractmethod
    def synchronizer(self) -> ty.Dict[ty.Type, ty.Type]:
        """Synchronizer classes that implement protocol in a domain"""
        pass
