# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

from multiprocessing.managers import (SharedMemoryManager as
                                      DelegateManager
                                      )
from multiprocessing.shared_memory import SharedMemory
import typing as ty


class SharedMemoryManager:
    def __init__(self):
        self._shared_memory_handles: ty.List[SharedMemory] = []
        self._inner_shared_memory_manager: ty.Optional[DelegateManager] = (
            DelegateManager())

    def start(self) -> None:
        self._inner_shared_memory_manager.start()

    def shutdown(self):
        if self._inner_shared_memory_manager is not None:
            for shm in self._shared_memory_handles:
                shm.close()
            self._shared_memory_handles.clear()
            self._inner_shared_memory_manager.shutdown()
            self._inner_shared_memory_manager = None

    def create_shared_memory(self, size: int) -> SharedMemory:
        shm = self._inner_shared_memory_manager.SharedMemory(size)
        self._shared_memory_handles.append(shm)
        return shm
