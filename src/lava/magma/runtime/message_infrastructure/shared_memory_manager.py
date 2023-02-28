# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

import multiprocessing.managers as managers
from multiprocessing.shared_memory import SharedMemory
import typing as ty


class SharedMemoryManager:
    def __init__(self) -> None:
        self._shared_memory_handles: ty.List[SharedMemory] = []
        self._manager: ty.Optional[
            managers.SharedMemoryManager
        ] = managers.SharedMemoryManager()

    def start(self) -> None:
        self._manager.start()

    def shutdown(self) -> None:
        if self._manager is not None:
            for handle in self._shared_memory_handles:
                handle.close()
            self._shared_memory_handles.clear()
            self._manager.shutdown()
            self._manager = None

    def create_shared_memory(self, size: int) -> SharedMemory:
        handle = self._manager.SharedMemory(size)
        self._shared_memory_handles.append(handle)
        return handle
