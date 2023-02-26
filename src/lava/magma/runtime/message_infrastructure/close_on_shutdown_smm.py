# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/


from multiprocessing.managers import SharedMemoryManager


class CloseOnShutdownSMM:
    def __init__(self):
        self._shms = []
        self._smm = SharedMemoryManager()

    def start(self):
        self._smm.start()

    def shutdown(self):
        if self._smm is not None:
            for shm in self._shms:
                shm.close()
            self._shms.clear()
            self._smm.shutdown()
            self._smm = None

    def shared_memory(self, size: int):
        shm = self._smm.SharedMemory(size)
        self._shms.append(shm)
        return shm
