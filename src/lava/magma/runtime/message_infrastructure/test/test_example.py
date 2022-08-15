# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from MessageInfrastructurePywrapper import CppMultiProcessing
from MessageInfrastructurePywrapper import SharedMemManager
import time


def main():
    mp = CppMultiProcessing()
    for i in range(5):
        ret = mp.build_actor()
        if ret == 0 :
            print("child process, exit")
            exit(0)

    mp.check_actor()
    mp.stop()

    shm = SharedMemManager()
    for i in range(5):
        print("shared id:", shm.alloc_mem(10))


main()
