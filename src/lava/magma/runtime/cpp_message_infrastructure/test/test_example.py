# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from MessageInfrastructurePywrapper import MultiProcessing
import time


def print_hello():
    print("hello")


def main():
    mp = MultiProcessing()
    for i in range(5):
        mp.build_actor(print_hello)

    mp.check_actor()


main()
print("sleep 5")
time.sleep(5)
