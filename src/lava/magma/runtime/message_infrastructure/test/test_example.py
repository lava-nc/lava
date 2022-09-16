# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import traceback
from functools import partial

from MessageInfrastructurePywrapper import CppMultiProcessing
from MessageInfrastructurePywrapper import ProcessType
from MessageInfrastructurePywrapper import Actor


class Builder():
    def build(self):
        print("Builder run build")

    def start(self, *args, **kwargs):
        print("Builder run start")


def target_fn(*args, **kwargs):
    """
    Function to build and attach a system process to

    :param args: List Parameters to be passed onto the process
    :param kwargs: Dict Parameters to be passed onto the process
    :return: None
    """
    try:
        builder = kwargs.pop("builder")
        actor = builder.build()
    except Exception as e:
        print("Encountered Fatal Exception: " + str(e))
        print("Traceback: ")
        print(traceback.format_exc())
        raise e


def main():
    builder = Builder()
    mp = CppMultiProcessing()
    bound_target_fn = partial(target_fn, builder=builder)
    bound_target_fn()
    for i in range(5):
        ret = mp.build_actor(bound_target_fn)
        if ret == ProcessType.ChildProcess :
            print("child process, exit")
            exit(0)

    mp.check_actor()
    mp.stop()

    actors = mp.get_actors()
    print(actors)
    print("actor status: ", actors[0].get_status())

main()
