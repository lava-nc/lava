# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import traceback
import unittest
import time
import numpy as np
from functools import partial

from lava.magma.runtime.message_infrastructure import \
    PURE_PYTHON_VERSION


def nbytes_cal(shape, dtype):
    return np.prod(shape) * np.dtype(dtype).itemsize


class Builder:
    def build(self, i):  # pylint: disable=no-self-use
        time.sleep(0.0001)


def target_fn(*args, **kwargs):
    """
    Function to build and attach a system process to

    :param args: List Parameters to be passed onto the process
    :param kwargs: Dict Parameters to be passed onto the process
    :return: None
    """
    try:
        builder = kwargs.pop("builder")
        idx = kwargs.pop("idx")
        builder.build(idx)
        return 0
    except Exception as e:
        print("Encountered Fatal Exception: " + str(e))
        print("Traceback: ")
        print(traceback.format_exc())
        raise e


class TestMultiprocessing(unittest.TestCase):

    @unittest.skipIf(PURE_PYTHON_VERSION, "cpp msg lib version")
    def test_multiprocessing_actors(self):  # pylint: disable=no-self-use
        from lava.magma.runtime.message_infrastructure.multiprocessing \
            import MultiProcessing
        mp = MultiProcessing()
        mp.start()
        builder = Builder()
        for i in range(5):
            bound_target_fn = partial(target_fn, idx=i)
            mp.build_actor(bound_target_fn, builder)

        time.sleep(0.1)
        mp.stop()
        mp.cleanup(True)


if __name__ == '__main__':
    unittest.main()
