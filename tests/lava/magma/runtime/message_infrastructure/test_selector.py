import numpy as np
import unittest
from functools import partial
from lava.magma.runtime.message_infrastructure \
    import Selector
from lava.magma.runtime.message_infrastructure import (
    PURE_PYTHON_VERSION,
    Channel)

class Builder:
    def build(self, i):
        pass

def prepare_data():
    arr1 = np.array([1] * 9990)
    arr2 = np.array([1, 2, 3, 4, 5,
                    6, 7, 8, 9, 0])
    return np.concatenate((arr2, arr1))

def bound_target_a1(loop, actor_0_to_mp, actor_1_to_mp,
                    actor_2_to_mp, builder):
    to_mp_0 = actor_0_to_mp.src_port
    to_mp_1 = actor_1_to_mp.src_port
    to_mp_2 = actor_2_to_mp.src_port
    to_mp_0.start()
    to_mp_1.start()
    to_mp_2.start()
    predata = prepare_data()
    while loop > 0:
        loop = loop - 1
        to_mp_0.send(predata)
        to_mp_1.send(predata)
        to_mp_2.send(predata)
    to_mp_0.join()
    to_mp_1.join()
    to_mp_2.join()


class TestSelector(unittest.TestCase):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.loop_ = 1000

    @unittest.skipIf(PURE_PYTHON_VERSION, "cpp msg lib test")
    def test_selector(self):
        from lava.magma.runtime.message_infrastructure \
            .MessageInfrastructurePywrapper import ChannelType
        from lava.magma.runtime.message_infrastructure \
            .multiprocessing \
            import MultiProcessing

        loop = self.loop_*3
        mp = MultiProcessing()
        mp.start()
        predata = prepare_data()
        queue_size = 1
        nbytes = np.prod(predata.shape) * predata.dtype.itemsize
        selector = Selector()
        actor_0_to_mp = Channel(
            ChannelType.SHMEMCHANNEL,
            queue_size,
            nbytes,
            "actor_0_to_mp",
            "actor_0_to_mp",
            (2, 2),
            np.int32)
        actor_1_to_mp = Channel(
            ChannelType.SHMEMCHANNEL,
            queue_size,
            nbytes,
            "actor_1_to_mp",
            "actor_1_to_mp",
            (2, 2),
            np.int32)
        actor_2_to_mp = Channel(
            ChannelType.SHMEMCHANNEL,
            queue_size,
            nbytes,
            "actor_2_to_mp",
            "actor_2_to_mp",
            (2, 2),
            np.int32)

        target_a1 = partial(bound_target_a1, self.loop_, actor_0_to_mp,
                            actor_1_to_mp, actor_2_to_mp)

        builder = Builder()

        mp.build_actor(target_a1, builder)  # actor1

        from_a0 = actor_0_to_mp.dst_port
        from_a1 = actor_1_to_mp.dst_port
        from_a2 = actor_2_to_mp.dst_port

        from_a0.start()
        from_a1.start()
        from_a2.start()
        expect_result = predata*3*self.loop_
        recv_port_list = [from_a0, from_a1,from_a2]
        channel_actions = [(recv_port, (lambda y: (lambda: y))(
                recv_port)) for recv_port in recv_port_list]
        real_result = np.array(0)
        while loop > 0:
            loop = loop - 1
            recv_port = selector.select(*channel_actions)
            data = recv_port.recv()
            real_result = real_result + data
        if not np.array_equal(expect_result, real_result):
            print("expect: ", expect_result)
            print("result: ", real_result)
            raise AssertionError()
        from_a0.join()
        from_a1.join()
        from_a2.join()
        mp.stop()
        mp.cleanup(True)
        
if __name__ == '__main__':
    unittest.main()