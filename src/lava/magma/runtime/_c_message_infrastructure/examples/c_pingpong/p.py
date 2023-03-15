import os
import numpy as np

from lava.magma.runtime.message_infrastructure. \
    MessageInfrastructurePywrapper import (
        TempChannel
    )

C2PY = "./c2py"
PY2C = "./py2c"

# float equal
def f_eq(a, b):
    return abs(a - b) < 0.001


def main():
    """
    Limitations of current TempChennel:

    TempChannel uses socket file.

    The Recv port will bind the socket file in initialization,
    listen in start() and accept in recv(). After established a connection,
    the port closes it immediately after reading from the socket.

    The Send port will connect to the recv port in initialization and
    write to the socket in send().

    Therefore,
    1. The send port can only be initialized after corresponding
        Recv port called start()
    2. In each round, the send port is used one-off. One needs to
        create a new TempChannel() and get the send port from it
        each time. (The send port will be initialized when accessing
        the .dst_port property at the first time)

    One needs to run the python script first, start the ./cprocess binary
    once seeing the prompt and hit the enter as indicated.
    """
    if os.path.exists(C2PY):
        os.remove(C2PY)
    if os.path.exists(PY2C):
        os.remove(PY2C)

    # order matters
    ch2 = TempChannel(C2PY)
    rc = ch2.dst_port
    rc.start()

    input("Start the c process, hit enter when you see *receiving*")

    for i in range(10):
        # send port is one-off
        ch = TempChannel(PY2C)
        sd = ch.src_port
        sd.start()

        print("round ", i)
        rands = np.array([np.random.random() * 100 for __ in range(10)])  # noqa
        print("Sending array to C: ", rands)
        sd.send(rands)

        rands2 = rc.recv()
        print("Got array from C: ", rands2)

        print("Correctness: ", all([f_eq(x, y) for x, y in zip(rands, rands2)]))  # noqa
        print("========================================")
        sd.join()
    rc.join()


if __name__ == "__main__":
    main()
