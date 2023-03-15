import os
import sys
import numpy as np

from lava.magma.runtime.message_infrastructure. \
    MessageInfrastructurePywrapper import (
        TempChannel
    )


# float equal
def f_eq(a, b):
    return abs(a - b) < 0.001


def soc_names_from_args():
    # default file names
    C2PY = "./c2py"
    PY2C = "./py2c"

    socket_file_names = [C2PY, PY2C]
    filename_args = sys.argv[1:3]
    if len(filename_args) == 1: 
        socket_file_names[0] = filename_args[0]
    if len(filename_args) == 2: 
        socket_file_names = filename_args

    return socket_file_names


def main():
    c2py, py2c = soc_names_from_args()

    if os.path.exists(c2py): 
        os.remove(c2py)
    if os.path.exists(py2c): 
        os.remove(py2c)

    # order matters
    ch2 = TempChannel(c2py)
    rc = ch2.dst_port
    rc.start()

    input("Start the c process, hit enter when you see *receiving*")

    for i in range(10):
        # send port is one-off
        ch = TempChannel(py2c)
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
