from message_infrastructure import PyDataTransfer
import numpy as np


def main():
    pter = PyDataTransfer()
    data = np.array([1, 2, 3], dtype=np.int32)
    pter.set_obj(data)
    pter.change_var()

    x = pter.get_obj()
    print(x)
    print(data)

main()
