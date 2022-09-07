from message_infrastructure import PyDataTransfer
import numpy as np


def main():
    pter = PyDataTransfer()
    data = np.array([1, 2, 3], dtype=np.int32)
    pter.mdata_from_object(data)

    x = pter.get_obj()
    print(x)

main()
