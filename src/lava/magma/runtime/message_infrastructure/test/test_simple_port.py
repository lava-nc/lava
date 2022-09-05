from message_infrastructure import ProxySimplePort
import numpy as np


def main():
    port = ProxySimplePort()
    data = np.array([1,2,3], dtype=np.int32)
    print(port.set_data(data))
#    port.transfer()
    ret = port.get_data()
    print(ret)


main()
