import numpy as np
from lava.magma.core.learning.utils import saturate
from lava.magma.core.learning.constants import *

num_1 = 2 ** (BITS_HIGH - 1) - 1
print(num_1)
print(np.binary_repr(num_1, 18))

num_2 = -(2 ** (BITS_HIGH - 1)) - 1
print(num_2)
print(np.binary_repr(num_2, 18))

print("hello", 15 - W_WEIGHTS_U)

num = 2 ** (W_ACCUMULATOR_U - W_WEIGHTS_U) - 1
print(num)
print(np.binary_repr(num, 18))
# mask = ~(~0 << W_MASKS_DICT["weights"])
# print(mask)
# print(np.binary_repr(mask, 16))
