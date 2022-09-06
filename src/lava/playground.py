import numpy as np
from lava.magma.core.learning.utils import saturate
from lava.magma.core.learning.constants import *


def add_impulse(trace_values, rth, impulses_int, impulses_frac):
    trace_new = trace_values + impulses_int
    trace_new = np.where(rth < impulses_frac, trace_new + 1, trace_new)
    trace_new = saturate(0, trace_new, 2**BITS_LOW - 1)

    return trace_new


def stochastic_round(var, random):
    exp_mant = 2**BITS_LOW
    var_w = var / exp_mant
    var_f = var_w % 1
    var_w = np.floor(var_w)
    var_w += (var_f > random / 2 ** (BITS_HIGH - 1)).astype(int)
    result = (var_w * exp_mant).astype(var.dtype)

    return result


def decay_trace(trace_values, t, taus, rth):
    result = np.exp(-t / taus) * trace_values
    decimals = result % 1
    rand = rth / (2**BITS_LOW - 1)
    result = np.floor(result)
    result += (rand < decimals).astype(int)

    return result


def new_func(random, value):
    return