# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import lava.magma.core.learning.string_symbols as str_symbols

# ---------------------------------------------------------------------------
# Width constants (only for fixed-point implementation)
W_EPOCH_TIME = 6

W_WEIGHTS_U = 8
W_WEIGHTS_S = W_WEIGHTS_U + 1
W_TAG_2_U = 7
W_TAG_2_S = W_TAG_2_U + 1
W_TAG_1_U = 8
W_TAG_1_S = W_TAG_1_U + 1

W_SYN_VAR_U = {
    "weights": W_WEIGHTS_U,
    "tag_2": W_TAG_2_U,
    "tag_1": W_TAG_1_U
}

W_SYN_VAR_S = {
    "weights": W_WEIGHTS_S,
    "tag_2": W_TAG_2_S,
    "tag_1": W_TAG_1_S
}

W_TRACE = 7
W_TRACE_FRACTIONAL_PART = 8
W_CONST = 8

W_S_MANT = 4

W_ACCUMULATOR_U = 15
W_ACCUMULATOR_S = W_ACCUMULATOR_U + 1

FACTOR_TO_WIDTH_DICT = {
    str_symbols.X0: 1,
    str_symbols.X1: W_TRACE,
    str_symbols.X2: W_TRACE,
    str_symbols.Y0: 1,
    str_symbols.Y1: W_TRACE,
    str_symbols.Y2: W_TRACE,
    str_symbols.Y3: W_TRACE,
    str_symbols.W: W_WEIGHTS_S,
    str_symbols.D: W_TAG_2_S,
    str_symbols.T: W_TAG_1_S,
    str_symbols.C: W_CONST
}

DEP_TO_IDX_DICT = {
    str_symbols.X0: 0,
    str_symbols.Y0: 1,
    str_symbols.U: 2
}

TRACE_TO_IDX_DICT = {
    str_symbols.X1: 0,
    str_symbols.X2: 1,
    str_symbols.Y1: 0,
    str_symbols.Y2: 1,
    str_symbols.Y3: 2
}
