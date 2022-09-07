# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
#
# Copyright © 2021-2022 Intel Corporation.
#
# This software and the related documents are Intel copyrighted
# materials, and your use of them is governed by the express
# license under which they were provided to you (License). Unless
# the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit  this software or the
# related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with
# no express or implied warranties, other than those that are
# expressly stated in the License.

import lava.magma.core.learning.string_symbols as str_symbols

# ---------------------------------------------------------------------------
# Width constants (only for fixed-point implementation)

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
