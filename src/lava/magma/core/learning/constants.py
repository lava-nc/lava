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

BITS_LOW = 7
BITS_MID = 9
BITS_HIGH = 16

W_MASKS_DICT = {
    "weights": BITS_MID,
    "tag_2": BITS_MID,
    "tag_1": BITS_MID
}

W_SYN_VAR_DICT = {
    "weights": BITS_MID,
    "tag_2": BITS_MID,
    "tag_1": BITS_MID
}

FACTOR_TO_WIDTH_DICT = {
    str_symbols.X0: 1,
    str_symbols.X1: BITS_LOW,
    str_symbols.X2: BITS_LOW,
    str_symbols.Y0: 1,
    str_symbols.Y1: BITS_LOW,
    str_symbols.Y2: BITS_LOW,
    str_symbols.Y3: BITS_LOW,
    str_symbols.W: BITS_MID,
    str_symbols.D: BITS_MID,
    str_symbols.T: BITS_MID,
    str_symbols.C: BITS_MID
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
