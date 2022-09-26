# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

X0 = "x0"
Y0 = "y0"
SPIKE_DEPENDENCIES = {X0, Y0}

U = "u"
DEPENDENCIES = SPIKE_DEPENDENCIES.union({U})

X1 = "x1"
X2 = "x2"
PRE_TRACES = {X1, X2}

Y1 = "y1"
Y2 = "y2"
Y3 = "y3"
POST_TRACES = {Y1, Y2, Y3}

TRACES = PRE_TRACES.union(POST_TRACES)

W = "w"
D = "d"
T = "t"
SYNAPTIC_VARIABLES = {W, D, T}

C = "C"

FACTOR_STATE_VARS = SPIKE_DEPENDENCIES.union(TRACES)\
    .union(SYNAPTIC_VARIABLES).\
    union({C})

DW = "dw"
DD = "dd"
DT = "dt"

LEARNING_RULE_TARGETS = {DW, DD, DT}

SYNAPTIC_VARIABLE_VAR_MAPPING = {
    W: "weights",
    D: "tag_2",
    T: "tag_1"
}
