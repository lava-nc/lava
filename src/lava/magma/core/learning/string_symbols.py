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
