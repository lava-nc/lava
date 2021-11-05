# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

"""
This module will contain a tool to automatically validate the correctness of
one ProcessModel of Process against another ProcessModel of the same Process.

For instance, an application might use processes having both PyProcessModels
and NcProcessModels. In order to ensure that the Python SW model of a
NeuroCore produces the same result, this tool can be used to compare both
implementations. In doing so, it will point out exactly at which point in
time in which state variable a divergence in the internal dynamics occurs in
order to ease mismatch resolution.
"""
