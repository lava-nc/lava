# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

"""
This module will contain a tool to perform automated float- to fixed-point
conversion of of Lava Processes.

Given a set of processes and data, the tool will perform the following steps:

1. Find fixed-point ProcessModels of every floating-point ProcessModel if
available and throw an error otherwise.
2. Run the floating-point model and collect activation statistics.
3. Given activation statistic and type/precision/dynamic range specifications
of fixed-point models, perform input parameter conversion for fixed-point model.
4. Replace floating-point ProcessModels by corresponding fixed-point
ProcessModels
5. Initialize fixed-point ProcessModels

ProcessModel extraction might be a shared function with compiler.

This will likely lead to some performance degradation of the application so
perhaps some iterative optimization may have to be included within this
parameter conversion process. Reuse other libraries for this purpose where
possible.
 """
