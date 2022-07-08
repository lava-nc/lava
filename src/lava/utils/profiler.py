# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

"""
This module will contain a tool to determine power and performance of workloads
for Loihi 1 or Loihi 2 based on software simulations or hardware measurements.

The execution time and energy of a workload will be either measured on hardware
during execution or estimated in simulation. The estimation is based on
elementary hardware operations which are counted during the simulation. Each
elementary operation has a defined execution time and energy cost, which is
used in a performance model to calculate execution time and energy.
"""
