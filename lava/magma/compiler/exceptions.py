# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


class NoProcessModelFound(Exception):
    def __init__(self, proc):
        msg = f"No ProcessModels found that implement Process '{proc}'"
        super().__init__(msg)


class ProcessAlreadyCompiled(Exception):
    def __init__(self, proc):
        msg = f"Process '{proc.name}::{proc.__class__.__name__}' has been " \
              f"compiled already. Processes can't be compiled more than once."
        super().__init__(msg)
