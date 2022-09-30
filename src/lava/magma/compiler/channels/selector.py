# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
from message_infrastructure import RecvPort


class Selector:
    def select(
            self,
            *args: ty.Tuple[RecvPort, ty.Callable[[], ty.Any]],
    ):
        for recv_port, action in args:
            if recv_port.probe():
                return action()
        return None
