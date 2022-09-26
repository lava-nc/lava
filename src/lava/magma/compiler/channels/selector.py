# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
from message_infrastructure import SendPort, RecvPort


class Selector:
    def select(
            self,
            *args: ty.Tuple[
                ty.Union[SendPort, RecvPort], ty.Callable[[], ty.Any]
            ],
    ):
        for channel, action in args:
            if channel.probe():
                return action()
        return None
