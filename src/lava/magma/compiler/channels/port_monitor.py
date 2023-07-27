# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/
from contextlib import ContextDecorator
from datetime import datetime
from multiprocessing import Process, Event


class PortMonitor(ContextDecorator):
    """
    Monitors a Port to observe if it is blocked.

    Spins up a process which observes and waits on an event. If the wait
    completes successfully, no msg is printed. If the wait times out,
    a msg gets printed with the current time, port and function name.
    """
    def __init__(self, method_type: str, channel_name: str):
        self._method_type: str = method_type
        self._channel_name: str = channel_name
        # Choosing large enough timeouts. These can be made
        # configurable via configs in future
        if 'runtime_to_service' in self._channel_name or \
            'service_to_process' in self._channel_name or \
            'process_to_service' in self._channel_name or \
            'service_to_runtime' in self._channel_name:
            self._monitor_timeout: float = 600.0  # seconds
        else:
            self._monitor_timeout: float = 60.0  # seconds
        self._event = Event()

    def __enter__(self):
        p = Process(target=PortMonitor.monitor,
                    args=(self._event,
                          self._method_type,
                          self._channel_name,
                          self._monitor_timeout,))
        p.daemon = True
        p.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._event.set()
        return False

    @staticmethod
    def monitor(e: Event,
                method_type: str,
                channel_name: str,
                monitor_timeout: float):
        while True:
            if not e.wait(monitor_timeout):
                now = datetime.now()
                t = datetime.strftime(now, "%H:%M:%S")
                msg = f"{t} : Blocked on {channel_name} :: {method_type}"
                print(msg, flush=True)
            else:
                break
