# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/
import typing as ty
from contextlib import ContextDecorator
from datetime import datetime
from multiprocessing import Process, Event, Queue, Manager
from abc import abstractmethod
from dataclasses import dataclass
import atexit
from threading import Thread

from lava.magma.compiler.builders.interfaces import AbstractBuilder


@dataclass
class EventMetadata:
    """Data class to store an event to be monitored and associated string
    (channel_name.method_type) to display in case it is timing out."""
    event: Event
    channel_name: str
    method_type: str


class EventCompletionMonitor:
    """A Generic Monitor class which watches a queue. For every entry put
    into the queue, it spins up an event monitor thread to watch its
    completion or print msg if timeout happens."""
    @staticmethod
    def monitor(queue: Queue, timeout: float):
        threads: ty.List[Thread] = []
        while True:
            v: EventMetadata = queue.get()
            if v is None:
                break  # Stopping Criterion

            t = Thread(target=EventCompletionMonitor.event_monitor,
                       args=(v, timeout,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    @staticmethod
    def event_monitor(v: EventMetadata, timeout: float):
        while True:
            if not v.event.wait(timeout):
                now = datetime.now()
                t = datetime.strftime(now, "%H:%M:%S")
                msg = f"{t} : Blocked on {v.channel_name} :: {v.method_type}"
                print(msg, flush=True)
            else:
                break


@dataclass
class WatchdogToken:
    """A Token to encapsulate related entries"""
    event: Event
    queue: Queue
    channel_name: str
    method_type: str


class Watchdog(ContextDecorator):
    """
    Monitors a Port to observe if it is blocked.

    Writes the EventMetadata to a queue which is observed by a
    EventCompletionMonitor process. If the wait completes successfully,
    no msg is printed. If the wait times out, a msg gets printed with the
    current time, channel/port and function name.
    """
    def __init__(self, w: ty.Optional[WatchdogToken]):
        self._w = w

    def __enter__(self):
        w: WatchdogToken = self._w
        w.event.clear()
        v = EventMetadata(w.event, w.channel_name, w.method_type)
        w.queue.put(v)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._w.event.set()
        return False


class WatchdogManagerInterface:
    """Generic Interface for Watchdogs"""
    @abstractmethod
    def lq(self) -> ty.Optional[Queue]:
        pass

    @abstractmethod
    def sq(self) -> ty.Optional[Queue]:
        pass

    @abstractmethod
    def create_watchdog(self,
                        queue: Queue,
                        channel_name: str,
                        method_type: str
                        ) -> Watchdog:
        pass

    def start(self):
        pass

    def stop(self):
        pass


class WatchdogManager(WatchdogManagerInterface):
    """A Wrapper around Multiprocessing Manager which allocates the
    multiprocessing queues, events and event monitors"""
    def __init__(self, long_event_timeout: float, short_event_timeout: float):
        self._mp_manager = None
        self._lq = None
        self._sq = None
        self._long_event_timeout = long_event_timeout
        self._short_event_timeout = short_event_timeout
        self._lm = None
        self._sm = None
        atexit.register(self.stop)

    @property
    def lq(self) -> ty.Optional[Queue]:
        return self._lq

    @property
    def sq(self) -> ty.Optional[Queue]:
        return self._sq

    def create_watchdog(self,
                        queue: Queue,
                        channel_name: str,
                        method_type: str
                        ) -> Watchdog:
        w = WatchdogToken(event=self._mp_manager.Event(),
                          queue=queue,
                          channel_name=channel_name,
                          method_type=method_type)
        return Watchdog(w)

    def start(self):
        self._mp_manager = Manager()

        self._lq = self._mp_manager.Queue()
        self._sq = self._mp_manager.Queue()

        self._lm = Process(target=EventCompletionMonitor.monitor,
                           args=(self._lq, self._long_event_timeout))
        self._lm.start()

        self._sm = Process(target=EventCompletionMonitor.monitor,
                           args=(self._sq, self._short_event_timeout))
        self._sm.start()

    def stop(self):
        if self._mp_manager:
            self._lq.put(None)  # This signals long event_monitor to stop
            self._sq.put(None)  # This signals short event_monitor to stop

            self._lm.join()
            self._sm.join()

            self._mp_manager.shutdown()
            self._mp_manager = None


class NoOPWatchdog(Watchdog):
    """Dummy Watchdog for NoOP"""
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class NoOPWatchdogManager(WatchdogManagerInterface):
    """Dummy Watchdog Manager for NoOP"""
    @property
    def lq(self) -> ty.Optional[Queue]:
        return None

    @property
    def sq(self) -> ty.Optional[Queue]:
        return None

    def create_watchdog(self,
                        queue: Queue,
                        channel_name: str,
                        method_type: str) -> Watchdog:
        return NoOPWatchdog(None)

    def start(self):
        pass

    def stop(self):
        pass


class WatchdogManagerBuilder(AbstractBuilder):
    """Builds a Watchdog Manager given timeout values and debug level"""
    def __init__(self,
                 compile_config: ty.Dict[str, ty.Any],
                 log_level: int):
        self._long_event_timeout = compile_config["long_event_timeout"]
        self._short_event_timeout = compile_config["short_event_timeout"]
        self._debug = compile_config["use_watchdog"]
        if self._debug:
            print("!!!!!!! Using Watchdog to Monitor Ports !!!!!!!")
            print("!!!!!!! Impacts Latency Sensitive Applications !!!!!!!")

    def build(self):
        if self._debug:
            return WatchdogManager(self._long_event_timeout,
                                   self._short_event_timeout)
        else:
            return NoOPWatchdogManager()
