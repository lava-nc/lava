import numpy as np


class MockServicePort:
    phases: int
    phase: int = 0

    def __init__(self, phases: int = 2):
        self.phases = phases

    def probe(self) -> int:
        return 1

    def recv(self) -> int:
        self.phase = (self.phase + 1) % self.phases
        return self.phase

    def peek(self) -> int:
        return 1

    def send(self, msg) -> int:
        return 1


class MockNpServicePort(MockServicePort):
    def recv(self) -> int:
        return np.array([super().recv()])


class MockDataPort:
    sent = None
    recd = None

    def peek(self):
        print("peek")
        return 1

    def probe(self):
        print("probe")
        return 1

    def recv(self):
        # data = np.ones(1, dtype=np.int64)
        data = np.random.randint(0, 10, size=(1,), dtype=np.int64)
        print(f"recv: {data}")
        self.recd = data
        return data

    def send(self, data):
        print(f"send: {data}")
        self.sent = data
        return data.size

    def flush(self):
        print("flush")
        pass
