"""Model-neutral sampler and transactional run ownership for bounded pilots."""

import json
import os
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class SamplerCursor:
    size: int
    seed: int
    epoch: int = 0
    position: int = 0
    order: list[int] = field(default_factory=list)

    def __post_init__(self):
        if self.size < 1:
            raise ValueError("empty train selection")
        if not self.order:
            self._shuffle()
        if sorted(self.order) != list(range(self.size)) or not 0 <= self.position <= self.size:
            raise ValueError("invalid sampler checkpoint")

    def _shuffle(self):
        self.order = list(range(self.size))
        random.Random(self.seed + self.epoch).shuffle(self.order)

    def next(self, batch_size):
        if batch_size < 1:
            raise ValueError("positive batch required")
        if self.position == self.size:
            self.epoch += 1
            self.position = 0
            self._shuffle()
        end = min(self.position + batch_size, self.size)
        result = self.order[self.position : end]
        self.position = end
        return result  # drop_last=False; never merge the tail with a new epoch

    def state(self):
        return asdict(self)


@contextmanager
def condition_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "running.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            "condition already owned; inspect stale lock before manual recovery"
        ) from exc
    with os.fdopen(descriptor, "w") as handle:
        json.dump({"pid": os.getpid()}, handle)
    try:
        yield
    finally:
        lock.unlink()


def check_completed(path, identity):
    if not Path(path).exists():
        return False
    record = json.loads(Path(path).read_text())
    if record.get("identity") != identity:
        raise ValueError("condition identity differs; cannot skip or resume")
    return record.get("status") == "completed"
