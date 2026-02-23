import csv
import os
from typing import Optional

from src.dataset import Dataset


class LocalDataset(Dataset):
    """Streaming CSV dataset (no full load into memory)."""

    def __init__(
        self,
        path: str,
        cache_dir: str,
        absolute_path: bool = False,
        limit: Optional[int] = None,
    ):
        super().__init__(path if absolute_path else os.path.join(cache_dir, path))

        self._limit = limit
        self._file = None
        self._reader = None
        self._idx = 0

    def _open(self):
        if self._reader is not None:
            return

        self._file = open(self.address(), "r", newline="", encoding="utf-8")
        self._reader = csv.DictReader(self._file)
        self._idx = 0

    def next(self):
        self._open()

        if self._limit is not None and self._idx >= self._limit:
            raise StopIteration

        try:
            row = next(self._reader)
        except StopIteration:
            raise

        self._idx += 1
        return row

    def reset(self):
        if self._file:
            self._file.close()

        self._file = None
        self._reader = None
        self._idx = 0

    def count(self):
        # avoid full load: count lazily
        self.reset()
        self._open()

        count = 0
        for _ in self._reader:
            if self._limit is not None and count >= self._limit:
                break
            count += 1

        self.reset()
        return count
