"""Append-only JSONL caches shared by the LLM-assisted stages.

Closure adjudication (ticket 7) and intent discovery (ticket 10) both call a
language model once per case and record every verdict as it completes, so a
slow or interrupted run resumes without paying twice. The crash-safety
contract is identical for both caches:

* appends are atomic with respect to worker threads;
* a writer killed mid-record can leave a partial final line, which readers
  drop so the verdicts before it still resume the run;
* a complete final record whose terminating newline was lost is kept, and the
  file is terminated before the next append so records never merge.

This module owns that mechanism once so the edge cases are fixed in one place.
Stages supply a record parser, a serializer, their own error type, and the id
that keys an entry; entries are stored and returned as the stage's own
dataclasses.
"""

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Generic, TypeVar

T = TypeVar("T")


class CacheFileError(Exception):
    """Raised when a cache file cannot be parsed or appended."""


def read_cache(
    input_path: Path | str,
    parse: Callable[[str, str], T],
    key: Callable[[T], int],
    error_type: type[Exception],
) -> dict[int, T]:
    """Read a cache file; a missing file is an empty cache.

    The cache is an append-only log, so the last record for an id wins (a
    recomputed verdict supersedes a stale one). Every writer terminates a
    record with a newline, so a malformed final line without one can only be a
    write cut short by an interruption or a full disk: it is dropped so earlier
    verdicts still resume the run. Any other malformed record fails the run
    rather than silently dropping verdicts.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        return {}
    data = input_path.read_text(encoding="utf-8")
    complete = data.endswith("\n")
    lines = data.splitlines()
    entries: dict[int, T] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        location = f"line {line_number} of {input_path}"
        try:
            entry = parse(line, location)
        except error_type:
            if line_number == len(lines) and not complete:
                break
            raise
        entries[key(entry)] = entry
    return entries


def repair_tail(
    path: Path | str,
    parse: Callable[[str, str], T],
    error_type: type[Exception],
) -> None:
    """Make a trailing line safe to append after.

    A killed writer can leave the file without its final newline. When the
    trailing bytes are already a complete record, keep them and terminate the
    line so the next record starts fresh; only a genuinely partial or
    malformed tail is dropped.
    """
    path = Path(path)
    if not path.is_file():
        return
    with path.open("r+b") as handle:
        data = handle.read()
        if not data or data.endswith(b"\n"):
            return
        tail_start = data.rfind(b"\n") + 1
        try:
            tail = data[tail_start:].decode("utf-8")
            parse(tail, str(path))
        except (UnicodeDecodeError, error_type):
            handle.truncate(tail_start)
        else:
            handle.write(b"\n")


class AppendOnlyCache(Generic[T]):
    """Append-only JSONL cache, safe under worker threads.

    Callers load the entries once before a run and let every completed verdict
    be recorded immediately, so a run killed mid-way resumes without paying
    for calls it already made.
    """

    def __init__(
        self,
        path: Path | str,
        serialize: Callable[[T], object],
        parse: Callable[[str, str], T],
        key: Callable[[T], int],
        error_type: type[Exception],
    ):
        self.path = Path(path)
        self._serialize = serialize
        self._parse = parse
        self._key = key
        self._error_type = error_type
        self._lock = threading.Lock()

    def entries(self) -> dict[int, T]:
        """Read the cached entries; a missing cache file is an empty cache."""
        return read_cache(self.path, self._parse, self._key, self._error_type)

    def record(self, entry: T) -> None:
        """Append one entry, atomically with respect to worker threads."""
        line = json.dumps(self._serialize(entry)) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            repair_tail(self.path, self._parse, self._error_type)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
