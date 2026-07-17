"""Small crash-recoverable process lock for one dated delivery task."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path


class ProcessLock:
    """Create an exclusive lock, recovering only demonstrably stale owners."""

    def __init__(
        self, directory: Path, task: str, key: str, *, lease_seconds: int = 3600
    ) -> None:
        self.path = Path(directory) / f"{task}-{key}.lock"
        self.task, self.key, self.lease_seconds, self.acquired = (
            task,
            key,
            lease_seconds,
            False,
        )

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _stale(self) -> bool:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            pid = int(value["pid"])
            created = datetime.fromisoformat(value["created_at"])
            return not self._pid_alive(pid) or datetime.now(
                timezone.utc
            ) >= created + timedelta(seconds=self.lease_seconds)
        except Exception:
            return False

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "pid": os.getpid(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "task": self.task,
                "key": self.key,
            }
        )
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                self.acquired = True
                return True
            except FileExistsError:
                if self._stale():
                    stale = self.path.with_suffix(".stale")
                    try:
                        os.replace(self.path, stale)
                    except FileNotFoundError:
                        continue
                    continue
                return False
        return False

    def release(self) -> None:
        if self.acquired:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            self.acquired = False

    def __enter__(self) -> "ProcessLock":
        if not self.acquire():
            raise RuntimeError("Delivery task is already running")
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
