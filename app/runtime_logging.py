"""Private bounded logs with conservative redaction."""

from __future__ import annotations

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path


class RedactingFilter(logging.Filter):
    """Avoid persisting credential-shaped text in operational logs."""

    pattern = re.compile(r"(Bearer\s+|context_token[=:]\s*|token[=:]\s*)[^\s,]+", re.I)

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self.pattern.sub(r"\1[redacted]", str(record.msg))
        record.args = ()
        return True


def configure_rotating_log(
    name: str, directory: Path, *, max_bytes: int = 1_048_576, backups: int = 3
) -> logging.Logger:
    """Return a logger writing to a private, bounded file."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    handler = RotatingFileHandler(
        target, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
    )
    handler.addFilter(RedactingFilter())
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger = logging.getLogger(f"runtime.{name}")
    logger.handlers[:] = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
