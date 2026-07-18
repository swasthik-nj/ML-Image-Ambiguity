"""Central logging configuration for console and rotating file handlers."""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Any

from image_ambiguity.config import get_settings


def build_logging_dict(
    log_level: str | None = None,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    """Build a ``dictConfig``-compatible logging configuration.

    Args:
        log_level: Override for the root log level.
        log_dir: Directory for rotating log files.

    Returns:
        Logging configuration dictionary.
    """
    settings = get_settings()
    level = (log_level or settings.log_level).upper()
    directory = log_dir or settings.log_dir
    directory.mkdir(parents=True, exist_ok=True)
    log_file = directory / "app.log"

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": (
                    "%(asctime)s | %(levelname)-8s | %(name)s | "
                    "%(filename)s:%(lineno)d | %(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "simple": {
                "format": "%(levelname)s | %(name)s | %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": level,
                "formatter": "standard",
                "filename": str(log_file),
                "maxBytes": 5_000_000,
                "backupCount": 5,
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "image_ambiguity": {
                "level": level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "uvicorn": {
                "level": level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
        },
        "root": {
            "level": level,
            "handlers": ["console", "file"],
        },
    }


def setup_logging(
    log_level: str | None = None,
    log_dir: Path | None = None,
) -> None:
    """Configure application-wide logging.

    Args:
        log_level: Optional override for log level (e.g. ``DEBUG``).
        log_dir: Optional override for the log directory.
    """
    logging.config.dictConfig(
        build_logging_dict(log_level=log_level, log_dir=log_dir)
    )


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the ``image_ambiguity`` namespace.

    Args:
        name: Optional child logger name. Defaults to ``image_ambiguity``.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    if name is None or name == "image_ambiguity":
        return logging.getLogger("image_ambiguity")
    if name.startswith("image_ambiguity."):
        return logging.getLogger(name)
    return logging.getLogger(f"image_ambiguity.{name}")
