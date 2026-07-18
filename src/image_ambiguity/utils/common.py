"""Common helpers used across data, training, and API layers."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

from image_ambiguity.logging_config import get_logger

logger = get_logger("utils")

T = TypeVar("T")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if needed and return it.

    Args:
        path: Target directory path.

    Returns:
        Resolved :class:`~pathlib.Path`.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_global_seed(seed: int) -> None:
    """Seed Python's ``random`` module for reproducible sampling.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    logger.debug("Global random seed set to %s", seed)


def load_json(path: str | Path) -> Any:
    """Load a JSON file.

    Args:
        path: Path to a JSON document.

    Returns:
        Parsed JSON content.
    """
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Any, path: str | Path, *, indent: int = 2) -> Path:
    """Serialize ``data`` as JSON, creating parent directories as needed.

    Args:
        data: JSON-serializable object.
        path: Destination file path.
        indent: Pretty-print indentation level.

    Returns:
        Path written to disk.
    """
    destination = Path(path)
    ensure_dir(destination.parent)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, ensure_ascii=False)
    return destination


@contextmanager
def timed(operation: str) -> Iterator[None]:
    """Context manager that logs elapsed wall-clock time.

    Args:
        operation: Human-readable operation label for the log line.
    """
    start = time.perf_counter()
    logger.info("Started: %s", operation)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("Finished: %s (%.3fs)", operation, elapsed)


def chunked(items: list[T], size: int) -> Iterator[list[T]]:
    """Yield successive chunks from ``items``.

    Args:
        items: Source sequence.
        size: Maximum chunk length (must be >= 1).

    Yields:
        Contiguous sub-lists of ``items``.
    """
    if size < 1:
        raise ValueError("size must be >= 1")
    for index in range(0, len(items), size):
        yield items[index : index + size]


def identity(value: T) -> T:
    """Return ``value`` unchanged (useful as a default transform)."""
    return value


def compose(*functions: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Compose unary callables left-to-right."""

    def _composed(value: Any) -> Any:
        result = value
        for function in functions:
            result = function(result)
        return result

    return _composed
