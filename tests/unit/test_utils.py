"""Unit tests for shared utilities."""

from __future__ import annotations

from pathlib import Path

from image_ambiguity.logging_config import setup_logging
from image_ambiguity.utils import ensure_dir, load_json, save_json, set_global_seed, timed
from image_ambiguity.utils.common import chunked


def test_ensure_dir_creates_nested_path(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c"
    created = ensure_dir(target)
    assert created.is_dir()


def test_save_and_load_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "out" / "sample.json"
    payload = {"image_id": 1, "captions": ["hello"]}

    save_json(payload, path)
    loaded = load_json(path)

    assert loaded == payload


def test_set_global_seed_is_reproducible() -> None:
    import random

    set_global_seed(42)
    first = [random.random() for _ in range(3)]
    set_global_seed(42)
    second = [random.random() for _ in range(3)]
    assert first == second


def test_chunked_splits_list() -> None:
    assert list(chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_timed_context_logs(tmp_path: Path) -> None:
    setup_logging(log_level="INFO", log_dir=tmp_path)
    with timed("unit-test-op"):
        pass
