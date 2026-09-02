"""Shared pytest fixtures for the LiteLLM config test suite."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from cleanup_base import configure_disk_cache


@pytest.fixture(autouse=True)
def _disable_disk_cache(tmp_path):
    """
    Keep tests hermetic: never read from or write to the real on-disk HTTP
    cache. Tests that exercise the disk cache re-enable it explicitly with a
    per-test ``tmp_path`` directory.
    """
    configure_disk_cache(enabled=False, directory=tmp_path / "http-cache")
    yield
    configure_disk_cache(enabled=False)
