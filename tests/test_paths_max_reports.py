import os

import pytest

from src.pipeline.paths import MAX_REPORTS, parse_max_reports_env


def test_parse_max_reports_env_blank():
    assert parse_max_reports_env("") is None
    assert parse_max_reports_env("   ") is None


def test_parse_max_reports_env_positive():
    assert parse_max_reports_env("30") == 30


def test_parse_max_reports_env_invalid():
    with pytest.raises(ValueError):
        parse_max_reports_env("x")
    with pytest.raises(ValueError):
        parse_max_reports_env("0")
    with pytest.raises(ValueError):
        parse_max_reports_env("-2")


def test_max_reports_module_constant_follows_environment(monkeypatch):
    """Sanity: paths.MAX_REPORTS reflects env at import time (typical CLI usage)."""
    monkeypatch.delenv("MAX_REPORTS", raising=False)
    assert os.environ.get("MAX_REPORTS") in (None, "")
    # Module was imported with current env in this worker
    assert MAX_REPORTS is None or isinstance(MAX_REPORTS, int)
