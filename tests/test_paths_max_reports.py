import os

import pytest

from src.pipeline.paths import MAX_REPORTS, parse_max_reports_env


def test_parse_max_reports_env_blank_defaults_to_thirty():
    assert parse_max_reports_env("") == 30
    assert parse_max_reports_env("   ") == 30


def test_parse_max_reports_env_all_means_unlimited():
    assert parse_max_reports_env("all") is None
    assert parse_max_reports_env("ALL") is None


def test_parse_max_reports_env_positive():
    assert parse_max_reports_env("30") == 30


def test_parse_max_reports_env_invalid():
    with pytest.raises(ValueError):
        parse_max_reports_env("x")
    with pytest.raises(ValueError):
        parse_max_reports_env("0")
    with pytest.raises(ValueError):
        parse_max_reports_env("-2")


def test_max_reports_module_constant_is_int_or_none(monkeypatch):
    monkeypatch.delenv("MAX_REPORTS", raising=False)
    assert isinstance(MAX_REPORTS, int) or MAX_REPORTS is None
