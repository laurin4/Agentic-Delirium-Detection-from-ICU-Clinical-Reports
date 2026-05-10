"""Tests for USZ/Gemma response JSON extraction."""

import json

import pytest

from src.models.llm_interface import extract_first_json_object


def test_clean_json_unchanged():
    s = '{"a": 1, "b": "x"}'
    assert extract_first_json_object(s) == s


def test_json_followed_by_end_of_turn_and_noise():
    s = '{"klasse": 1}<end_of_turn>\n<start_of_turn>user\nmore junk'
    assert extract_first_json_object(s) == '{"klasse": 1}'


def test_json_followed_by_repeated_prompt_style_text():
    s = (
        '{"signals": []}Return a JSON object with exactly these keys'
        ' and more prompt repetition {"fake": "nested"}'
    )
    out = extract_first_json_object(s)
    assert out == '{"signals": []}'
    assert json.loads(out) == {"signals": []}


def test_braces_inside_json_strings():
    s = '{"msg": "foo{bar}baz", "nested": "}\\\" edge"}'
    out = extract_first_json_object(s)
    assert json.loads(out) == {"msg": "foo{bar}baz", "nested": '}" edge'}


def test_no_json_returns_stripped_original():
    assert extract_first_json_object("  not json at all  ") == "not json at all"


def test_incomplete_object_returns_stripped_original():
    s = '  {"a": '
    assert extract_first_json_object(s) == '{"a":'


@pytest.mark.parametrize(
    "tokens",
    [
        "<start_of_turn>model\n",
        "<start_of_turn>user\n",
        "<start_of_turn>\n",
    ],
)
def test_tokens_stripped_before_extraction(tokens):
    inner = '{"ok": true}'
    s = inner + tokens + "trailing"
    assert extract_first_json_object(s) == inner
