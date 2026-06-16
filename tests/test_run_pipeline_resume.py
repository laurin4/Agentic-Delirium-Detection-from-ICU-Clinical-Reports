"""Checkpoint resume for full-corpus run_pipeline."""

from src.pipeline.run_pipeline import (
    _filter_unprocessed_reports,
    _resume_key_from_mapping,
    _resume_keys_from_rows,
)
from src.preprocessing.report_identity import SOURCE_REPORT_ROW_ID_COL


def test_resume_key_prefers_source_report_row_id():
    assert _resume_key_from_mapping(
        {SOURCE_REPORT_ROW_ID_COL: "berichte_row_7", "PatientenID": "p1", "bericht": "a"}
    ) == ("sid", "berichte_row_7")


def test_filter_unprocessed_reports():
    records = [
        {SOURCE_REPORT_ROW_ID_COL: "berichte_row_0", "PatientenID": "p1", "bericht": "a"},
        {SOURCE_REPORT_ROW_ID_COL: "berichte_row_1", "PatientenID": "p1", "bericht": "b"},
        {SOURCE_REPORT_ROW_ID_COL: "berichte_row_2", "PatientenID": "p2", "bericht": "c"},
    ]
    done = _resume_keys_from_rows(
        [{SOURCE_REPORT_ROW_ID_COL: "berichte_row_0", "PatientenID": "p1", "bericht": "a"}]
    )
    remaining = _filter_unprocessed_reports(records, done)
    assert len(remaining) == 2
    assert remaining[0][SOURCE_REPORT_ROW_ID_COL] == "berichte_row_1"
