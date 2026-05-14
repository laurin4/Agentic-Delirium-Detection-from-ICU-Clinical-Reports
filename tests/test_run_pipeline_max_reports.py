import src.pipeline.run_pipeline as run_pipeline_mod


def test_get_report_records_respects_max_reports(monkeypatch):
    fake = [{"PatientenID": f"p{i}", "report_text": "t", "bericht": f"{i}.csv"} for i in range(10)]
    monkeypatch.setattr(run_pipeline_mod, "INPUT_MODE", "diagnosis")
    monkeypatch.setattr(run_pipeline_mod, "build_patient_level_report_records", lambda: list(fake))
    monkeypatch.setattr(run_pipeline_mod, "MAX_REPORTS", 3)
    out = run_pipeline_mod._get_report_records()
    assert len(out) == 3
    assert out[0]["PatientenID"] == "p0"


def test_get_report_records_all_when_max_none(monkeypatch):
    fake = [{"PatientenID": "p0", "report_text": "t", "bericht": "0.csv"}]
    monkeypatch.setattr(run_pipeline_mod, "INPUT_MODE", "diagnosis")
    monkeypatch.setattr(run_pipeline_mod, "build_patient_level_report_records", lambda: list(fake))
    monkeypatch.setattr(run_pipeline_mod, "MAX_REPORTS", None)
    out = run_pipeline_mod._get_report_records()
    assert len(out) == 1
