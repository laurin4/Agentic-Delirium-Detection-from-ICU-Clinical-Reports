"""End-to-end pipeline smoke test with stubbed LLM (no USZ/Ollama, no real Berichte.csv)."""

import json

import pandas as pd

import src.models.llm_interface as llm_interface
from src.pipeline import run_pipeline


def _fake_call_llm(messages):
    """Return valid JSON for Agent 1 (extraction) vs Agent 2 (interpretation)."""
    user = ""
    if messages and isinstance(messages[-1], dict):
        user = str(messages[-1].get("content", ""))
    if "Extrahierte Signale (JSON):" in user:
        return json.dumps(
            {
                "signalstaerke": "niedrig",
                "kontext": "stub",
                "alternative_erklaerung": False,
                "alternative_erklaerung_keywords": [],
                "begruendung": [],
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "desorientierung": [],
            "delir_explizit": [],
            "hyperaktivitaet_agitation": [],
            "vigilanz": [],
            "delir_therapie": [],
            "delir_prophylaxe": [],
        },
        ensure_ascii=False,
    )


def test_run_pipeline_prompt_mode_with_stubbed_llm(monkeypatch, tmp_path, capsys):
    assert run_pipeline.INTERPRETATION_MODE == "prompt"

    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_pipeline, "PREDICTIONS_DIR", pred_dir)

    stub_records = [
        {
            "PatientenID": "stub_patient_001",
            "bericht": "berichte_stub_patient_001.txt",
            "report_text": "Kein Delir dokumentiert. Patient orientiert.",
        }
    ]
    monkeypatch.setattr(run_pipeline, "_get_report_records", lambda: stub_records)
    monkeypatch.setattr(llm_interface, "call_llm", _fake_call_llm)

    run_pipeline.main()

    standard_path = pred_dir / "agent1_agent2_agent3_results_prompt.csv"
    assert standard_path.is_file()

    df = pd.read_csv(standard_path)
    assert df["klasse"].isin([0, 1]).all()
    assert df["klassifikation"].iloc[0] == "kein_delir"

    slug = run_pipeline._sanitize_provider_model_slug(
        run_pipeline.LLM_PROVIDER,
        run_pipeline.LLM_MODEL_LABEL,
    )
    tagged_path = pred_dir / f"agent_results_{slug}.csv"
    assert tagged_path.is_file()

    capsys.readouterr()


def test_fake_llm_returns_distinct_json_shapes():
    """Stub distinguishes extraction vs interpretation user prompts."""
    ext_messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Bericht:\nx"},
    ]
    interp_messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Klinischer Bericht:\nx\n\nExtrahierte Signale (JSON):\n{}"},
    ]
    out_ext = json.loads(_fake_call_llm(ext_messages))
    out_int = json.loads(_fake_call_llm(interp_messages))
    assert "delir_explizit" in out_ext
    assert "signalstaerke" in out_int
