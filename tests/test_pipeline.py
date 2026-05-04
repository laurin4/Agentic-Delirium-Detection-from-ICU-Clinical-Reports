import pandas as pd

from src.agents.classification import classify_delirium
from src.agents.interpretation import interpret_signals
from src.pipeline.prepare_structured_data import prepare_icd10, prepare_icdsc, add_reference_class
from src.preprocessing.diagnosis_mapper import build_patient_level_reports


def test_classify_delirium_high_returns_class_2():
    interpretation = {
        "signalstaerke": "hoch",
        "kontext": "explizite Delirdiagnose dokumentiert",
        "alternative_erklaerung": False,
        "begruendung": ["explizite Delirdiagnose vorhanden"],
    }

    result = classify_delirium(interpretation)

    assert result["klasse"] == 1
    assert result["klassifikation"] == "delir"



def test_classify_delirium_medium_returns_class_1():
    interpretation = {
        "signalstaerke": "mittel",
        "kontext": "indirekte Delir-Signale vorhanden",
        "alternative_erklaerung": False,
        "begruendung": ["mehrere indirekte Delir-Signale vorhanden"],
    }

    result = classify_delirium(interpretation)

    assert result["klasse"] == 1
    assert result["klassifikation"] == "delir"



def test_classify_delirium_low_returns_class_0():
    interpretation = {
        "signalstaerke": "niedrig",
        "kontext": "keine relevanten Delir-Signale dokumentiert",
        "alternative_erklaerung": False,
        "begruendung": ["keine relevanten Delir-Signale vorhanden"],
    }

    result = classify_delirium(interpretation)

    assert result["klasse"] == 0
    assert result["klassifikation"] == "kein_delir"



def test_interpret_signals_explicit_delir_is_high():
    text = "Patient entwickelte ein Delir auf Intensivstation."
    signals = {
        "desorientierung": [],
        "delir_explizit": ["Delir auf Intensivstation"],
        "hyperaktivitaet_agitation": [],
        "vigilanz": [],
        "delir_therapie": [],
        "delir_prophylaxe": [],
    }

    result = interpret_signals(text, signals)

    assert result["signalstaerke"] == "hoch"
    assert result["kontext"] == "explizite Delirdiagnose dokumentiert"



def test_interpret_signals_only_prophylaxis_is_low():
    text = "Es wurde eine Delirprophylaxe empfohlen."
    signals = {
        "desorientierung": [],
        "delir_explizit": [],
        "hyperaktivitaet_agitation": [],
        "vigilanz": [],
        "delir_therapie": [],
        "delir_prophylaxe": ["Delirprophylaxe"],
    }

    result = interpret_signals(text, signals)

    assert result["signalstaerke"] == "niedrig"
    assert result["kontext"] == "nur Delirprophylaxe dokumentiert"



def test_interpret_signals_multiple_indirect_signals_are_medium():
    text = "Patient war zeitlich desorientiert und zeigte Vigilanzminderung."
    signals = {
        "desorientierung": ["zeitlich desorientiert"],
        "delir_explizit": [],
        "hyperaktivitaet_agitation": [],
        "vigilanz": ["Vigilanzminderung"],
        "delir_therapie": [],
        "delir_prophylaxe": [],
    }

    result = interpret_signals(text, signals)

    assert result["signalstaerke"] == "mittel"
    assert result["kontext"] == "indirekte Delir-Signale vorhanden"



def test_prepare_icd10_detects_delir_code_and_main_diagnosis():
    df = pd.DataFrame(
        {
            "PatientenID": [1001, 1001, 1002],
            "Code": ["F05.0", "I10", "J44.1"],
            "IsHauptDiagn": ["1", "NULL", "1"],
        }
    )

    result = prepare_icd10(df)
    result = result.sort_values("PatientenID").reset_index(drop=True)

    assert result.loc[0, "PatientenID"] == "1001"
    assert result.loc[0, "has_delir_icd10"] == 1
    assert result.loc[0, "has_main_delir_icd10"] == 1
    assert "F05.0" in result.loc[0, "delir_codes"]

    assert result.loc[1, "PatientenID"] == "1002"
    assert result.loc[1, "has_delir_icd10"] == 0



def test_prepare_icdsc_aggregates_per_patient():
    df = pd.DataFrame(
        {
            "PatientenID": [1001, 1001, 1002],
            "ICDSC_Time": [
                "2026-01-01 08:00:00",
                "2026-01-01 16:00:00",
                "2026-01-02 08:00:00",
            ],
            "ICDSC_Value": [2, 5, 3],
            "ICDSC_DelirFlag": [0, 1, 0],
        }
    )

    result = prepare_icdsc(df)
    result = result.sort_values("PatientenID").reset_index(drop=True)

    assert result.loc[0, "PatientenID"] == "1001"
    assert result.loc[0, "max_icdsc"] == 5
    assert result.loc[0, "any_delir_flag"] == 1
    assert result.loc[0, "n_icdsc_measurements"] == 2

    assert result.loc[1, "PatientenID"] == "1002"
    assert result.loc[1, "max_icdsc"] == 3
    assert result.loc[1, "any_delir_flag"] == 0


def test_add_reference_class_uses_three_class_logic():
    df = pd.DataFrame(
        {
            "PatientenID": [1001, 1002, 1003],
            "has_delir_icd10": [1, 1, 0],
            "any_delir_flag": [1, 0, 0],
            "max_icdsc": [5, 2, 1],
        }
    )

    result = add_reference_class(df).sort_values("PatientenID").reset_index(drop=True)
    assert result.loc[0, "baseline_reference_class"] == 2
    assert result.loc[1, "baseline_reference_class"] == 2
    assert result.loc[2, "baseline_reference_class"] == 0
    assert result.loc[0, "baseline_delir_reference"] == 1
    assert result.loc[1, "baseline_delir_reference"] == 1
    assert result.loc[0, "baseline_icd10"] == 1
    assert result.loc[2, "baseline_icdsc_ge_1"] == 1
    assert result.loc[2, "baseline_icdsc_ge_2"] == 0


def test_build_patient_level_reports_groups_and_sorts(tmp_path):
    raw = pd.DataFrame(
        {
            "PatientID": [1001, 1001, 1002],
            "ParameterID": [1, 1, 2],
            "Time": ["2026-01-01 09:00:00", "2026-01-01 08:00:00", "2026-01-02 12:00:00"],
            "Value": ["later entry", "earlier entry", "single entry"],
        }
    )
    input_file = tmp_path / "diagnose.csv"
    raw.to_csv(input_file, index=False)

    reports = build_patient_level_reports(tmp_path).sort_values("PatientenID").reset_index(drop=True)

    assert list(reports["PatientenID"]) == ["1001", "1002"]
    assert reports.loc[0, "bericht"] == "diagnosis_1001.txt"
    assert reports.loc[0, "report_text"] == "earlier entry\nlater entry"
    assert reports.loc[1, "report_text"] == "single entry"