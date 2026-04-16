from __future__ import annotations

"""Generate synthetic CSV test data. Set DATA_MODE = 'synthetic' in paths.py to use outputs."""

import random
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.paths import DIAGNOSIS_EXAMPLES_DIR, STRUCTURED_RAW_DIR

RANDOM_SEED = 42
N_PATIENTS = 50

DIAGNOSIS_OUTPUT_PATH = DIAGNOSIS_EXAMPLES_DIR / "synthetic_diagnoses.csv"
ICD10_OUTPUT_PATH = STRUCTURED_RAW_DIR / "synthetic_icd10.csv"
ICDSC_OUTPUT_PATH = STRUCTURED_RAW_DIR / "synthetic_icdsc.csv"

VALID_DELIR_CODES = ["F05", "F05.0", "F05.8"]
EXCLUDED_DELIR_CODE = "F05.1"
NON_DELIR_CODES = ["I10", "J44.1", "A41", "K70.3", "N17.9", "R65.1", "E87.1"]

CORE_TEXTS = [
    "V.a. Delir bei Vigilanzminderung",
    "Z.n. OP, postoperativ intermittierend Desorientiert",
    "Sepsis mit Kreislaufinstabilitaet",
    "Enzephalopathie unklarer Genese",
    "Alkoholabusus in der Anamnese",
    "Desorientiert zu Zeit und Ort",
    "Vigilanzminderung im Nachtverlauf",
]
NOISE_TEXTS = [
    "Hypertonie bekannt",
    "COPD Exazerbation fraglich",
    "Elektrolytstoerung, Verlaufskontrolle empfohlen",
    "Mobilisation mit Physio fortsetzen",
    "Ernaehrung enteral, Bilanz beachten",
    "Wundkontrolle reizlos",
]
MESSY_SUFFIXES = [
    " ; DD: metabolisch/toxisch ",
    " | Verlauf: schwankend ",
    " ;;;;; reevaluation morgen",
    " // klinisch korrelieren",
]


def build_patient_cohorts() -> Tuple[Dict[str, int], Dict[int, int]]:
    patient_ids = [f"P{idx:04d}" for idx in range(1, N_PATIENTS + 1)]
    class_map: Dict[str, int] = {}

    for idx, pid in enumerate(patient_ids):
        if idx < 15:
            class_map[pid] = 2
        elif idx < 30:
            class_map[pid] = 1
        else:
            class_map[pid] = 0

    expected_distribution = {0: 20, 1: 15, 2: 15}
    return class_map, expected_distribution


def _build_diagnosis_value(target_class: int, rng: random.Random) -> str:
    core = rng.choice(CORE_TEXTS)
    noise = rng.choice(NOISE_TEXTS)

    if target_class == 2:
        class_phrase = rng.choice(
            [
                "delirantes Zustandsbild klinisch deutlich",
                "akuter Verwirrtheitszustand, engmaschige Ueberwachung",
                "psychomotorische Unruhe und Aufmerksamkeitsstoerung",
            ]
        )
    elif target_class == 1:
        class_phrase = rng.choice(
            [
                "fluktuierende Symptomatik, Delir nicht sicher",
                "diskrete kognitive Auffaelligkeiten",
                "episodische Unruhe ohne klare Delirdiagnose",
            ]
        )
    else:
        class_phrase = rng.choice(
            [
                "kein klinischer Hinweis auf Delir",
                "neurologisch weitgehend unauffaellig",
                "kognitiv adaequat bei Visite",
            ]
        )

    value = f"{core}. {class_phrase}. {noise}."
    if rng.random() < 0.35:
        value = f"{value}{rng.choice(MESSY_SUFFIXES)}"
    if rng.random() < 0.15:
        value = value.replace(" ", "  ")
    return value.strip()


def generate_diagnosis_data(class_map: Dict[str, int], rng: random.Random) -> pd.DataFrame:
    rows: List[dict] = []
    base_time = datetime(2026, 1, 1, 8, 0, 0)
    parameter_pool = [1110020, 1110042, 1110100, 1201001, 1303004]

    for patient_id, target_class in class_map.items():
        n_rows = rng.randint(5, 15)
        start_offset_days = rng.randint(0, 5)
        patient_base = base_time + timedelta(days=start_offset_days)

        for _ in range(n_rows):
            day_offset = rng.randint(0, 6)
            hour_offset = rng.randint(0, 23)
            minute_offset = rng.choice([0, 15, 30, 45])
            timestamp = patient_base + timedelta(days=day_offset, hours=hour_offset, minutes=minute_offset)

            rows.append(
                {
                    "PatientID": patient_id,
                    "ParameterID": rng.choice(parameter_pool),
                    "Time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "Value": _build_diagnosis_value(target_class, rng),
                }
            )

    df = pd.DataFrame(rows).sort_values(["PatientID", "Time"]).reset_index(drop=True)
    return df


def generate_icd10_data(class_map: Dict[str, int], rng: random.Random) -> pd.DataFrame:
    rows: List[dict] = []
    class1_modes = ["icd_only", "icdsc_high_no_flag", "flag_only"]
    class1_patients = [pid for pid, cls in class_map.items() if cls == 1]
    class1_mode_map = {pid: class1_modes[idx % len(class1_modes)] for idx, pid in enumerate(class1_patients)}

    for patient_id, target_class in class_map.items():
        n_codes = rng.randint(1, 3)
        code_entries: List[Tuple[str, str]] = []

        if target_class == 2:
            code_entries.append((rng.choice(VALID_DELIR_CODES), "1"))
        elif target_class == 1:
            mode = class1_mode_map[patient_id]
            if mode == "icd_only":
                code_entries.append((rng.choice(VALID_DELIR_CODES), rng.choice(["1", "0"])))
            elif mode == "icdsc_high_no_flag":
                if rng.random() < 0.5:
                    code_entries.append((EXCLUDED_DELIR_CODE, "0"))
            elif mode == "flag_only":
                if rng.random() < 0.6:
                    code_entries.append((EXCLUDED_DELIR_CODE, "0"))
        else:
            if rng.random() < 0.3:
                code_entries.append((EXCLUDED_DELIR_CODE, "0"))

        while len(code_entries) < n_codes:
            code_entries.append((rng.choice(NON_DELIR_CODES), rng.choice(["1", "0", "NULL"])))

        for code, main_flag in code_entries:
            rows.append(
                {
                    "PatientID": patient_id,
                    "Code": code,
                    "IsHauptDiagn": main_flag,
                }
            )

    return pd.DataFrame(rows)


def generate_icdsc_data(class_map: Dict[str, int], rng: random.Random) -> pd.DataFrame:
    rows: List[dict] = []
    base_time = datetime(2026, 1, 1, 6, 0, 0)
    class1_modes = ["icd_only", "icdsc_high_no_flag", "flag_only"]
    class1_patients = [pid for pid, cls in class_map.items() if cls == 1]
    class1_mode_map = {pid: class1_modes[idx % len(class1_modes)] for idx, pid in enumerate(class1_patients)}

    for patient_id, target_class in class_map.items():
        n_measures = rng.randint(4, 10)
        patient_base = base_time + timedelta(days=rng.randint(0, 5))

        for _ in range(n_measures):
            measure_time = patient_base + timedelta(hours=rng.randint(0, 24 * 7), minutes=rng.choice([0, 30]))

            if target_class == 2:
                value = rng.randint(4, 8)
                flag = 1 if rng.random() < 0.8 else 0
            elif target_class == 1:
                mode = class1_mode_map[patient_id]
                if mode == "icd_only":
                    value = rng.randint(0, 3)
                    flag = 0
                elif mode == "icdsc_high_no_flag":
                    value = rng.randint(4, 7)
                    flag = 0
                else:  # flag_only
                    value = rng.randint(1, 3)
                    flag = 1 if rng.random() < 0.7 else 0
            else:
                value = rng.randint(0, 3)
                flag = 0

            rows.append(
                {
                    "PatientID": patient_id,
                    "ICDSC_Time": measure_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "ICDSC_Value": value,
                    "ICDSC_DelirFlag": flag,
                }
            )

    return pd.DataFrame(rows).sort_values(["PatientID", "ICDSC_Time"]).reset_index(drop=True)


def compute_generated_classes(icd10_df: pd.DataFrame, icdsc_df: pd.DataFrame) -> Dict[str, int]:
    icd10 = icd10_df.copy()
    icdsc = icdsc_df.copy()

    icd10["PatientID"] = icd10["PatientID"].astype(str).str.strip()
    icd10["Code"] = icd10["Code"].astype(str).str.strip().str.upper()
    icd10["valid_delir_icd10"] = icd10["Code"].str.startswith("F05") & (icd10["Code"] != "F05.1")

    icd10_agg = (
        icd10.groupby("PatientID")["valid_delir_icd10"]
        .max()
        .reset_index()
        .rename(columns={"valid_delir_icd10": "has_delir_icd10"})
    )

    icdsc["PatientID"] = icdsc["PatientID"].astype(str).str.strip()
    icdsc["ICDSC_Value"] = pd.to_numeric(icdsc["ICDSC_Value"], errors="coerce")
    icdsc["ICDSC_DelirFlag"] = pd.to_numeric(icdsc["ICDSC_DelirFlag"], errors="coerce").fillna(0).astype(int)

    icdsc_agg = (
        icdsc.groupby("PatientID")
        .agg(max_icdsc=("ICDSC_Value", "max"), any_delir_flag=("ICDSC_DelirFlag", "max"))
        .reset_index()
    )

    merged = icd10_agg.merge(icdsc_agg, on="PatientID", how="outer").fillna({"has_delir_icd10": 0, "max_icdsc": 0, "any_delir_flag": 0})
    merged["has_delir_icd10"] = merged["has_delir_icd10"].astype(int)
    merged["any_delir_flag"] = merged["any_delir_flag"].astype(int)
    merged["max_icdsc"] = pd.to_numeric(merged["max_icdsc"], errors="coerce").fillna(0)

    generated_class = {}
    for _, row in merged.iterrows():
        has_icd = row["has_delir_icd10"] == 1
        has_flag = row["any_delir_flag"] == 1
        high_score = row["max_icdsc"] >= 4

        if has_icd and has_flag:
            cls = 2
        elif has_icd or has_flag or high_score:
            cls = 1
        else:
            cls = 0
        generated_class[str(row["PatientID"])] = cls

    return generated_class


def print_summary(
    expected_distribution: Dict[int, int],
    class_map: Dict[str, int],
    generated_class_map: Dict[str, int],
    diagnosis_df: pd.DataFrame,
    icd10_df: pd.DataFrame,
    icdsc_df: pd.DataFrame,
) -> None:
    expected = dict(sorted(expected_distribution.items()))
    target = dict(sorted(Counter(class_map.values()).items()))
    generated = dict(sorted(Counter(generated_class_map.values()).items()))

    print("\n=== Synthetic Data Generation Summary ===")
    print(f"Patients total: {len(class_map)}")
    print(f"Expected class distribution: {expected}")
    print(f"Target class distribution:   {target}")
    print(f"Generated class distribution:{generated}")
    print(f"Diagnosis rows: {len(diagnosis_df)}")
    print(f"ICD10 rows:     {len(icd10_df)}")
    print(f"ICDSC rows:     {len(icdsc_df)}")
    print(f"Saved diagnosis: {DIAGNOSIS_OUTPUT_PATH}")
    print(f"Saved ICD10:     {ICD10_OUTPUT_PATH}")
    print(f"Saved ICDSC:     {ICDSC_OUTPUT_PATH}")


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    class_map, expected_distribution = build_patient_cohorts()

    diagnosis_df = generate_diagnosis_data(class_map, rng)
    icd10_df = generate_icd10_data(class_map, rng)
    icdsc_df = generate_icdsc_data(class_map, rng)

    DIAGNOSIS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    STRUCTURED_RAW_DIR.mkdir(parents=True, exist_ok=True)

    diagnosis_df.to_csv(DIAGNOSIS_OUTPUT_PATH, index=False)
    icd10_df.to_csv(ICD10_OUTPUT_PATH, index=False)
    icdsc_df.to_csv(ICDSC_OUTPUT_PATH, index=False)

    generated_class_map = compute_generated_classes(icd10_df, icdsc_df)
    print_summary(expected_distribution, class_map, generated_class_map, diagnosis_df, icd10_df, icdsc_df)


if __name__ == "__main__":
    main()
