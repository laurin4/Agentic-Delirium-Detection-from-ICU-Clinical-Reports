from __future__ import annotations

"""
Generate realistic synthetic ICU delirium data under `data/raw/` for offline testing.

Creates exactly:
  - data/raw/Diagnosenliste.csv
  - data/raw/ICD.csv
  - data/raw/ICDSC.csv
  - data/raw/Berichte.xlsx

All CSVs use semicolon separator to match pipeline expectations.
"""

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd


@dataclass(frozen=True)
class PatientTextScenario:
    # Controls diagnosis/report text content which drives the text model signals.
    category: str  # "explicit_delir" | "indirect" | "normal"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _raw_dir() -> Path:
    return _project_root() / "data" / "raw"


def _iso_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _build_patient_sets(
    rng: random.Random,
    n_patients: int,
    icd10_valid_n: int,
    icd10_f051_only_n: int,
    icdsc_high_n: int,
    icdsc_mid_n: int,
) -> Tuple[Set[int], Set[int], Set[int], Set[int], Set[int], Set[int]]:
    patient_ids = list(range(1, n_patients + 1))
    rng.shuffle(patient_ids)

    icd10_valid_ids = set(patient_ids[:icd10_valid_n])
    icd10_f051_only_ids = set(patient_ids[icd10_valid_n : icd10_valid_n + icd10_f051_only_n])
    icd10_none_ids = set(patient_ids) - icd10_valid_ids - icd10_f051_only_ids

    rng.shuffle(patient_ids)
    icdsc_high_ids = set(patient_ids[:icdsc_high_n])
    icdsc_mid_ids = set(patient_ids[icdsc_high_n : icdsc_high_n + icdsc_mid_n])
    icdsc_none_ids = set(patient_ids) - icdsc_high_ids - icdsc_mid_ids

    return (
        icd10_valid_ids,
        icd10_f051_only_ids,
        icd10_none_ids,
        icdsc_high_ids,
        icdsc_mid_ids,
        icdsc_none_ids,
    )


def _assign_text_scenarios(
    rng: random.Random,
    n_patients: int,
    icd10_valid_ids: Set[int],
    icdsc_high_ids: Set[int],
    icdsc_mid_ids: Set[int],
    icdsc_none_ids: Set[int],
) -> Dict[int, PatientTextScenario]:
    """
    Create partial alignment between structured baselines and text.

    Explicit delirium / indirect delirium phrases influence the text model signals.
    We intentionally introduce mismatches so the evaluation can show non-perfect agreement.
    """

    scenarios: Dict[int, PatientTextScenario] = {}
    for pid in range(1, n_patients + 1):
        if pid in icd10_valid_ids:
            # Mostly aligned: valid ICD10 delir -> explicit delir text.
            roll = rng.random()
            if roll < 0.70:
                scenarios[pid] = PatientTextScenario("explicit_delir")
            else:
                scenarios[pid] = PatientTextScenario("normal")
        elif pid in icdsc_high_ids:
            # ICDSC high -> often delir-ish text, but not always.
            roll = rng.random()
            if roll < 0.30:
                scenarios[pid] = PatientTextScenario("explicit_delir")
            elif roll < 0.80:
                scenarios[pid] = PatientTextScenario("indirect")
            else:
                scenarios[pid] = PatientTextScenario("normal")
        elif pid in icdsc_mid_ids:
            # Mid ICDSC -> mostly normal, but some indirect symptoms.
            roll = rng.random()
            if roll < 0.40:
                scenarios[pid] = PatientTextScenario("indirect")
            else:
                scenarios[pid] = PatientTextScenario("normal")
        elif pid in icdsc_none_ids:
            # No ICDSC -> mostly normal; a few mismatches (indirect symptoms) remain.
            roll = rng.random()
            if roll < 0.15:
                scenarios[pid] = PatientTextScenario("indirect")
            else:
                scenarios[pid] = PatientTextScenario("normal")
        else:
            scenarios[pid] = PatientTextScenario("normal")
    return scenarios


def _build_diagnosis_values(scenario: PatientTextScenario, rng: random.Random) -> List[str]:
    explicit_delir_phrases = [
        "Patient delirant",
        "akutes Delir",
        "delirös",
        "Delir",
        "Delirverdacht",
        "hyperaktives Delir",
        "Delirium",
        "Delirtherapie mit Haloperidol",
        "neuroleptische Behandlung: Quetiapin",
        "Delirprophylaxe mit Melatonin",
        "hypoaktives Delir",
    ]
    # Indirect symptoms (no explicit "Delir"/"Delirium" tokens to keep it indirect).
    indirect_phrases = [
        "verwirrt",
        "unruhig",
        "somnolent",
        "reduzierte Vigilanz",
        "Vigilanzminderung",
        "Desorientiert zu Zeit und Ort",
        "zeitlich desorientiert",
        "Bewusstseinsstörung",
        "Agitiertheit",
        "motorisch unruhig",
        "soporös",
        "hypoaktive Symptomatik",
    ]
    normal_phrases = [
        "wach und orientiert",
        "stabil",
        "keine Auffälligkeiten",
        "neurologisch weitgehend unauffaellig",
        "kognitiv unauffaellig",
        "keine Auffälligkeiten im Verlauf",
    ]

    if scenario.category == "explicit_delir":
        pool = explicit_delir_phrases
    elif scenario.category == "indirect":
        pool = indirect_phrases
    else:
        pool = normal_phrases

    # Return a small list; caller can pick per row.
    return rng.sample(pool, k=min(6, len(pool)))


def generate_diagnosenliste(
    rng: random.Random,
    scenarios: Dict[int, PatientTextScenario],
    n_patients: int,
) -> pd.DataFrame:
    rows: List[dict] = []
    base_time = datetime(2024, 1, 1, 10, 0, 0)

    # Ensure we land in ~300–600 rows total.
    for pid in range(1, n_patients + 1):
        n_rows = rng.randint(6, 12)
        patient_start = base_time + timedelta(days=rng.randint(0, 40))
        values_pool = _build_diagnosis_values(scenarios[pid], rng)

        for _ in range(n_rows):
            t = patient_start + timedelta(
                days=rng.randint(0, 25),
                hours=rng.randint(0, 23),
                minutes=rng.choice([0, 15, 30, 45]),
            )
            value = rng.choice(values_pool)
            rows.append(
                {
                    "PatientID": pid,
                    "ParameterID": "diag",
                    "Time": _iso_dt(t),
                    "Value": value,
                }
            )

    df = pd.DataFrame(rows).sort_values(["PatientID", "Time"]).reset_index(drop=True)
    return df


def generate_icd10(
    rng: random.Random,
    n_patients: int,
    icd10_valid_ids: Set[int],
    icd10_f051_only_ids: Set[int],
    icd10_none_ids: Set[int],
) -> pd.DataFrame:
    valid_codes = ["F05.0", "F05.8", "F05.9"]
    excluded_code = "F05.1"

    non_delir_codes = [
        "I10",
        "J96",
        "E11",
        "A41",
        "K70.3",
        "N17.9",
        "R65.1",
        "M62.81",
        "G93.4",
        "R13.1",
        "J44.1",
        "I48.0",
    ]

    rows: List[dict] = []
    for pid in range(1, n_patients + 1):
        n_codes = rng.randint(1, 3)
        codes: List[str] = []

        if pid in icd10_valid_ids:
            codes.append(rng.choice(valid_codes))
            while len(codes) < n_codes:
                codes.append(rng.choice(non_delir_codes))
        elif pid in icd10_f051_only_ids:
            codes.append(excluded_code)
            while len(codes) < n_codes:
                codes.append(rng.choice(non_delir_codes))
        elif pid in icd10_none_ids:
            while len(codes) < n_codes:
                codes.append(rng.choice(non_delir_codes))
        else:
            while len(codes) < n_codes:
                codes.append(rng.choice(non_delir_codes))

        # Mark exactly one code as main diagnosis.
        main_idx = rng.randrange(0, len(codes))
        for i, code in enumerate(codes):
            rows.append(
                {
                    "PatientID": pid,
                    "Code": code,
                    "IsHauptDiagn": 1 if i == main_idx else 0,
                }
            )

    df = pd.DataFrame(rows)
    return df


def _choose_icdsc_max_value(rng: random.Random, scenario: str) -> int:
    if scenario == "high":
        # Bias towards 4 so thresholds >=5 differ.
        if rng.random() < 0.45:
            return 4
        return rng.randint(5, 8)
    if scenario == "mid":
        return rng.randint(1, 3)
    return 0


def generate_icdsc(
    rng: random.Random,
    n_patients: int,
    icdsc_high_ids: Set[int],
    icdsc_mid_ids: Set[int],
    icdsc_none_ids: Set[int],
) -> pd.DataFrame:
    rows: List[dict] = []
    base_time = datetime(2024, 1, 1, 6, 0, 0)

    for pid in range(1, n_patients + 1):
        if pid in icdsc_high_ids:
            group = "high"
        elif pid in icdsc_mid_ids:
            group = "mid"
        else:
            group = "none"

        max_val = _choose_icdsc_max_value(rng, group)
        patient_base = base_time + timedelta(days=rng.randint(0, 40))
        n_measures = rng.randint(3, 5)  # keeps total around ~150–250

        # Create values <= max_val and ensure at least one at max_val.
        values: List[int] = []
        for j in range(n_measures):
            if j == n_measures - 1:
                values.append(max_val)
            else:
                values.append(rng.randint(0, max_val))
        # If max_val is 0, ensure all are 0.
        if max_val == 0:
            values = [0 for _ in range(n_measures)]

        for value in values:
            # Delir flag follows score but with noise.
            if value == 0:
                flag = 0
            elif value >= 4:
                flag = 1 if rng.random() < 0.80 else 0
            else:
                flag = 1 if rng.random() < 0.30 else 0

            t = patient_base + timedelta(
                days=rng.randint(0, 25),
                hours=rng.randint(0, 24),
                minutes=rng.choice([0, 30]),
            )
            rows.append(
                {
                    "PatientID": pid,
                    "ICDSC_Time": _iso_dt(t),
                    "ICDSC_Value": int(value),
                    "ICDSC_DelirFlag": int(flag),
                }
            )

    df = pd.DataFrame(rows).sort_values(["PatientID", "ICDSC_Time"]).reset_index(drop=True)
    return df


def _pick_name(rng: random.Random) -> str:
    names = ["Dr. Müller", "Dr. Schmidt", "Prof. Weber"]
    return rng.choice(names)


def generate_berichte_xlsx(
    rng: random.Random,
    scenarios: Dict[int, PatientTextScenario],
    n_patients: int,
    out_path: Path,
) -> None:
    base_date = datetime(2024, 1, 1)
    rows: List[dict] = []

    for pid in range(1, n_patients + 1):
        berdat = base_date + timedelta(days=rng.randint(0, 220))
        scenario = scenarios[pid]

        if scenario.category == "explicit_delir":
            diag = rng.choice(
                [
                    "Deliröses Zustandsbild mit zeitlicher Desorientierung",
                    "akutes Delir bei Vigilanzminderung",
                    "hyperaktives Delir mit Unruhe und Aufmerksamkeitsstörung",
                ]
            )
            epikrise__status = rng.choice(
                [
                    "Im Verlauf deutliche Delir-Symptomatik, zeitweise Sedierung unter engmaschiger Überwachung",
                    "Delirzeichen fluktuierten; zuletzt Verbesserung der Vigilanz",
                ]
            )
            jetziges_leiden__tv_titel = rng.choice(
                ["Delir-Symptomatik", "Akutes Delir", "Delirverdacht"]
            )
            prozedere__tv_inhalt = rng.choice(
                [
                    "Delirtherapie: Haloperidol; Delirprophylaxe mit Melatonin. Physiotherapie und Reorientierung fortführen.",
                    "Bei Agitiertheit neuroleptische Behandlung (Quetiapin) erwogen; Vigilanztraining und Schlafhygiene.",
                ]
            )
        elif scenario.category == "indirect":
            diag = rng.choice(
                [
                    "verwirrt und unruhig, reduzierte Vigilanz",
                    "somnolent mit Vigilanzminderung",
                    "desorientiert zu Zeit und Ort, agitationstypische Symptomatik",
                ]
            )
            epikrise__status = rng.choice(
                [
                    "zeitweise Vigilanzminderung, Reorientierung und nicht-pharmakologische Maßnahmen durchgeführt",
                    "Fluktuierende Unruhe ohne gesicherte Delirdiagnose; Verlaufskontrolle empfohlen",
                ]
            )
            jetziges_leiden__tv_titel = rng.choice(
                ["Akute Verwirrtheit", "Vigilanzminderung", "Somnolenz"]
            )
            prozedere__tv_inhalt = rng.choice(
                [
                    "Delirprophylaxe: Melatonin und strukturierte Tag-Nacht-Routine. Reorientierung und Hör-/Sehhilfen prüfen.",
                    "Nicht-pharmakologische Delirprophylaxe fortführen; Mobilisation und Schlafhygiene.",
                ]
            )
        else:
            diag = rng.choice(
                [
                    "wach und orientiert bei stabiler Situation",
                    "stabil, keine Auffälligkeiten im neurologischen Status",
                    "keine Auffälligkeiten, kognitiv unauffaellig",
                ]
            )
            epikrise__status = rng.choice(
                [
                    "klinisch unauffaellig; Patient wach und orientiert, Verlauf unauffaellig",
                    "klinisch stabile Phase ohne agitiertes Unruhezustandsbild",
                ]
            )
            jetziges_leiden__tv_titel = rng.choice(
                ["Stabile Situation", "Keine Auffälligkeiten", "Kognitiv unauffaellig"]
            )
            prozedere__tv_inhalt = rng.choice(
                [
                    "Weiterhin Standardtherapie; Mobilisation und ausreichende Flüssigkeitszufuhr. Schlafhygiene und Reorientierung nach Standard.",
                    "Reorientierung im Tagesverlauf, Schlafhygiene; sonst keine spezifischen Maßnahmen erforderlich.",
                ]
            )

        rows.append(
            {
                "patnr": pid,
                "fallnr": f"FALL{pid:05d}",
                "berdat": berdat.date().isoformat(),
                "bertyp": "Austrittsbericht",
                "bername": _pick_name(rng),
                "diag": diag,
                "epikrise__status": epikrise__status,
                "jetziges_leiden__tv_titel": jetziges_leiden__tv_titel,
                "prozedere__tv_inhalt": prozedere__tv_inhalt,
            }
        )

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Berichte")


def main() -> None:
    rng = random.Random(42)

    n_patients = 50
    icd10_valid_n = 20
    icd10_f051_only_n = 7  # should NOT count as delir in ICD10 baseline
    icdsc_high_n = 25  # some patients with max >= 4
    icdsc_mid_n = 15  # some patients with max in 1..3
    icdsc_none_n = n_patients - icdsc_high_n - icdsc_mid_n
    assert icdsc_none_n >= 0

    (
        icd10_valid_ids,
        icd10_f051_only_ids,
        icd10_none_ids,
        icdsc_high_ids,
        icdsc_mid_ids,
        icdsc_none_ids,
    ) = _build_patient_sets(
        rng=rng,
        n_patients=n_patients,
        icd10_valid_n=icd10_valid_n,
        icd10_f051_only_n=icd10_f051_only_n,
        icdsc_high_n=icdsc_high_n,
        icdsc_mid_n=icdsc_mid_n,
    )

    scenarios = _assign_text_scenarios(
        rng=rng,
        n_patients=n_patients,
        icd10_valid_ids=icd10_valid_ids,
        icdsc_high_ids=icdsc_high_ids,
        icdsc_mid_ids=icdsc_mid_ids,
        icdsc_none_ids=icdsc_none_ids,
    )

    diagnoses_df = generate_diagnosenliste(
        rng=rng,
        scenarios=scenarios,
        n_patients=n_patients,
    )
    icd10_df = generate_icd10(
        rng=rng,
        n_patients=n_patients,
        icd10_valid_ids=icd10_valid_ids,
        icd10_f051_only_ids=icd10_f051_only_ids,
        icd10_none_ids=icd10_none_ids,
    )
    icdsc_df = generate_icdsc(
        rng=rng,
        n_patients=n_patients,
        icdsc_high_ids=icdsc_high_ids,
        icdsc_mid_ids=icdsc_mid_ids,
        icdsc_none_ids=icdsc_none_ids,
    )

    out_dir = _raw_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    diagnoses_path = out_dir / "Diagnosenliste.csv"
    icd10_path = out_dir / "ICD.csv"
    icdsc_path = out_dir / "ICDSC.csv"
    berichte_path = out_dir / "Berichte.xlsx"

    diagnoses_df.to_csv(diagnoses_path, index=False, sep=";")
    icd10_df.to_csv(icd10_path, index=False, sep=";")
    icdsc_df.to_csv(icdsc_path, index=False, sep=";")
    generate_berichte_xlsx(
        rng=rng,
        scenarios=scenarios,
        n_patients=n_patients,
        out_path=berichte_path,
    )

    # Required prints
    icdsc_value_dist = icdsc_df["ICDSC_Value"].value_counts().sort_index()
    n_patients_total = n_patients
    valid_delir_icd10_patient_ids = set(
        icd10_df.loc[icd10_df["Code"].isin(["F05.0", "F05.8", "F05.9"]), "PatientID"].astype(int).tolist()
    )

    print("=== Synthetic raw data generated ===")
    print(f"Total number of patients: {n_patients_total}")
    print("ICDSC_Value distribution (counts):")
    print(icdsc_value_dist.to_dict())
    print(
        "Number of patients with valid delir ICD10 (F05.0, F05.8, F05.9):",
        len(valid_delir_icd10_patient_ids),
    )
    print(f"Saved: {diagnoses_path}")
    print(f"Saved: {icd10_path}")
    print(f"Saved: {icdsc_path}")
    print(f"Saved: {berichte_path}")


if __name__ == "__main__":
    main()

