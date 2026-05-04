import logging
import pandas as pd
from src.pipeline.paths import ICD10_PATH, ICDSC_PATH, STRUCTURED_BASELINE_PATH
from src.pipeline.tabular_io import read_tabular

LOGGER = logging.getLogger(__name__)


def _normalize_patient_column(df: pd.DataFrame) -> pd.DataFrame:
    if "PatientID" in df.columns: 
            df= df.rename(columns={"PatientID": "PatientenID"})
    return df


def load_data():
    if not ICD10_PATH.exists():
        raise FileNotFoundError(f"ICD10 input not found: {ICD10_PATH}")
    if not ICDSC_PATH.exists():
        raise FileNotFoundError(f"ICDSC input not found: {ICDSC_PATH}")
    icd10 = _normalize_patient_column(read_tabular(ICD10_PATH))
    icdsc = _normalize_patient_column(read_tabular(ICDSC_PATH))
    return icd10, icdsc



def _ensure_column(df: pd.DataFrame, column: str, default_value):
    if column not in df.columns:
        LOGGER.warning("Missing column '%s'. Filling default values.", column)
        df[column] = default_value
    return df


def _is_valid_delir_code(code: str) -> bool:
    normalized = str(code).strip().upper()
    return normalized in {"F05.0", "F05.8", "F05.9"}


def prepare_icd10(icd10: pd.DataFrame) -> pd.DataFrame:
    icd10 = icd10.copy()
    icd10 = _ensure_column(icd10, "PatientenID", "")
    icd10 = _ensure_column(icd10, "Code", "")
    icd10 = _ensure_column(icd10, "IsHauptDiagn", 0)

    icd10["PatientenID"] = icd10["PatientenID"].astype(str).str.strip()
    icd10["Code"] = icd10["Code"].astype(str)
    icd10["IsHauptDiagn"] = icd10["IsHauptDiagn"].astype(str)
    icd10["is_delir_icd10"] = icd10["Code"].apply(_is_valid_delir_code)

    grouped = (
        icd10.groupby("PatientenID")
        .agg(
            has_delir_icd10=("is_delir_icd10", "max"),
            delir_codes=("Code", lambda x: " | ".join(sorted(set(code for code in x if _is_valid_delir_code(code))))),
            has_main_delir_icd10=("IsHauptDiagn", lambda x: 0),  # Platzhalter, wird unten sauber gesetzt
        )
        .reset_index()
    )

    delir_subset = icd10.loc[icd10["is_delir_icd10"]].copy()
    if delir_subset.empty:
        main_delir = pd.DataFrame(columns=["PatientenID", "has_main_delir_icd10"])
    else:
        delir_subset["is_main"] = delir_subset["IsHauptDiagn"].astype(str).isin(["1", "True", "true", "JA", "Ja", "ja"])
        main_delir = (
            delir_subset.groupby("PatientenID")["is_main"]
            .max()
            .reset_index()
            .rename(columns={"is_main": "has_main_delir_icd10"})
        )

    grouped = grouped.drop(columns=["has_main_delir_icd10"]).merge(main_delir, on="PatientenID", how="left")
    grouped["has_main_delir_icd10"] = grouped["has_main_delir_icd10"].fillna(False).astype(int)
    grouped["has_delir_icd10"] = grouped["has_delir_icd10"].fillna(False).astype(int)
    grouped["delir_codes"] = grouped["delir_codes"].fillna("")

    return grouped


def prepare_icdsc(icdsc: pd.DataFrame) -> pd.DataFrame:
    icdsc = icdsc.copy()
    icdsc = _ensure_column(icdsc, "PatientenID", "")
    icdsc = _ensure_column(icdsc, "ICDSC_Time", None)
    icdsc = _ensure_column(icdsc, "ICDSC_Value", 0)
    icdsc = _ensure_column(icdsc, "ICDSC_DelirFlag", 0)

    icdsc["PatientenID"] = icdsc["PatientenID"].astype(str).str.strip()
    icdsc["ICDSC_Value"] = pd.to_numeric(icdsc["ICDSC_Value"], errors="coerce")
    icdsc["ICDSC_DelirFlag"] = pd.to_numeric(icdsc["ICDSC_DelirFlag"], errors="coerce").fillna(0).astype(int)
    icdsc["ICDSC_Time"] = pd.to_datetime(icdsc["ICDSC_Time"], errors="coerce")

    grouped = (
        icdsc.groupby("PatientenID")
        .agg(
            max_icdsc=("ICDSC_Value", "max"),
            any_delir_flag=("ICDSC_DelirFlag", "max"),
            n_icdsc_measurements=("ICDSC_Value", "count"),
            first_icdsc_time=("ICDSC_Time", "min"),
            last_icdsc_time=("ICDSC_Time", "max"),
        )
        .reset_index()
    )

    grouped["any_delir_flag"] = grouped["any_delir_flag"].fillna(0).astype(int)

    return grouped


def add_binary_baselines(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required_input_columns = ["has_delir_icd10", "max_icdsc"]
    missing_input_columns = [col for col in required_input_columns if col not in df.columns]
    if missing_input_columns:
        raise ValueError(
            "Cannot generate binary baselines: missing required columns: "
            + ", ".join(missing_input_columns)
        )

    df["has_delir_icd10"] = (
        pd.to_numeric(df["has_delir_icd10"], errors="coerce").fillna(0).astype(int)
    )
    df["max_icdsc"] = pd.to_numeric(df["max_icdsc"], errors="coerce").fillna(0)

    for threshold in [1, 2, 3, 4, 5]:
        df[f"baseline_icdsc_ge_{threshold}"] = (df["max_icdsc"] >= threshold).astype(int)
    df["baseline_icd10"] = (df["has_delir_icd10"] == 1).astype(int)
    # Additional ICDSC-shaped baselines (do not replace threshold columns above).
    df["baseline_icdsc_0"] = (df["max_icdsc"] == 0).astype(int)
    df["baseline_icdsc_1_to_3"] = (
        (df["max_icdsc"] >= 1) & (df["max_icdsc"] <= 3)
    ).astype(int)
    df["baseline_icdsc_ge_4_grouped"] = (df["max_icdsc"] >= 4).astype(int)
    return df


def add_reference_class(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_binary_baselines(df)

    def _assign_class(row):
        has_icd10_delir = row["has_delir_icd10"] == 1
        max_icdsc = row["max_icdsc"]

        if has_icd10_delir:
            return 2
        if max_icdsc >= 6:
            return 2
        if max_icdsc >= 4:
            return 1
        return 0

    df["baseline_reference_class"] = df.apply(_assign_class, axis=1)
    df["baseline_delir_reference"] = (df["baseline_reference_class"] == 2).astype(int)

    return df


def main():
    STRUCTURED_BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)

    icd10, icdsc = load_data()

    icd10_prepared = prepare_icd10(icd10)
    icdsc_prepared = prepare_icdsc(icdsc)

    merged = icd10_prepared.merge(icdsc_prepared, on="PatientenID", how="outer")
    merged = add_reference_class(merged)

    merged.to_csv(STRUCTURED_BASELINE_PATH, index=False)
    print(f"Gespeichert: {STRUCTURED_BASELINE_PATH}")
    print(f"Anzahl Patienten: {len(merged)}")


if __name__ == "__main__":
    main()