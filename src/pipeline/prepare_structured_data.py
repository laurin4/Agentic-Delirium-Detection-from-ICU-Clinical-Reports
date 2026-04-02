import pandas as pd
from src.pipeline.paths import ICD10_PATH, ICDSC_PATH, STRUCTURED_BASELINE_PATH


def load_data():
    icd10 = pd.read_csv(ICD10_PATH)
    icdsc = pd.read_csv(ICDSC_PATH)
    return icd10, icdsc


def prepare_icd10(icd10: pd.DataFrame) -> pd.DataFrame:
    icd10 = icd10.copy()

    icd10["Code"] = icd10["Code"].astype(str)
    icd10["IsHauptDiagn"] = icd10["IsHauptDiagn"].astype(str)

    # Delir-relevante ICD-10 Codes, kann später erweitert werden
    delir_prefixes = ("F05",)

    icd10["is_delir_icd10"] = icd10["Code"].str.startswith(delir_prefixes)

    grouped = (
        icd10.groupby("PatientenID")
        .agg(
            has_delir_icd10=("is_delir_icd10", "max"),
            delir_codes=("Code", lambda x: " | ".join(sorted(set(x[x.astype(str).str.startswith(delir_prefixes)])))),
            has_main_delir_icd10=("IsHauptDiagn", lambda x: 0),  # Platzhalter, wird unten sauber gesetzt
        )
        .reset_index()
    )

    main_delir = (
        icd10[icd10["is_delir_icd10"]]
        .assign(is_main=lambda df: df["IsHauptDiagn"].astype(str).isin(["1", "True", "true", "JA", "Ja", "ja"]))
        .groupby("PatientenID")["is_main"]
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


def main():
    STRUCTURED_BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)

    icd10, icdsc = load_data()

    icd10_prepared = prepare_icd10(icd10)
    icdsc_prepared = prepare_icdsc(icdsc)

    merged = icd10_prepared.merge(icdsc_prepared, on="PatientenID", how="outer")

    merged.to_csv(STRUCTURED_BASELINE_PATH, index=False)
    print(f"Gespeichert: {STRUCTURED_BASELINE_PATH}")
    print(f"Anzahl Patienten: {len(merged)}")


if __name__ == "__main__":
    main()