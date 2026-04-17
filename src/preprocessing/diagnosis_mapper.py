import logging
from pathlib import Path
from typing import List

import pandas as pd

from src.pipeline.paths import DIAGNOSIS_INPUT_PATH
from src.pipeline.tabular_io import read_tabular

LOGGER = logging.getLogger(__name__)
EXPECTED_COLUMNS = ["PatientID", "ParameterID", "Time", "Value"]
_CSV_ENCODINGS = ["utf-8-sig", "utf-16", "cp1252", "latin-1"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {col: str(col).strip() for col in df.columns}
    df = df.rename(columns=renamed).copy()

    if set(EXPECTED_COLUMNS).issubset(df.columns):
        return df

    # Fallback for files where the first column packs all fields as
    # "PatientID;ParameterID;Time;Value" and continuation text may be in another column.
    if len(df.columns) == 0:
        LOGGER.warning("Diagnosis input has no columns. Returning empty defaults.")
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    first_col = df.columns[0]
    packed = df[first_col].astype(str)
    split = packed.str.split(";", n=3, expand=True)
    if split.shape[1] == 4:
        split.columns = EXPECTED_COLUMNS
        split["Value"] = split["Value"].fillna("")

        if len(df.columns) > 1:
            continuation_col = df.columns[1]
            current_idx = None
            for idx, row in split.iterrows():
                row_patient_id = str(row["PatientID"]).strip()
                if row_patient_id and row_patient_id.lower() != "nan":
                    current_idx = idx
                else:
                    continuation = str(df.at[idx, continuation_col]).strip()
                    if continuation and continuation.lower() != "nan" and current_idx is not None:
                        previous_text = str(split.at[current_idx, "Value"]).strip()
                        split.at[current_idx, "Value"] = f"{previous_text}\n{continuation}".strip()

        split = split[split["PatientID"].astype(str).str.strip().str.lower() != "nan"]
        return split

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    for col in missing:
        LOGGER.warning("Diagnosis input missing column '%s'. Filling with defaults.", col)
        df[col] = ""

    return df[EXPECTED_COLUMNS]


def _read_diagnosis_file(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return _read_diagnosis_csv_manual(file_path)
    return read_tabular(file_path)


def _read_diagnosis_csv_manual(file_path: Path) -> pd.DataFrame:
    """
    Robuster CSV-Parser für Diagnose-Dateien mit Semikolon-Separator.
    Erwartet 4 Felder; alles nach dem dritten Semikolon bleibt in Value.
    """
    raw_text = None
    used_encoding = None
    for enc in _CSV_ENCODINGS:
        try:
            raw_text = file_path.read_text(encoding=enc)
            used_encoding = enc
            break
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            raise ValueError(f"Diagnose-CSV konnte nicht gelesen werden: {file_path} ({exc})") from exc

    if raw_text is None:
        raise ValueError(
            f"Diagnose-CSV konnte mit keiner unterstützten Kodierung gelesen werden: {file_path} "
            f"(versucht: {', '.join(_CSV_ENCODINGS)})"
        )

    lines = raw_text.splitlines()
    if not lines:
        raise ValueError(f"Diagnose-CSV ist leer: {file_path}")

    header = lines[0].strip().lstrip("\ufeff")
    expected_header = "PatientID;ParameterID;Time;Value"
    comma_header = "PatientID,ParameterID,Time,Value"
    if ";" not in header and "," in header:
        LOGGER.warning(
            "Diagnose-CSV %s scheint komma-separiert zu sein. Fallback auf pandas-Parser wird verwendet.",
            file_path.name,
        )
        return read_tabular(file_path)

    if header.replace(" ", "") != expected_header:
        LOGGER.warning(
            "Unerwarteter Header in Diagnose-CSV (%s): '%s' (erwartet: '%s')",
            file_path.name,
            header,
            expected_header,
        )

    parsed_rows = []
    malformed_count = 0
    for line_number, raw_line in enumerate(lines[1:], start=2):
        line = raw_line.strip("\n\r")
        if not line.strip():
            continue
        parts = line.split(";", 3)
        if len(parts) != 4:
            malformed_count += 1
            LOGGER.warning(
                "Zeile %d in %s konnte nicht korrekt geparst werden und wird übersprungen.",
                line_number,
                file_path.name,
            )
            continue
        patient_id, parameter_id, time_value, value = [p.strip() for p in parts]
        parsed_rows.append(
            {
                "PatientID": patient_id,
                "ParameterID": parameter_id,
                "Time": time_value,
                "Value": value,
            }
        )

    if not parsed_rows:
        raise ValueError(
            f"Diagnose-CSV konnte nicht geparst werden: {file_path}. "
            f"Keine gültigen Datenzeilen gefunden (Kodierung: {used_encoding})."
        )

    if malformed_count > 0:
        LOGGER.warning(
            "Diagnose-CSV %s: %d fehlerhafte Zeile(n) wurden übersprungen (Kodierung: %s).",
            file_path.name,
            malformed_count,
            used_encoding,
        )

    return pd.DataFrame(parsed_rows, columns=EXPECTED_COLUMNS)


def _load_diagnosis_rows(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        LOGGER.warning("Diagnosis input path does not exist: %s", input_path)
        return pd.DataFrame(columns=EXPECTED_COLUMNS)
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted([p for p in input_path.iterdir() if p.suffix.lower() in {".xlsx", ".xls", ".csv"}])
    else:
        LOGGER.warning("Diagnosis input is not a file or directory: %s", input_path)
        return pd.DataFrame(columns=EXPECTED_COLUMNS)
    if not files:
        LOGGER.warning("No diagnosis files found in %s. Returning empty report set.", input_path)
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    frames = []
    for file_path in files:
        try:
            frame = _read_diagnosis_file(file_path)
            frame = _normalize_columns(frame)
            frame["source_file"] = file_path.name
            frames.append(frame)
        except Exception as exc:
            LOGGER.warning("Could not parse diagnosis file %s: %s", file_path, exc)

    if not frames:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    for col in EXPECTED_COLUMNS:
        if col not in combined.columns:
            LOGGER.warning("Combined diagnosis data missing '%s'. Filling defaults.", col)
            combined[col] = ""

    return combined[EXPECTED_COLUMNS]


def build_patient_level_reports(input_dir: Path | None = None) -> pd.DataFrame:
    base_input = input_dir or DIAGNOSIS_INPUT_PATH
    rows = _load_diagnosis_rows(base_input)
    if rows.empty:
        return pd.DataFrame(columns=["PatientenID", "bericht", "report_text"])

    rows = rows.copy()
    rows["PatientID"] = rows["PatientID"].astype(str).str.strip()
    rows["Value"] = rows["Value"].fillna("").astype(str)

    if "Time" in rows.columns:
        rows["sort_time"] = pd.to_datetime(rows["Time"], errors="coerce")
    else:
        LOGGER.warning("Diagnosis data missing Time column. Keeping input order per patient.")
        rows["sort_time"] = pd.NaT

    rows = rows[rows["PatientID"].ne("") & rows["PatientID"].str.lower().ne("nan")]
    rows = rows.sort_values(["PatientID", "sort_time"], kind="stable")

    grouped = (
        rows.groupby("PatientID", dropna=False)["Value"]
        .apply(lambda values: "\n".join(v.strip() for v in values if str(v).strip()))
        .reset_index(name="report_text")
    )

    grouped = grouped.rename(columns={"PatientID": "PatientenID"})
    grouped["bericht"] = grouped["PatientenID"].apply(lambda pid: f"diagnosis_{pid}.txt")
    grouped["report_text"] = grouped["report_text"].fillna("")

    return grouped[["PatientenID", "bericht", "report_text"]]


def build_patient_level_report_records(input_dir: Path | None = None) -> List[dict]:
    df = build_patient_level_reports(input_dir)
    return df.to_dict(orient="records")
