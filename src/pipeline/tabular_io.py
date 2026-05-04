from pathlib import Path
import pandas as pd


def read_tabular(path):
    import pandas as pd

    for enc in ["utf-8-sig", "utf-16", "cp1252", "latin-1"]:
        try:
            return pd.read_csv(
                path,
                sep=";",
                encoding=enc,
                engine="python",
                on_bad_lines="skip"
            )
        except Exception:
            continue

    raise ValueError(f"CSV-Datei konnte nicht gelesen werden: {path}")