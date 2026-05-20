"""Robust Berichte.csv loading (malformed rows)."""

from pathlib import Path

import pandas as pd

from src.preprocessing.berichte_mapper import read_berichte_csv_robust


def test_read_berichte_csv_robust_skips_malformed_row(tmp_path):
    path = tmp_path / "Berichte.csv"
    path.write_text(
        "PatientID;bericht;bertyp;berdat\n"
        "p1;r1.txt;Verlaufseintrag;2024-01-01\n"
        "p2;r2.txt;Verlaufseintrag;2024-02-01;extra;bad;columns\n"
        "p3;r3.txt;Austrittsbericht;2024-03-01\n",
        encoding="utf-8",
    )
    df = read_berichte_csv_robust(path, log_context="test")
    assert len(df) == 2
    assert set(df["PatientID"].astype(str)) == {"p1", "p3"}
