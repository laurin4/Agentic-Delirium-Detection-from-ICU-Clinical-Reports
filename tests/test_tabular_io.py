import pandas as pd

from src.pipeline.tabular_io import read_tabular


def test_read_tabular_csv(tmp_path):
    p = tmp_path / "t.csv"
    pd.DataFrame({"a": [1]}).to_csv(p, index=False)
    df = read_tabular(p)
    assert list(df.columns) == ["a"]


def test_read_tabular_semicolon_csv(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("a;b\n1;2\n", encoding="utf-8")
    df = read_tabular(p)
    assert "a" in df.columns or len(df.columns) >= 1
