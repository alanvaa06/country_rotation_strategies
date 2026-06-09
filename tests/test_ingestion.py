import numpy as np
import pandas as pd
import pytest

from country_rotation.data import ingestion


def _mini():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")  # includes weekend
    df = pd.DataFrame({"A": np.arange(10.0), "B": np.arange(10.0) * 2}, index=idx)
    return {"PE": df.copy(), "EBITDA": df.copy()}


def test_remove_weekends():
    out = ingestion.remove_weekends(_mini())
    assert all(out["PE"].index.dayofweek < 5)


def test_slice_and_drop():
    dfs = _mini()
    out = ingestion.slice_by_date(dfs, "2020-01-05")
    assert out["PE"].index.min() >= pd.Timestamp("2020-01-05")
    out2 = ingestion.drop_countries(dfs, ("B",))
    assert "B" not in out2["PE"].columns


def test_publication_lag_shifts_data():
    dfs = _mini()
    out = ingestion.apply_publication_lag(dfs, {"EBITDA": 2})
    assert out["EBITDA"]["A"].iloc[2] == dfs["PE"]["A"].iloc[0]  # shifted by 2
    assert out["PE"].equals(dfs["PE"])  # unlagged metric untouched


def test_read_inputs_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingestion.read_inputs(str(tmp_path / "does_not_exist"))


def test_fill_gaps_never_uses_future():
    dfs = _mini()
    dfs["EBITDA"].iloc[3:5, 0] = np.nan
    out = ingestion.fill_gaps(dfs, metrics=("EBITDA",), limit=63)
    # gap filled with PAST value (2.0), never the future value (5.0)
    assert out["EBITDA"]["A"].iloc[3] == 2.0
    assert out["EBITDA"]["A"].iloc[4] == 2.0
    # leading NaN stays NaN (nothing in the past to fill from)
    dfs2 = _mini()
    dfs2["EBITDA"].iloc[0:2, 0] = np.nan
    out2 = ingestion.fill_gaps(dfs2, metrics=("EBITDA",), limit=63)
    assert np.isnan(out2["EBITDA"]["A"].iloc[0])
