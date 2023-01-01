"""Test filtering"""

from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from singing_classifier.filter import filter_data


def test_filter_data(tmp_path: Path):
    """Filtered data does not contain records with None outputh path."""
    split_summary = tmp_path / "split_summary.parquet"
    pd.DataFrame(
        {
            "tag": ["BaW_jenozKc", "BaW_jenozKc", "pRnPUCeL76M"],
            "num": [0, 1, 0],
            "path": ["data/0.m4a", None, "data/2.m4a"],
            "message": ["", "", ""],
        }
    ).convert_dtypes().to_parquet(split_summary)

    data = tmp_path / "data.parquet"
    orig_data = pd.DataFrame(
        {
            "tagx": ["BaW_jenozKc", "BaW_jenozKc", "pRnPUCeL76M"],
            "seg_num": [0, 1, 0],
            "col2": [0, 4, 7],
            "col5": ["f", "o", "o"],
        }
    ).convert_dtypes()
    orig_data.to_parquet(data)

    expected = orig_data.iloc[[0, 2], :]

    out = tmp_path / "clean.parquet"

    filter_data(split_summary, data, out, tag_col="tagx", num_col="seg_num")
    received = pd.read_parquet(out)

    assert_frame_equal(received, expected)
