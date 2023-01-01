"""After ETL, filter datasets to only include existing segments."""


import argparse
from pathlib import Path

import pandas as pd


def filter_data(
    split_summary: Path,
    data: Path,
    out: Path,
    tag_col: str = "tag",
    num_col: str = "num",
):
    orig_data = pd.read_parquet(data)
    summary_data = pd.read_parquet(split_summary)
    joined = pd.merge(
        orig_data,
        summary_data[["tag", "num", "path"]],
        left_on=[tag_col, num_col],
        right_on=["tag", "num"],
        how="inner",
    )
    joined = joined[~joined["path"].isna()].drop(
        columns=[i for i in ["tag", "num", "path"] if i not in orig_data.columns]
    )

    joined.to_parquet(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tag-col",
        help="Column name in data for tag",
        default="tag",
        type=str,
    )
    parser.add_argument(
        "--num-col",
        help="Column name in data for segment number",
        default="num",
        type=str,
    )
    parser.add_argument(
        "split_summary",
        help="Split summary Parquet location.",
        type=Path,
    )
    parser.add_argument(
        "data",
        help="Segment definition input Parquet file location.",
        type=Path,
    )
    parser.add_argument(
        "out",
        help="Filtered output file location.",
        type=Path,
    )

    parser_args = parser.parse_args()
    parser_kwargs = vars(parser_args)
    filter_data(**parser_kwargs)
