"""Merge paths from ETL results back to data."""

import argparse
from pathlib import Path

import pandas as pd


def merge_data(
    split_summary: Path,
    data: Path,
    out: Path,
    tag_col: str = "tag",
    num_col: str = "num",
    remove_empty_paths: bool = True,
):
    """Merge path from split_summary to data.

    Args:
        split_summary: Path to split summary parquet file. The data
          must contain columns 'tag', 'num' and 'path'.
        data: Path to data file, containing records with a tag and
          record number, corresponding to split_summary file.
        out: Path to output Parquet file location.
        tag_col: Column in `data`, which maps to split_summary 'tag'
          column. Defaults to "tag".
        num_col: Column in `data`, which maps to split_summary 'seg_num`
          column. Defaults to "num".
        remove_empty_paths: Drop records, whose path in split_summary
          is NA. Defaults to True.
    """
    orig_data = pd.read_parquet(data)
    summary_data = pd.read_parquet(split_summary)
    joined = pd.merge(
        orig_data,
        summary_data[["tag", "num", "path"]],
        left_on=[tag_col, num_col],
        right_on=["tag", "num"],
        how="inner",
    )

    joined = joined.drop(
        columns=[i for i in ["tag", "num"] if i not in orig_data.columns]
    )

    if remove_empty_paths:
        joined = joined[~joined["path"].isna()]

    joined.to_parquet(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--filter",
        dest="remove_empty_paths",
        help="Remove recods, whose path is not determined in split_summary",
        action="store_true",
    )
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
    merge_data(**parser_kwargs)
