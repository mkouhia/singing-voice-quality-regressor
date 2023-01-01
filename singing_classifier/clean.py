"""Clean input dataframes."""

import argparse
from collections.abc import Collection, Sequence
from os import PathLike
from pathlib import Path

import pandas as pd


def extract_id(series: pd.Series, pattern=r"([^&]+).*$") -> pd.Series:
    """Extract video id from url parameter series."""
    return series.str.extract(pattern, expand=False)


def ensure_unique(
    data: pd.DataFrame,
    columns: Collection[str] = None,
    ignore_cols: Collection[str] = None,
) -> pd.DataFrame:
    """Make sure that every row in the input data is unique.
    
    If duplicate keys are found, rows are dropped silently if the whole
    row is equal.
    
    Args:
        data: Input dataframe.
        columns: Names of columns, that represent unique keys for the row.
        ignore_cols: Names of columns, to be ignored in comparison.
    
    Raises:
        UserWarning If such rows are found, where keys are equal and
        other content in the row is not equal
    
    Returns:
        Dataframe with possible duplicate rows removed.
    """
    ignore_cols = ignore_cols or []
    duplicate_idx = data.drop(columns=ignore_cols).duplicated(subset=columns)
    duplicate_rows = data.drop(columns=ignore_cols).duplicated()
    
    # Duplicate indices, that are not whole row matches
    if (duplicate_idx ^ duplicate_rows).any():
        dup_values = (
            (data if columns is None else data[columns])[duplicate_idx]
            .drop_duplicates()
            .values.tolist()
        )
        raise UserWarning(f"Data contains duplicate keys: {dup_values}")
    
    # Drop duplicate indices, that are whole row matches
    return data[~duplicate_idx]


def clean_segments(
    origin: PathLike,
    name_col: str = "name",
    tag_col: str = "tag",
    other_unique_cols: Collection[str] = ("num",),
    ensure_order: Collection[tuple[str, str]] = (("time_start", "time_end"),),
    drop_tags: Collection[str] = None,
) -> pd.DataFrame:
    """Fix identifiers in origin csv file.

    Args:
        origin: Source csv location.
        name_col: Column name, from where tag is extracted.
        tag_col: Column name, to which tags are stored in result.
        other_unique_cols: Key columns in addition to `tag_col`.
        drop_tags: Drop rows, whose tag matches any of given values.
        ensure_order: Collection of tuples, whose values are column
          names than should be ordered. For example, value in
          `time_start` column should always be smaller than the value
          in `time_end` column. Drop rows, that do not meet this
          condition.

    Returns:
        Parsed dataframe, with new tag column and other unique
        columns at the front.

    Raises:
        UserWarning if input data contains non-unique rows by
        `name_col` and `other_unique_cols`, or if after tag parsing
        there are non-unique rows in combination of `tag_col`
        and `other_unique_cols`.
    """
    other_unique_cols = list(other_unique_cols) or []
    data = pd.read_csv(origin)

    data[tag_col] = extract_id(data[name_col])
    
    for col_small, col_large in ensure_order:
        data = data[data[col_small] < data[col_large]]

    key_cols = [tag_col] + list(other_unique_cols)
    new_order = key_cols + [col for col in data.columns if col not in key_cols]
    data = data[new_order]

    if drop_tags is not None:
        data = data[~data[tag_col].isin(drop_tags)]

    return ensure_unique(data, columns=key_cols, ignore_cols=[name_col])


def clean_data(
    origin: PathLike,
    segments: pd.DataFrame,
    origin_join_cols: Sequence[str] = ("Link", "seg_num"),
    segments_join_cols: Sequence[str] = ("name", "num"),
    drop_cols: Sequence[str] = None,
    tag_col: str = "tag",
) -> pd.DataFrame:
    """Clean data file, given known segments.

    Keep only those records, which can be merged from segments.

    Args:
        origin: Source csv location.
        segments: Data frame containing segment data.
        origin_join_cols: Column names on data, join keys.
        segments_join_cols: Column names on `segments`, join keys.
        tag_col: Column name to merge from segments.

    Returns:
        Cleaned data, with corrected data types, included
        `tag_col`.
    """
    data = pd.read_csv(origin)
    segments_join_cols = list(segments_join_cols)
    origin_join_cols = list(origin_join_cols)
    drop_cols = list(drop_cols) if drop_cols is not None else []
    joined = pd.merge(
        left=segments[[tag_col] + segments_join_cols],
        right=data,
        left_on=segments_join_cols,
        right_on=origin_join_cols,
        how="left",
    )

    return joined.drop(columns=segments_join_cols + list(drop_cols)).convert_dtypes()


def main(
    segments: PathLike,
    segments_out: PathLike,
    data: list[PathLike],
    data_out: list[PathLike],
):
    """Clean and save segments and additional data."""
    seg = clean_segments(segments)
    seg.to_parquet(segments_out)

    for data_, out_ in zip(data, data_out):
        data_i = clean_data(data_, seg)
        data_i.to_parquet(out_)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        help="Additional CSV data file path.",
        nargs="*",
    )
    parser.add_argument(
        "--data-out",
        help="Additional data cleaned output Parquet file path.",
        nargs="*",
    )
    parser.add_argument(
        "segments",
        help="Segment CSV file path.",
        type=Path,
    )
    parser.add_argument(
        "segments_out",
        help="Segment data cleaned output Parquet file path.",
        type=Path,
    )
    args = parser.parse_args()

    if len(args.data) != len(args.data_out):
        parser.error(
            f"Data length {len(args.data)} and data-out length "
            f"{len(args.data_out)} do not match!"
        )
    
    main(**vars(args))
