"""Test cleanup methods."""

from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from singing_classifier.clean import (
    clean_data,
    clean_segments,
    ensure_unique,
    extract_id,
)


@pytest.fixture(name="segments_csv", scope="module")
def fx_segments_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create test segments file on disk, return its location."""
    content = """name,num,time_start,time_end
test,0,0.0,0.0
1-YOrUq0s7w,0,24.466,32.272
1-YOrUq0s7w,1,33.193,44.001
VB8HpZ7Aw2E&list=PLpEpmwpNk-6dBCxujxeIo9fZGZfxLzcTH&index=56&t=0s,0,4.763,9.287
"""
    loc = tmp_path_factory.mktemp("data") / "segments.csv"
    loc.write_text(content, encoding="utf-8")
    return loc


@pytest.fixture(name="segments")
def fx_segments() -> pd.DataFrame:
    """Returns example of cleaned segment file."""
    return pd.DataFrame(
        {
            "tag": ["1-YOrUq0s7w", "1-YOrUq0s7w", "VB8HpZ7Aw2E"],
            "num": [0, 1, 0],
            "name": [
                "1-YOrUq0s7w",
                "1-YOrUq0s7w",
                "VB8HpZ7Aw2E&list=PLpEpmwpNk-6dBCxujxeIo9fZGZfxLzcTH&index=56&t=0s",
            ],
            "time_start": [24.466, 33.193, 4.763],
            "time_end": [32.272, 44.001, 9.287],
        },
        index=[1, 2, 3],
    )


@pytest.fixture(name="train_csv", scope="module")
def fx_train_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create test data file on disk, return its location."""
    content = """Link,seg_num,Head,Chest,Open,Breathy,Vibrato,Front,Back,song
1-YOrUq0s7w,0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,Che_gelida_manina
1-YOrUq0s7w,1,1.0,1.0,0.0,1.0,1.0,0.0,0.0,Che_gelida_manina
VB8HpZ7Aw2E&list=PLpEpmwpNk-6dBCxujxeIo9fZGZfxLzcTH&index=56&t=0s,0,2.0,1.0,0.0,0.0,1.0,0.0,1.0,Nessun_Dorma
"""
    loc = tmp_path_factory.mktemp("data") / "train.csv"
    loc.write_text(content, encoding="utf-8")
    return loc


@pytest.fixture(name="train_data")
def fx_train_data() -> pd.DataFrame:
    """Returns example of cleaned data file."""
    data = pd.DataFrame(
        {
            "tag": ["1-YOrUq0s7w", "1-YOrUq0s7w", "VB8HpZ7Aw2E"],
            "Link": [
                "1-YOrUq0s7w",
                "1-YOrUq0s7w",
                "VB8HpZ7Aw2E&list=PLpEpmwpNk-6dBCxujxeIo9fZGZfxLzcTH&index=56&t=0s",
            ],
            "seg_num": [0, 1, 0],
            "Head": [1, 1, 2],
            "Chest": [0, 1, 1],
            "Open": [0, 0, 0],
            "Breathy": [0, 1, 0],
            "Vibrato": [1, 1, 1],
            "Front": [0, 0, 0],
            "Back": [0, 0, 1],
            "song": ["Che_gelida_manina", "Che_gelida_manina", "Nessun_Dorma"],
        }
    )
    for col in data.columns:
        dtype = "string" if col in ["tag", "Link", "song"] else "Int64"
        data[col] = data[col].astype(dtype)
    return data


def test_extract_id():
    """Only first url parameter is left."""
    content = "abc,abd&list=xyz&t=0,aaf&list=xyy,abc&"
    series = pd.Series(content.split(","))

    expected = pd.Series(["abc", "abd", "aaf", "abc"])
    received = extract_id(series)

    assert_series_equal(received, expected)


def test_ensure_unique():
    """Dataframe contains only unique values."""
    data = pd.DataFrame(
        {
            "tag": ["abc", "abd", "aaf", "abc"],
            "num": [0, 0, 4, 1],
            "name": ["abc", "abd&list=xyz&t=0", "aaf&list=xyy", "abc="],
            "time_start": [0.0, 1.700, 0.0, 20.7],
            "time_end": [13.720, 12.30, 1.0, 34.5],
        }
    )
    received = ensure_unique(data, columns=["tag", "num"])
    
    print(received)
    print(data)
    
    assert_frame_equal(received, data)


def test_ensure_unique_full_duplicates():
    """Duplicate rows are removed iff whole rows are equal."""
    data = pd.DataFrame(
        {
            "tag": ["abc", "abd", "aaf", "abc"],
            "num": [0, 0, 4, 0],
            "time_start": [0.0, 1.700, 0.0, 0.0],
            "time_end": [13.720, 12.30, 1.0, 13.720],
        }
    )
    received = ensure_unique(data, columns=["tag", "num"])
    expected = data.drop(data.tail(1).index)
    
    assert_frame_equal(received, expected)


def test_ensure_unique_fail(segments: pd.DataFrame):
    """On non-unique values, UserWarning is raised."""
    data = pd.DataFrame(
        {
            "tag": ["abc", "abd", "aaf", "abc"],
            "num": [0, 0, 4, 0],
            "time_start": [0.0, 1.700, 0.0, 0.1],
            "time_end": [13.720, 12.30, 1.0, 13.720],
        }
    )
    with pytest.raises(UserWarning):
        ensure_unique(data, columns=["tag", "num"])


def test_clean_segments(segments_csv: Path, segments: pd.DataFrame):
    """Csv is read and dataframe returned."""
    received = clean_segments(segments_csv, drop_tags=["test"])
    assert_frame_equal(received, segments)


def test_clean_segments_incorrect_time(segments_csv: Path, segments: pd.DataFrame):
    """Entries with time_end < time_start are dropped."""
    with open(segments_csv, "a", encoding="utf-8") as file_:
        file_.write("1-YOrUq0s7w,2,50.0,40.0\n")
    received = clean_segments(segments_csv, drop_tags=["test"])

    assert_frame_equal(received, segments)


def test_clean_data(train_csv: Path, segments: pd.DataFrame, train_data: pd.DataFrame):
    """Tags are joined from segments, data types corrected."""
    received = clean_data(origin=train_csv, segments=segments)
    assert_frame_equal(received, train_data)
