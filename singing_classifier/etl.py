"""Download data files for the SVQTD dataset"""

from dataclasses import dataclass
from os import PathLike
from typing import Callable

import pandas as pd


@dataclass(frozen=True, eq=True)
class AudioSegment:
    """Time segment in audio source."""

    num: int
    start_time: float
    end_time: float


@dataclass(frozen=True, eq=True)
class AudioSource:
    """Audio source, with segment information."""

    name: str
    url: str
    segments: frozenset[AudioSegment]


def to_youtube_url(tag: str) -> str:
    """Returns URL from youtube video ID."""
    return "https://www.youtube.com/watch?v=" + tag


def extract_audio_sources(
    source_csv: PathLike,
    url_creator: Callable = to_youtube_url,
    ignore_names: list[str] = None,
) -> list[AudioSource]:
    data = pd.read_csv(source_csv)[["name", "num", "time_start", "time_end"]]
    ignore_names = ignore_names or []

    grouped = data.groupby("name")
    return [
        _grouped_df_to_audiosource(
            name=group_name,
            data=grouped.get_group(group_name),
            url_creator=url_creator,
        )
        for group_name in grouped.groups
        if group_name not in ignore_names
    ]


def _grouped_df_to_audiosource(
    name: str, data: pd.DataFrame, url_creator: Callable
) -> AudioSource:
    segments = (
        data.sort_values(by="num", ascending=True)
        .apply(_row_to_segment, axis=1)
        .tolist()
    )
    return AudioSource(name=name, url=url_creator(name), segments=frozenset(segments))


def _row_to_segment(row):
    return AudioSegment(row["num"], row["time_start"], row["time_end"])

