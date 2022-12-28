"""Download data files for the SVQTD dataset"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from collections.abc import Callable

import pandas as pd
import youtube_dl


class CachedFile(ABC):

    """File, that may be cached logally, or downloaded/produced."""

    def __init__(self):
        self._dest: Path = None
        self._cache_dir: Path = None

    @property
    @abstractmethod
    def _file_glob(self) -> str:
        """Returns `glob` expression with which to find file in cache dir"""

    @abstractmethod
    def get(self, cache_dir: Path = None, force=False) -> Path:
        """Get file, store to cache_dir.

        If file is cached, return cached file location.
        """
        self._cache_dir = cache_dir

    def _get_cache_dir(self) -> Path:
        """Get dir from user config"""
        raise NotImplementedError

    @property
    def cache_dir(self) -> Path:
        """Cache directory location on disk."""
        return self._cache_dir or self._get_cache_dir()

    @cache_dir.setter
    def cache_dir(self, value: PathLike):
        self._cache_dir = Path(value)

    @property
    def dest(self) -> Path | None:
        """Cached file location on disk, or None if not in cache."""
        if self._dest is not None:
            return self._dest
        if self.cache_dir is None:
            return None
        return next(self.cache_dir.glob(self._file_glob), None)

    @dest.setter
    def dest(self, value: Path):
        """Set cached file location on disk."""
        self._dest = value

    @property
    def is_cached(self) -> bool:
        """Downloaded content exists on disk."""
        dest_ = self.dest
        return dest_ is not None and dest_.is_file()


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

    def get_youtube(self, dest_dir: PathLike, force=False) -> Path:
        """Download video content from YouTube.

        Args:
            url: URL to YouTube video, in format
                'https://www.youtube.com/watch?v=...'
            dest_dir: Destination directory location on disk.
            force: Force download. If false, use already downloaded content.

        Returns:
            Location of downloaded file on disk.
        """
        dest_dir_path = Path(dest_dir)
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(dest_dir_path.absolute()) + "/%(id)s.%(ext)s",
            # 'postprocessors': [{
            #     'key': 'FFmpegExtractAudio',
            #     'preferredcodec': 'mp3',
            #     'preferredquality': '192',
            # }],
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl_return_code = ydl.download([self.url])

        if ydl_return_code == 0:
            matching_files = list(dest_dir_path.glob(f"{self.name}.*"))
            assert len(matching_files) == 1
            # TODO set downloaded location to object attributes
            return matching_files[0]
        raise NotImplementedError


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


if __name__ == "__main__":
    ...
