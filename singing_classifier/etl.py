"""Download data files for the SVQTD dataset"""

import argparse
import itertools
import multiprocessing
import re
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from functools import partial
from io import TextIOBase
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
import youtube_dl

from .processing import perform_multiprocess


class CachedFile(ABC):

    """File, that may be cached logally, or downloaded/produced."""

    _summary_attrs: list[tuple[str, str, str]] = []
    """List of specifiers, from which summary dataframe is produced.

    Each tuple is (display name, attribute, data type).
    """

    def __init__(self):
        self._dest: Path = None
        self._cache_dir: Path = None
        self._get_exception_msg: str = ""

    @property
    @abstractmethod
    def _file_pattern(self) -> str:
        """Returns regex expression with which to match file name in cache dir."""

    @abstractmethod
    def get(self, cache_dir: PathLike = None, force=False) -> Path:
        """Get file, store to cache_dir.

        If file is cached, return cached file location.
        """
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None

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
        p = re.compile(self._file_pattern)
        for candidate_path in self.cache_dir.glob("**/*"):
            if p.match(str(candidate_path.relative_to(self.cache_dir))):
                return candidate_path
        return None

    @dest.setter
    def dest(self, value: Path):
        """Set cached file location on disk."""
        self._dest = value

    @property
    def is_cached(self) -> bool:
        """Downloaded content exists on disk."""
        dest_ = self.dest
        return dest_ is not None and dest_.is_file()

    @classmethod
    @abstractmethod
    def summarize(cls, items: Iterable["CachedFile"]) -> pd.DataFrame:
        """Returns summary dataframe of all cached files."""
        records = (
            {
                att[0]: getattr(item, att[1])
                if getattr(item, att[1]) is not None
                else np.nan
                for att in cls._summary_attrs
            }
            for item in items
        )
        data = pd.DataFrame(records, columns=[att[0] for att in cls._summary_attrs])
        for att in cls._summary_attrs:
            data[att[0]] = data[att[0]].astype(att[2])
        return data


class YTAudio(CachedFile):

    """YouTube audio file."""

    _summary_attrs = [
        ("tag", "tag", "string"),
        ("path", "dest", "string"),
        ("message", "_get_exception_msg", "string"),
    ]

    def __init__(
        self,
        tag: str,
        segment_splits: dict[int, tuple[float, float]] = None,
        ydl_opts=None,
    ):
        """Create descriptor of YouTube audio file

        Args:
            tag: YouTube video tag, e.g. in an url like
              https://www.youtube.com/watch?v=..., tag is the value of
              the url parameter `v`.
            segment_splits: Description of how to split file into audio
              segments. This is employed by `self.form_audio_segments`.
            ydl_opts: Additional options to youtube-dl.
        """
        super().__init__()
        self.tag = tag
        self.segment_splits = segment_splits or {}
        self.ydl_opts = ydl_opts or {}

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, YTAudio):
            return False

        return (
            self.tag == __o.tag
            and self.segment_splits == __o.segment_splits
            and self.ydl_opts == __o.ydl_opts
        )

    def __hash__(self) -> int:
        return hash(
            (self.tag, tuple(self.segment_splits.items()), tuple(self.ydl_opts.items()))
        )

    def __repr__(self) -> str:
        return (
            f"YTAudio({repr(self.tag)}, "
            f"segment_splits={self.segment_splits}, "
            f"_cache_dir={self._cache_dir}, "
            f"_dest={self._dest}, "
            f"ydl_opts={self.ydl_opts})"
        )

    @property
    def _file_pattern(self) -> str:
        return rf"{self.tag}\.(?!part$)[^.]*$"

    @property
    def url(self) -> str:
        """Returns URL from youtube video ID."""
        return f"https://www.youtube.com/watch?v={self.tag}"

    def get(self, cache_dir: PathLike = None, force=False) -> Path:
        """Download audio from YouTube.

        Attempt to serve content from cache, if present. File
        location is stored in `self.dest`.

        Args:
            cache_dir: Directory, where cached content is stored.
            force: Download even if file is already found in cache.

        Returns:
            File location for the downloaded file.

        Raises:
            youtube_dl.DownloadError on failed download.
            UserWarning on other download error.
        """
        super().get(cache_dir, force)
        if not force and self.is_cached:
            return self.dest

        # Ensure that competing files are not in download location
        for disk_file in self.cache_dir.glob(f"{self.tag}.*"):
            disk_file.unlink()

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(self.cache_dir.absolute()) + "/%(id)s.%(ext)s",
            # TODO Examine the need for running postprocessor at this stage
            # 'postprocessors': [{
            #     'key': 'FFmpegExtractAudio',
            #     'preferredcodec': 'best',
            # }],
        } | self.ydl_opts
        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl_return_code = ydl.download([self.url])
        except youtube_dl.DownloadError as exc:
            self._get_exception_msg = str(exc)
            raise UserWarning(
                f"Download failed on {self.url}, {ydl_opts=}; {exc}"
            ) from exc

        if ydl_return_code == 0:
            matching_files = list(self.cache_dir.glob(f"{self.tag}.*"))
            assert len(matching_files) == 1
            file_name = matching_files[0]
        else:
            raise UserWarning("Download failed")

        self._dest = self.cache_dir / file_name
        return self.dest

    def form_segments(self, **segment_kwargs) -> list["AudioSegment"]:
        """Returns list of AudioSegment objects."""
        return [
            AudioSegment(i, *split, source=self, **segment_kwargs)
            for i, split in self.segment_splits.items()
        ]

    @classmethod
    def from_parquet(
        cls,
        source_csv: PathLike | TextIOBase,
        cols: Sequence[str] = ("tag", "num", "time_start", "time_end"),
    ) -> list["YTAudio"]:
        """Create group of audio sources from dataframe."""
        data = pd.read_parquet(source_csv)[list(cols)]

        grouped = data.groupby(cols[0])
        return [
            cls._grouped_df_to_cls(group_name, grouped.get_group(group_name), cols)
            for group_name in grouped.groups
        ]

    @classmethod
    def _grouped_df_to_cls(
        cls, name: str, data: pd.DataFrame, cols: Sequence[str]
    ) -> "YTAudio":
        segments = (
            data.drop(columns=[cols[0]], errors="ignore")
            .sort_values(by=cols[1], ascending=True)
            .apply(lambda r: (int(r[cols[1]]), (r[cols[2]], r[cols[3]])), axis=1)
            .tolist()
        )
        return cls(name, dict(segments))

    @classmethod
    def summarize(cls, items: Iterable["YTAudio"]) -> pd.DataFrame:
        """Returns summary dataframe of all downloaded audio files.

        Columns are defined in
        - `tag`: Tag for original audio.
        - `path`: Path to cached file location on disk.
        - `message`: Possible error message.
        """
        return super().summarize(items)


class AudioSegment(CachedFile):

    """Audio segment, from YouTube audio."""

    _summary_attrs = [
        ("tag", "tag", "string"),
        ("num", "id_", "Int64"),
        ("path", "dest", "string"),
        ("message", "_get_exception_msg", "string"),
    ]

    formats = ["copy", "wav", "flac", "opus", "ogg"]

    def __init__(
        self,
        id_: int,
        time_start: float,
        time_end: float,
        /,
        source: YTAudio = None,
        *,
        sample_rate: int = None,
        format_: str = "copy",
    ):
        """Initialize audio segment.

        Args:
            id_ (int): _description_
            time_start (float): _description_
            time_end (float): _description_
            source (YTAudio, optional): _description_. Defaults to None.
            sample_rate: Output audio file sample rate. If None, do not
              resample when cropping the audio. If format is 'opus',
              sample_rate must be 48000 or None.
            format_: Output file format. Must be one of 'copy', 'wav',
              'flac', 'opus' or 'ogg'. If 'copy', use the same encoding
              as in the source file.

        Raises:
            ValueError if format is not one of supported values, or
            sample_rate is incorrect on opus format.
        """
        super().__init__()
        self.id_ = id_
        self.time_start = time_start
        self.time_end = time_end
        self.source = source

        self.format = format_ or "copy"

        self.ffmpeg_extra = ["-ac", "1"]
        if sample_rate is not None:
            self.ffmpeg_extra.extend(["-ar", str(sample_rate)])

        if format_ not in self.formats:
            raise ValueError(f"Unsupported format: {format_}")
        if format_ == "opus" and sample_rate not in [48000, None]:
            raise ValueError("For opus format, sample_rate must be 48000 or None.")

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, AudioSegment):
            return False

        return (
            self.id_ == __o.id_
            and self.time_start == __o.time_start
            and self.time_end == __o.time_end
            and self.source == __o.source
            and self.format == __o.format
            and self.ffmpeg_extra == __o.ffmpeg_extra
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.id_,
                self.time_start,
                self.time_end,
                self.source,
                self.format,
                tuple(self.ffmpeg_extra),
            )
        )

    @property
    def _file_pattern(self) -> str:
        return rf"{self.source.tag}_{self.id_}\..*"

    def get(self, cache_dir: PathLike = None, force=False) -> Path:
        """Split audio segment from source file.

        Attempt to serve content from cache, if present. File
        location is stored in `self.dest`.

        Args:
            cache_dir: Directory, where cached content is stored.
            force: Download even if file is already found in cache.

        Returns:
            File location for the split file.

        Raises:
            UserWarning if source is not set or cached,
              or on conversion failure.
        """
        super().get(cache_dir, force)
        if not force and self.is_cached:
            return self.dest

        if self.source is None:
            raise UserWarning("Audio source is not set.")
        if not self.source.is_cached:
            raise UserWarning("Audio source is not cached.")

        source_path = self.source.dest
        suffix = source_path.suffix if self.format == "copy" else f".{self.format}"
        dest = self.cache_dir / f"{self.source.tag}_{self.id_}{suffix}"

        call_args = [
            "-y",
            "-vn",
            "-i",
            str(source_path.absolute()),
            "-ss",
            f"{self.time_start:.3f}",
            "-to",
            f"{self.time_end:.3f}",
        ]

        if self.format == "ogg":
            acodec_opts = ["-acodec", "libvorbis"]
        elif self.format == "opus":
            acodec_opts = ["-acodec", "libopus"]
        elif self.format == "wav":
            acodec_opts = []
        else:
            acodec_opts = ["-acodec", self.format]

        command = (
            ["ffmpeg"]
            + call_args
            + acodec_opts
            + self.ffmpeg_extra
            + [str(dest.absolute())]
        )
        try:
            subprocess.run(command, check=True)
        except subprocess.SubprocessError as err:
            self._get_exception_msg = str(err)
            raise UserWarning(
                f"Audio conversion error, command: '{' '.join(command)}'"
            ) from err

        self._dest = dest
        return self.dest

    @property
    def tag(self):
        """Tag of source audio."""
        return self.source.tag if self.source is not None else None

    @classmethod
    def summarize(cls, items: Iterable["AudioSegment"]) -> pd.DataFrame:
        """Returns summary dataframe of all audio segments.

        Columns are defined in
        - `tag`: Tag for original audio.
        - `num`: Segment ID number.
        - `path`: Path to cached file location on disk.
        - `message`: Possible error message.
        """
        return super().summarize(items)


def get_and_split(
    segments: Path,
    raw_dir: Path,
    split_dir: Path,
    download_summary: Path = None,
    split_summary: Path = None,
    n_processes: int = 1,
    num_tries: int = 1,
    retry_delay_seconds: float = 5.0,
    segment_kwargs: dict = None,
    force_download: bool = False,
    force_split: bool = False,
):
    """Get audio files from YouTube.

    Args:
        segments: Path to segment definition Parquet file.
        raw_dir: Directory, where downloaded audio files are stored.
        split_dir: Directory, where split audio files are stored.
        download_summary: Path, where to write download summary Parquet
          file. If None, do not write a summary.
        split_summary: Path, where to write split summary Parquet file.
          If None, do not write a summary.
        n_processes: Number of parallel download processes.
        num_tries: Number of times to try a failed download, before
          leaving it as failed.
        retry_delay_seconds: Wait time after a failed download before
          a retry.
        segment_kwargs: Keyword arguments that are directed to
          `YTAudio.form_segments`.
        force_download: Force download all files.
        force_split: Force splitting of files.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    audio_defs = YTAudio.from_parquet(segments)
    dl_results = perform_multiprocess(
        partial(_get_audio, cache_dir=raw_dir, force=force_download),
        audio_defs,
        n_processes=n_processes,
        num_tries=num_tries,
        retry_delay_seconds=retry_delay_seconds,
        attach_stats=True,
    )

    audio_defs = [i[0] for i in dl_results]
    if download_summary is not None:
        dl_summary = YTAudio.summarize(audio_defs)
        dl_summary.to_parquet(download_summary)

    segments = list(
        itertools.chain.from_iterable(
            [
                yt_audio.form_segments(**(segment_kwargs or {}))
                for yt_audio in audio_defs
            ]
        )
    )
    split_results = perform_multiprocess(
        partial(_get_segment, cache_dir=split_dir, force=force_split),
        segments,
        n_processes=n_processes,
        num_tries=1,
    )

    segments = (i[0] for i in split_results)
    if split_summary is not None:
        summary_data = AudioSegment.summarize(segments)
        summary_data.to_parquet(split_summary)


def _get_audio(yt_audio, *args, **kwargs):
    return yt_audio.get(*args, **kwargs)


def _get_segment(segment, *args, **kwargs):
    return segment.get(*args, **kwargs)


def _type_dir_or_create(dest: str) -> Path:
    """Create directory if required, raise if there is a file."""
    if (dest_path := Path(dest)).is_file():
        raise FileExistsError(f"Cannot write to {dest}: file exists in place")

    if not dest_path.exists():
        dest_path.mkdir(parents=True)
    return dest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    n_cpus = multiprocessing.cpu_count()
    parser.add_argument(
        "--download-summary",
        help="Download summary Parquet output location. If not specified, do not produce a summary.",
        type=Path,
    )
    parser.add_argument(
        "--split-summary",
        help="Split summary Parquet output location. If not specified, do not produce a summary.",
        type=Path,
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        default=n_cpus,
        help=f"Number of parallel download processes. Default: {n_cpus}.",
    )
    parser.add_argument(
        "--num-tries",
        type=int,
        default=3,
        help="Number of times to try a download, before leaving it as failed. Default: 3.",
    )
    parser.add_argument(
        "--retry_delay_seconds",
        type=float,
        default=10.0,
        help="Minimum wait time in seconds after a failed download, before a retry on the same item. Default: 5",
    )
    parser.add_argument(
        "--force-download", action="store_true", help="Re-download every file."
    )
    parser.add_argument(
        "--force-split", action="store_true", help="Re-split every file."
    )
    parser.add_argument(
        "segments",
        help="Segment definition input Parquet file location.",
        type=Path,
    )
    parser.add_argument(
        "raw_dir",
        help="Directory for downloaded audio, created if necessary.",
        type=_type_dir_or_create,
    )
    parser.add_argument(
        "split_dir",
        help="Directory for split audio, created if necessary.",
        type=_type_dir_or_create,
    )

    parser_args = parser.parse_args()
    parser_kwargs = vars(parser_args)
    get_and_split(**parser_kwargs)
