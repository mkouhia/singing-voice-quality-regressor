"""Download data files for the SVQTD dataset"""

import argparse
import multiprocessing
import queue
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from io import TextIOBase
from multiprocessing import Lock, Process, Queue
from multiprocessing.synchronize import Lock as LockBase
from os import PathLike
from pathlib import Path

import numpy as np
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


class YTAudio(CachedFile):

    """YouTube audio file."""

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
            f"YTAudio({repr(self.tag)}, segment_splits={self.segment_splits}, "
            f"ydl_opts={self.ydl_opts})"
        )

    @property
    def _file_glob(self) -> str:
        return f"{self.tag}.*"

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
    def from_csv(
        cls,
        source_csv: PathLike,
    ) -> list["YTAudio"]:
        """Create group of audio sources from dataframe."""
        data = pd.read_csv(source_csv)[["name", "num", "time_start", "time_end"]]

        grouped = data.groupby("name")
        return [
            cls._grouped_df_to_cls(group_name, grouped.get_group(group_name))
            for group_name in grouped.groups
        ]

    @classmethod
    def _grouped_df_to_cls(cls, name: str, data: pd.DataFrame) -> "YTAudio":
        segments = (
            data.sort_values(by="num", ascending=True)
            .apply(lambda r: (r["num"], (r["time_start"], r["time_end"])), axis=1)
            .tolist()
        )
        return cls(name, dict(segments))


class AudioSegment(CachedFile):

    """Audio segment, from YouTube audio."""

    summary_columns = ["tag", "num", "path"]
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
    def _file_glob(self) -> str:
        return f"{self.source.tag}_{self.id_}.*"

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
            raise UserWarning(
                f"Audio conversion error, command: '{' '.join(command)}'"
            ) from err

        self._dest = dest
        return self.dest

    @classmethod
    def summarize(cls, segments: Iterable["AudioSegment"]) -> pd.DataFrame:
        """Returns summary dataframe of all segments.

        Columns are
        - `tag`: tag for original audio
        - `num`: segment ID number
        - `path`: path to cached file location on disk.
        """
        items = [
            {
                "tag": seg.source.tag if seg.source is not None else np.nan,
                "num": seg.id_,
                "path": str(seg.dest) if seg.dest is not None else np.nan,
            }
            for seg in segments
        ]
        data = pd.DataFrame(items)
        for col in ["tag", "path"]:
            data[col] = data[col].astype("string")
        return data


def get_audio_files(
    segments: PathLike | TextIOBase,
    out_dir: PathLike,
    summary_path: PathLike | TextIOBase = None,
    force: bool = False,
    n_processes: int = 1,
    num_tries: int = 0,
    retry_delay_seconds: int = 0,
):
    """Get audio files from YouTube.

    Args:
        segments: Segment definition csv file or path to such.
        out_dir: Directory, where audio files are stored.
        summary_path: Path or file, where to store summary. If None, do
          not write summary.
        force: Force download all files.
        n_processes: Number of parallel download processes.
        num_tries: Number of times to try a failed download, before
          leaving it as failed.
        retry_delay_seconds: Wait time after a failed download before
          a retry.
    """
    in_queue = Queue()
    summary_lock = Lock()
    for yt_audio in YTAudio.from_csv(segments):
        in_queue.put((yt_audio, {"num_tries": 0, "prev_time": 0.0}))

    if summary_path is not None:
        _write_row(summary_path, "tag", "path", "error_reason")

    processes = []
    for _ in range(n_processes or 1):
        proc = AudioDownloadProcess(
            in_queue,
            out_dir,
            summary_path,
            summary_lock,
            force,
            num_tries,
            retry_delay_seconds,
        )
        processes.append(proc)
        proc.start()

    for proc in processes:
        proc.join()


class AudioDownloadProcess(Process):

    """Process for downloading files in parallel."""

    def __init__(
        self,
        in_queue: Queue,
        out_dir: PathLike,
        summary_path: PathLike | TextIOBase,
        summary_lock: LockBase,
        force: bool,
        num_tries: int,
        retry_delay_seconds: float,
    ) -> None:
        super().__init__(daemon=True)
        self.in_queue = in_queue
        self.out_dir = out_dir
        self.summary_path = summary_path
        self.summary_lock = summary_lock
        self.force = force
        self.num_tries = num_tries
        self.retry_delay_seconds = retry_delay_seconds

    def run(self):
        while True:
            try:
                yt_audio, run_kwargs = self.in_queue.get(block=False)
            except queue.Empty:
                break

            if (
                prev_time := run_kwargs["prev_time"]
            ) > 0.0 and time.time() - prev_time < self.retry_delay_seconds:
                self.in_queue.put((yt_audio, run_kwargs))
                return

            try:
                out_path = yt_audio.get(self.out_dir, self.force)
                run_kwargs["out_path"] = out_path
                self._write_summary_row(yt_audio.tag, **run_kwargs)
            except (youtube_dl.DownloadError, UserWarning) as exc:
                run_kwargs["num_tries"] += 1
                run_kwargs["prev_time"] = time.time()
                run_kwargs["error_reason"] = str(exc)
                if run_kwargs["num_tries"] == self.num_tries:
                    self._write_summary_row(yt_audio.tag, **run_kwargs)
                else:
                    self.in_queue.put((yt_audio, run_kwargs))

    def _write_summary_row(
        self, tag: str, out_path: Path, error_reason: str = "", **kwargs
    ):
        """Write summary row to out_path, with file lock."""
        del kwargs  # unused, only sink for extra arguments
        if self.summary_path is None:
            return
        self.summary_lock.acquire()
        try:
            _write_row(self.summary_path, tag, out_path, error_reason)
        finally:
            self.summary_lock.release()


def _write_row(summary_path: PathLike | TextIOBase, *args):
    """Write content row to path or text io."""
    content = ",".join(str(i) for i in args) + "\n"
    if isinstance(summary_path, TextIOBase):
        summary_path.write(content)
    else:
        Path(summary_path).write_text(content, encoding="utf-8")


def _type_dir_or_create(dest: str) -> Path:
    """Create directory if required, raise if there is a file."""
    if (dest_path := Path(dest)).is_file():
        raise FileExistsError(f"Cannot write to {dest}: file exists in place")

    if not dest_path.exists():
        dest_path.mkdir(parents=True)
    return dest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="commands", dest="command")

    n_cpus = multiprocessing.cpu_count()
    parser_get = subparsers.add_parser(
        "get",
        help="Download audio from YouTube.",
    )
    parser_get.add_argument(
        "--n-processes",
        "-n",
        type=int,
        default=n_cpus,
        help=f"Number of parallel download processes. Default: {n_cpus}.",
    )
    parser_get.add_argument(
        "--summary-path",
        "-s",
        help="Summary csv output location. If not specified, do not produce a summary.",
        type=argparse.FileType("w"),
    )
    parser_get.add_argument(
        "--force", action="store_true", help="Re-download every file."
    )
    parser_get.add_argument(
        "segments",
        help="Segment definition input CSV file location.",
        type=argparse.FileType("r"),
    )
    parser_get.add_argument(
        "out_dir",
        help="Directory for downloaded audio, created if necessary.",
        type=_type_dir_or_create,
    )

    parser_split = subparsers.add_parser(
        "split",
        help="Split audio files into segments.",
    )

    parser_args = parser.parse_args()
    parser_kwargs = vars(parser_args)
    try:
        match parser_kwargs.pop("command"):
            case "get":
                get_audio_files(**parser_kwargs)
            case "split":
                raise NotImplementedError
            case _:
                ...
    finally:
        for item_ in parser_kwargs.values():
            if isinstance(item_, TextIOBase) and item_ not in [sys.stdin, sys.stdout]:
                item_.close()
