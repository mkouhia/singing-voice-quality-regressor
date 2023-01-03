"""Test data extraction, transformation and loading."""

from collections.abc import Iterable
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from singing_classifier.etl import AudioSegment, CachedFile, YTAudio, get_and_split
from .util import create_wavfile_at


@pytest.fixture(name="audio_file", scope="module")
def fx_test_audio(tmp_path_factory) -> Path:
    """Create test audio file on disk, return its location."""
    sample_rate = 44100
    loc = tmp_path_factory.mktemp("data") / "test_audio.wav"
    create_wavfile_at(loc, freq=440.0, length=2.0, sample_rate=sample_rate)
    return loc


@pytest.fixture(name="segments_path")
def fx_segments_path(tmp_path):
    """Returns path for test segment definition file."""
    content = """tag,num,time_start,time_end
BaW_jenozKc,0,0.0,2.4
BaW_jenozKc,1,2.8,7.0
pRnPUCeL76M,0,0.4,2.2
"""
    csv_io = StringIO(content)
    data = pd.read_csv(csv_io)

    dest = tmp_path / "segments.parquet"
    data.to_parquet(dest, index=False)

    return dest


class CacheImpl(CachedFile):

    """Test implementation for testing shared cache methods."""

    @property
    def _file_pattern(self):
        return r"subdir/test_id\..*$"

    def get(self, cache_dir: Path = None, force=False) -> Path:
        super().get(cache_dir, force)
        if not force and self.is_cached:
            return self.dest

        self._dest = self.cache_dir / "subdir" / "test_id.1"
        self._dest.mkdir(parents=True)
        self._dest.touch()
        return self._dest

    @classmethod
    def summarize(cls, items: Iterable["CacheImpl"]) -> pd.DataFrame:
        raise NotImplementedError


class TestCache:

    """Test file caching."""

    @pytest.fixture(name="cached_file")
    def fx_cached_file(self) -> CachedFile:
        """Returns test cached file object."""
        return CacheImpl()

    def test_is_cached(self, tmp_path: Path, cached_file: CachedFile):
        """If file exists on disk, returns True."""
        cached_file.dest = tmp_path / "file.ext"
        cached_file.dest.touch()

        assert cached_file.is_cached

    def test_cache_found(self, tmp_path: Path, cached_file: CachedFile):
        """Cache location is found, even though dest is not set."""
        cached_file.cache_dir = tmp_path
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "test_id.xyz").touch()

        assert cached_file.is_cached

    def test_not_cached(self, tmp_path: Path, cached_file: CachedFile):
        """If not-None dest does not exists on disk, returns False."""
        cached_file.dest = tmp_path / "file.ext"
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "test_id.xyz").touch()

        assert not cached_file.is_cached

    def test_get_cached(self, tmp_path: Path, cached_file: CachedFile):
        """If file is cached, return cached."""
        cached_file.dest = tmp_path / "file.ext"
        cached_file.dest.touch()

        received = cached_file.get()
        assert received == cached_file.dest

    def test_get(self, tmp_path: Path, cached_file: CachedFile):
        """If file is not cached, perform get action."""
        received = cached_file.get(cache_dir=tmp_path)
        assert received == tmp_path / "subdir" / "test_id.1"
        assert cached_file.dest == received

    def test_get_force(self, tmp_path: Path, cached_file: CachedFile):
        """Even though cached, if force, perform get action."""
        cached_file.dest = tmp_path / "file.ext"
        cached_file.dest.touch()

        assert cached_file.is_cached

        received = cached_file.get(cache_dir=tmp_path, force=True)
        assert received == tmp_path / "subdir" / "test_id.1"


class TestYTAudio:

    """Test audio extraction and segment forming."""

    @pytest.mark.integration_test
    def test_get(self, tmp_path: Path):
        """File is downloaded from YouTube."""
        tag = "BaW_jenozKc"
        source = YTAudio(tag, ydl_opts={"cachedir": False})

        received = source.get(cache_dir=tmp_path)
        assert received == tmp_path / f"{tag}.m4a"
        assert received.is_file()

    @pytest.mark.parametrize(
        "sample_rate,format_",
        [
            (None, None),
            (28000, None),
            (None, "copy"),
            (14000, "wav"),
        ],
    )
    def test_form_segments(self, sample_rate: int, format_: str):
        """Segment objects are created."""
        splits = {7: (0.8, 22.788), 10: (35.401, 45.023)}
        source = YTAudio("test", splits)

        kwargs = {"sample_rate": sample_rate, "format_": format_}
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        received = source.form_segments(**kwargs)
        expected = [
            AudioSegment(7, 0.8, 22.788, source, **kwargs),
            AudioSegment(10, 35.401, 45.023, source, **kwargs),
        ]

        assert received == expected

    def test_from_parquet(self, segments_path: Path):
        """Convert CSV file into list of audio sources."""
        received = YTAudio.from_parquet(segments_path)
        expected = {
            YTAudio("BaW_jenozKc", {0: (0.0, 2.4), 1: (2.8, 7.0)}),
            YTAudio("pRnPUCeL76M", {0: (0.4, 2.2)}),
        }
        assert set(received) == expected


class TestSegment:

    """Test audio splitting."""

    @pytest.mark.parametrize(
        "sample_rate,format_",
        [
            (16000, "wav"),
            (None, "copy"),
            (44100, "flac"),
            (None, "opus"),
            (48000, "opus"),
            (32000, "ogg"),
        ],
    )
    def test_get(
        self, audio_file: Path, tmp_path: Path, sample_rate: str, format_: str
    ):
        """File is cropped and resampled from source."""
        segment = AudioSegment(0, 0.1, 1.0, sample_rate=sample_rate, format_=format_)
        segment.source = YTAudio(None)
        segment.source.dest = audio_file

        segment_dest = segment.get(cache_dir=tmp_path)
        assert segment.dest.is_file()
        assert segment.dest == segment_dest

    def test_summarize(self):
        """Summary dataframe is created."""
        source_aa = YTAudio("aa")
        segments = [AudioSegment(i, None, None) for i in [0, 1, 0]]
        segments[0].source = source_aa
        segments[1].source = source_aa
        segments[2].source = YTAudio("ab")

        for seg in segments:
            seg.dest = Path("tmp", f"{seg.source.tag}_{seg.id_}.m4a")

        received = AudioSegment.summarize(segments)

        vals = [
            ("aa", 0, "tmp/aa_0.m4a", ""),
            ("aa", 1, "tmp/aa_1.m4a", ""),
            ("ab", 0, "tmp/ab_0.m4a", ""),
        ]

        # pylint: disable=protected-access
        col_names = [i[0] for i in AudioSegment._summary_attrs]
        data = [dict(zip(col_names, i)) for i in vals]
        expected = pd.DataFrame(data)

        for col in ["tag", "path", "message"]:
            expected[col] = expected[col].astype("string")
        expected["num"] = expected["num"].astype("Int64")

        assert_frame_equal(received, expected)


@pytest.mark.integration_test
def test_get_and_split(tmp_path: Path, segments_path: Path):
    """Download and split audio files."""
    out_dir = tmp_path / "out"
    raw_dir = tmp_path / "raw"
    download_summary = tmp_path / "dl_summary.csv"
    split_summary = tmp_path / "split_summary.csv"
    get_and_split(
        segments=segments_path,
        raw_dir=raw_dir,
        split_dir=out_dir,
        download_summary=download_summary,
        split_summary=split_summary,
        segment_kwargs={"format_": "opus"},
        n_processes=1,
        num_tries=3,
        retry_delay_seconds=3,
    )

    expected_raw_files = {raw_dir / i for i in ["BaW_jenozKc.m4a", "pRnPUCeL76M.webm"]}
    expected_splits = {
        out_dir / i
        for i in [
            "BaW_jenozKc_0.opus",
            "BaW_jenozKc_1.opus",
            "pRnPUCeL76M_0.opus",
        ]
    }

    expected_dl_content = f"""tag,path,message
BaW_jenozKc,{raw_dir}/BaW_jenozKc.m4a,
pRnPUCeL76M,{raw_dir}/pRnPUCeL76M.webm,
"""
    csv_io = StringIO(expected_dl_content)
    expected_dl_data = pd.read_csv(csv_io).fillna("").convert_dtypes()

    expected_split_content = f"""tag,num,path,message
BaW_jenozKc,0,{out_dir}/BaW_jenozKc_0.opus,
BaW_jenozKc,1,{out_dir}/BaW_jenozKc_1.opus,
pRnPUCeL76M,0,{out_dir}/pRnPUCeL76M_0.opus,
"""
    csv_io = StringIO(expected_split_content)
    expected_split_data = pd.read_csv(csv_io).fillna("").convert_dtypes()

    received_dl_summary = (
        pd.read_parquet(download_summary).reset_index(drop=True).sort_values(by=["tag"])
    )
    received_split_summary = (
        pd.read_parquet(split_summary)
        .reset_index(drop=True)
        .sort_values(by=["tag", "num"])
    )

    assert set(raw_dir.glob("*")) == expected_raw_files
    assert set(out_dir.glob("*")) == expected_splits
    assert_frame_equal(received_dl_summary, expected_dl_data)
    assert_frame_equal(received_split_summary, expected_split_data)
