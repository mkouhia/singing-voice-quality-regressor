"""Test data extraction, transformation and loading."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from scipy.io import wavfile

from singing_classifier.etl import AudioSegment, CachedFile, YTAudio, get_audio_files


def _create_sine(
    freq: float,
    length: float,
    sample_rate: int,
    start_phase: int = 0.0,
) -> np.ndarray[np.int16]:
    """Returns sine wave signal in 16 bit int format.

    Credit: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html
    """
    time = np.linspace(0.0, length, int(sample_rate * length))
    amplitude = np.iinfo(np.int16).max
    return (amplitude * np.sin(2 * np.pi * freq * time + start_phase)).astype(np.int16)


@pytest.fixture(name="audio_file", scope="module")
def fx_test_audio(tmp_path_factory) -> Path:
    """Create test audio file on disk, return its location."""
    sample_rate = 44100
    data = _create_sine(freq=440.0, length=2.0, sample_rate=sample_rate)

    loc = tmp_path_factory.mktemp("data") / "test_audio.wav"
    with open(loc, "wb") as file_:
        wavfile.write(file_, sample_rate, data)

    return loc


@pytest.fixture(name="segments_path")
def fx_segments_path(tmp_path):
    """Returns path for test segment definition file."""
    content = """name,num,time_start,time_end
BaW_jenozKc,0,0.0,2.4
BaW_jenozKc,1,2.8,7.0
pRnPUCeL76M,0,0.4,2.2
"""
    csv_loc = tmp_path / "test.csv"
    csv_loc.write_text(content)
    return csv_loc


class CacheImpl(CachedFile):

    """Test implementation for testing shared cache methods."""

    @property
    def _file_glob(self):
        return "subdir/test_id.*"

    def get(self, cache_dir: Path = None, force=False) -> Path:
        super().get(cache_dir, force)
        if not force and self.is_cached:
            return self.dest

        self._dest = self.cache_dir / "subdir" / "test_id.1"
        self._dest.mkdir(parents=True)
        self._dest.touch()
        return self._dest


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

    @pytest.fixture(name="expected_segments")
    def fx_expected_segments(self) -> set[YTAudio]:
        return {
            YTAudio("BaW_jenozKc", {0: (0.0, 2.4), 1: (2.8, 7.0)}),
            YTAudio("pRnPUCeL76M", {0: (0.4, 2.2)}),
        }

    def test_from_csv(self, segments_path: Path, expected_segments: set[YTAudio]):
        """Convert CSV file into list of audio sources."""
        received = YTAudio.from_csv(segments_path)
        assert set(received) == expected_segments

    def test_from_csv_file(self, segments_path: Path, expected_segments: set[YTAudio]):
        """Test reading open file buffer into segments."""
        with open(segments_path, "r", encoding="utf-8") as file_:
            received = YTAudio.from_csv(file_)
        assert set(received) == expected_segments


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
            ("aa", 0, "tmp/aa_0.m4a"),
            ("aa", 1, "tmp/aa_1.m4a"),
            ("ab", 0, "tmp/ab_0.m4a"),
        ]
        data = [dict(zip(AudioSegment.summary_columns, i)) for i in vals]
        expected = pd.DataFrame(data)

        for col in ["tag", "path"]:
            expected[col] = expected[col].astype("string")

        assert_frame_equal(received, expected)


@pytest.mark.integration_test
@pytest.mark.parametrize("n_processes", [1, 3])
def test_get_audio_files(tmp_path: Path, segments_path: Path, n_processes: int):
    """Get flow works correctly"""
    out_dir = tmp_path / "out"
    summary_path = tmp_path / "summary.csv"
    get_audio_files(
        segments_path,
        out_dir,
        summary_path,
        n_processes=n_processes,
        num_tries=2,
        retry_delay_seconds=1.5,
    )
