"""Test data extraction, transformation and loading."""

from pathlib import Path

import pytest

from singing_classifier.etl import (
    AudioSegment,
    AudioSource,
    CachedFile,
    extract_audio_sources,
    to_youtube_url,
)


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


def test_to_youtube_url():
    """Add tag at the end."""
    tag = "abc"
    expected = "https://www.youtube.com/watch?v=" + tag
    assert to_youtube_url(tag) == expected


def test_extract_audio_sources(tmp_path: Path):
    """Convert CSV file into list of audio sources."""
    content = """name,num,time_start,time_end
test,0,0.0,0.0
0HVjTPqe5I0,0,8.893,17.323
0HVjTPqe5I0,1,18.053,28.832
0HVjTPqe5I0,2,39.12,53.974
0QzJ86rMHzQ,0,7.464,25.834"""
    csv_loc = tmp_path / "test.csv"
    csv_loc.write_text(content)

    def url_creator(tag):
        return "abc" + tag

    expected = {
        AudioSource(
            "0HVjTPqe5I0",
            url_creator("0HVjTPqe5I0"),
            frozenset(
                [
                    AudioSegment(0, 8.893, 17.323),
                    AudioSegment(1, 18.053, 28.832),
                    AudioSegment(2, 39.12, 53.974),
                ]
            ),
        ),
        AudioSource(
            "0QzJ86rMHzQ",
            url_creator("0QzJ86rMHzQ"),
            frozenset(
                [
                    AudioSegment(0, 7.464, 25.834),
                ]
            ),
        ),
    }

    received = extract_audio_sources(
        csv_loc, url_creator=url_creator, ignore_names=["test"]
    )

    assert set(received) == expected


@pytest.mark.integration_test
def test_get_youtube(tmp_path: Path):
    """File from URL is stored onto disk"""
    source = AudioSource(
        "BaW_jenozKc", "https://www.youtube.com/watch?v=BaW_jenozKc", frozenset()
    )

    received_path = source.get_youtube(dest_dir=tmp_path)

    assert received_path.with_suffix("") == tmp_path / source.name
    assert received_path.is_file()
