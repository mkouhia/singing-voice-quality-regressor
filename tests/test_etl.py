"""Test data extraction, transformation and loading."""

from pathlib import Path

from singing_classifier.etl import (
    AudioSegment,
    AudioSource,
    extract_audio_sources,
    to_youtube_url,
)


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

