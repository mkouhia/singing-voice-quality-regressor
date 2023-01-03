"""Test utility functions."""

import numpy as np
from scipy.io import wavfile


def create_wavfile_at(
    dest,
    freq: float = 440.0,
    length: float = 1.0,
    sample_rate: int = 44100,
    start_phase: int = 0.0,
):
    """Create wav file containing sine wave at destination.

    Args:
        dest: Location, where to write the .wav file.
        freq: Sine wave frequency
        length: Audio length, seconds.
        sample_rate: Audio sample rate.
        start_phase: Sine wave start phase.
    """
    data = _create_sine(freq, length, sample_rate, start_phase)
    with open(dest, "wb") as file_:
        wavfile.write(file_, 44100, data)


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
