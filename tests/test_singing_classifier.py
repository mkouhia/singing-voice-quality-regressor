"""Module-level tests."""

from singing_classifier import __version__


def test_version():
    """Version is not changed by accident."""
    assert __version__ == "0.1.0"
