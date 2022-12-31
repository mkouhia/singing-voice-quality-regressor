"""Test multi-processing helpers."""

import random

import pytest

from singing_classifier.processing import perform_multiprocess


@pytest.mark.parametrize("n_processes", [1, 2, 8])
def test_perform_multiprocess(n_processes):
    """Results are obtained from multiprocess."""

    def action(item: int):
        return item + 1

    items = range(10)

    received = perform_multiprocess(action, items, n_processes)

    assert set(received) == {(i, i + 1) for i in items}


def test_perform_multiprocess_errors():
    """On error, None result is received."""

    def action(item: int):
        if item == 3:
            raise ValueError
        return item + 1

    items = range(5)
    received = perform_multiprocess(action, items, num_tries=2)

    assert set(received) == {(i, (i + 1 if i != 3 else None)) for i in items}


def test_perform_multiprocess_retries():
    """On errors, retries are made."""

    def action(item: int):
        if random.random() < 0.2:
            raise ValueError
        return item + 1

    items = range(40)
    received = perform_multiprocess(action, items, num_tries=20)

    assert set(received) == {(i, i + 1) for i in items}
