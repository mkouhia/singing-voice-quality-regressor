"""Multi-processing helpers."""

import time
from collections.abc import Generator, Iterable
from multiprocessing import Process, Queue
from typing import Any, Callable, TypeVar

T = TypeVar("T")
U = TypeVar("U")


class RetryPerformer(Process):

    """Process for downloading files in parallel."""

    def __init__(
        self,
        action: Callable[[T], U],
        in_queue: Queue,
        out_queue: Queue,
        num_tries: int,
        retry_delay_seconds: float,
        attach_stats: bool = False,
    ) -> None:
        super().__init__(daemon=True)
        self.action = action
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.num_tries = num_tries
        self.retry_delay_seconds = retry_delay_seconds
        self.attach_stats = attach_stats

    def run(self):
        while True:
            in_item: T
            run_kwargs: dict[str, Any]
            in_item, run_kwargs = self.in_queue.get(block=True)

            if (
                prev_time := run_kwargs["prev_time"]
            ) > 0.0 and time.time() - prev_time < self.retry_delay_seconds:
                self.in_queue.put((in_item, run_kwargs))
                continue

            try:
                result = self.action(in_item)
                out_item = (
                    (in_item, result, run_kwargs)
                    if self.attach_stats
                    else (in_item, result)
                )
                self.out_queue.put(out_item)
            except Exception as exc:
                run_kwargs["num_tries"] += 1
                run_kwargs["prev_time"] = time.time()
                run_kwargs["exception_message"] = str(exc)
                if run_kwargs["num_tries"] == self.num_tries:
                    out_item = (
                        (in_item, None, run_kwargs)
                        if self.attach_stats
                        else (in_item, None)
                    )
                    self.out_queue.put(out_item)
                else:
                    self.in_queue.put((in_item, run_kwargs))


def perform_multiprocess(
    action: Callable[[T], U],
    items: Iterable[T],
    n_processes: int = 1,
    num_tries: int = 1,
    retry_delay_seconds: float = 0.0,
    attach_stats=False,
) -> Generator[tuple[T, U], None, None] | Generator[
    tuple[T, U, dict[str, Any]], None, None
]:
    """Perform action using multiple processes, with retries.

    Args:
        action: A callable, which receives one argument from iterable
          `items`. The calls are made in `RetryPerformer` processes,
          and the results are yielded.
        items: Iterable of tems to be processed with `action`.
        n_processes: Number of parallel processes. Defaults to 1.
        num_tries: Number of how many times the call is attempted.
          If an exception is raised, the call is tried in total
          `num_tries` times. Defaults to 1, or no retries.
        retry_delay_seconds: How long a delay is made after a failed
          call to an item.
        attach_stats: If True, return descriptive statistics alongside
          the result. The dictionary contains at least keys 'num_tries',
          'prev_time', and possibly 'exception_message'

    Yields:
        Tuples (original item, result) that are received from action,
        or a tuple (original item, result, stats dict) if `attach_stats`
        is True.
    """
    in_queue = Queue()
    out_queue = Queue()
    pending_items = len(items)
    for item in items:
        in_queue.put((item, {"num_tries": 0, "prev_time": 0.0}))

    processes: list[RetryPerformer] = []
    for _ in range(n_processes or 1):
        proc = RetryPerformer(
            action,
            in_queue,
            out_queue,
            num_tries,
            retry_delay_seconds,
            attach_stats=attach_stats,
        )
        processes.append(proc)
        proc.start()

    while pending_items:
        item = out_queue.get(block=True)
        pending_items -= 1
        yield item

    for proc in processes:
        proc.terminate()
