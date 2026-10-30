"""Fixtures shared by the test modules."""

import signal

import pytest

SELF_REFERENCE_BUDGET = 2.0  # seconds; a passing test takes milliseconds


@pytest.fixture
def time_bound():
    """Fail the test once it runs past SELF_REFERENCE_BUDGET.

    A self-referencing form has no finite copy: without the checks the
    conversion grows memory without bound (about 120 MB/s), fast enough to
    exhaust the machine before any memory error. A nested form that costs
    exponential time to compile never finishes. The timer stops either first.
    Where setitimer is missing the test is skipped, never run unguarded.
    """
    if not hasattr(signal, "setitimer"):
        pytest.skip("needs signal.setitimer to bound a runaway self-reference")

    def expire(signum, frame):
        pytest.fail(f"ran past its {SELF_REFERENCE_BUDGET} s budget")

    old_handler = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, SELF_REFERENCE_BUDGET)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
