"""Fixtures shared by the test modules."""

import signal

import pytest

# Seconds. A runaway grows memory about 120 MB/s, so this also caps what a
# regression can take before the timer stops it. The slowest guarded test
# (test_nesting's fstring-format-spec case, 5000 levels at ~0.3 ms each, linear
# in depth) takes 1.3 s on 3.12 and 1.55 s on 3.11 on an idle box; the budget
# is ~4x that, so a loaded box (it failed at 2.0 s under merge load) does not
# trip it while a runaway or quadratic regression still does.
SELF_REFERENCE_BUDGET = 6.0


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
