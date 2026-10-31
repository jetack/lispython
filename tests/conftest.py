"""Fixtures shared by the test modules."""

import signal

import pytest

# Seconds: ~4x the slowest guarded test (test_nesting's fstring-format-spec
# case, roughly linear in DEPTH, ~1.6 s on 3.11 idle), so a loaded box does
# not trip it while a runaway or quadratic regression still does. It also
# caps a runaway's memory (see time_bound) at about 720 MB; raising it
# raises that cap.
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
