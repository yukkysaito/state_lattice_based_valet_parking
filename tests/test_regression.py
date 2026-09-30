"""Regression test against regression_baseline.json (see run_tests.py --regression).

Each representative case must still succeed, be collision-free (independent
validator), and stay within loose thresholds of the recorded baseline
(switches <= base + 1, length <= 1.15 x base, time <= max(4 x base, 3 s)).
Record a new baseline with: python run_tests.py --regression --update-baseline
"""
import json
import os

import pytest

import run_tests

pytestmark = pytest.mark.skipif(not os.path.exists(run_tests.REGRESSION_FILE),
                                reason="no regression baseline recorded")


def _baseline():
    with open(run_tests.REGRESSION_FILE) as f:
        return json.load(f)


@pytest.mark.parametrize("sid", run_tests.REGRESSION_IDS)
def test_regression_case(sid):
    data = _baseline()
    if sid not in data["cases"]:
        pytest.skip(f"{sid} not in baseline")
    row = run_tests.run_case((0, sid, 42, {}, "", False, "bidirectional"))
    problems = run_tests.regression_problems(row, data["cases"][sid], data.get("thresholds",
                                                                           run_tests.REGRESSION_THRESHOLDS))
    assert not problems, f"{sid}: " + "; ".join(problems)
