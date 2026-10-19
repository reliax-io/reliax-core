"""The lender's cut-off: break-even PD, the guaranteed cut-off and the bracket against it."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from reliax_core.cutoff import (APPROVE, DECLINE, REVIEW, break_even_pd, bracket_route,  # noqa: E402
                                guaranteed_cutoff, upper_bound)


def test_break_even_pd_matches_the_worked_example():
    assert break_even_pd(600, 5000) == pytest.approx(600 / 5600)
    with pytest.raises(ValueError):
        break_even_pd(0, 5000)


def test_bracket_route():
    assert bracket_route(0.02, 0.05, 0.107) == APPROVE
    assert bracket_route(0.15, 0.20, 0.107) == DECLINE
    assert bracket_route(0.09, 0.12, 0.107) == REVIEW


def test_upper_bound_is_clopper_pearson():
    assert upper_bound(0, 100, 0.05) == pytest.approx(1 - 0.05 ** (1 / 100))
    assert upper_bound(5, 5) == 1.0 and upper_bound(0, 0) == 1.0


def _draw(rng, n, c=0.5):
    """Scores uniform on [0, 1], default with probability c * score: the true default rate among
    approvals at cut-off t is c * t / 2."""
    s = rng.random(n)
    return s, (rng.random(n) < c * s).astype(int)


def test_the_guarantee_holds_over_many_calibration_sets():
    rng = np.random.default_rng(0)
    target, delta, c = 0.08, 0.05, 0.5                       # 25% default overall
    violations, approved, none = 0, [], 0
    for _ in range(400):
        s, y = _draw(rng, 3000, c)
        r = guaranteed_cutoff(s, y, target, delta)
        if r["cutoff"] is None:
            none += 1
            continue
        approved.append(r["approved_share"])
        violations += (c * r["cutoff"] / 2) > target           # the true rate at the chosen cut-off
    assert violations / 400 <= delta + 0.02
    # and it is not vacuous: the best possible cut-off approves 32%
    assert none <= 8 and 0.2 < np.mean(approved) < 0.32


def test_naive_cutoff_breaks_the_target_far_more_often():
    rng = np.random.default_rng(1)
    target, c, naive = 0.05, 0.5, 0
    for _ in range(400):
        s, y = _draw(rng, 3000, c)
        o = np.argsort(s)
        rate = np.cumsum(y[o]) / np.arange(1, len(s) + 1)
        ok = np.flatnonzero(rate <= target)
        naive += (c * s[o][ok[-1]] / 2) > target
    assert naive / 400 > 0.3


def test_no_cutoff_when_the_target_is_out_of_reach():
    rng = np.random.default_rng(2)
    s = rng.random(2000)
    y = (rng.random(2000) < 0.4).astype(int)                # 40% default whatever the score
    r = guaranteed_cutoff(s, y, 0.05)
    assert r["cutoff"] is None and r["candidates_tested"] == 1


def test_inputs_are_checked():
    with pytest.raises(ValueError):
        guaranteed_cutoff([0.1, 0.2], [0], 0.05)
    with pytest.raises(ValueError):
        guaranteed_cutoff([0.1], [0], 1.5)


def test_three_zones_hold_both_sides():
    """Every approved slice at most PD*, every declined slice at least PD*, jointly at 1 - delta."""
    from reliax_core.cutoff import three_zones, zone_of
    rng = np.random.default_rng(3)
    c, pd_star, delta = 0.5, 0.107, 0.05                      # a case at score s defaults with probability c * s
    cross = pd_star / c                                       # where the true rate c * s meets PD*
    bad, approve_to, decline_from = 0, [], []
    for _ in range(300):
        s, y = _draw(rng, 5000, c)
        z = three_zones(s, y, pd_star, delta)
        wrong = False
        for row in z["slices_table"]:
            a = 0.0 if row["from"] is None else row["from"]
            b = 1.0 if row["to"] is None else row["to"]
            true = c * (a + b) / 2
            wrong |= (row["zone"] == "approve" and true > pd_star) or (row["zone"] == "decline" and true < pd_star)
        bad += wrong
        approve_to.append(z["approve_to"])
        decline_from.append(z["decline_from"])
        assert z["approve_to"] is None or z["decline_from"] is None or z["approve_to"] <= z["decline_from"]
    assert bad / 300 <= delta + 0.02
    # the review band sits around the true crossing and is not the whole range
    assert 0.1 < np.mean(approve_to) < cross < np.mean(decline_from) < 0.33
    assert zone_of(0.01, z) == "approve" and zone_of(0.95, z) == "decline" and zone_of(cross, z) == "review"
