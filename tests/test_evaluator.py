"""The six rows fire in order, first match wins, and the result replays exactly."""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from reliax_core.evaluator import (  # noqa: E402
    ALLOW, BLOCK, REVIEW, Decision, Envelope, Policy, evaluate,
    ENVELOPE_INVALID, OOD_EXTREME, OOD_INPUT, EMPTY_SET, SET_AMBIGUOUS,
    PD_UPPER_EXCEEDS_CEILING, DRIFT_WATCH, CERTIFIED,
)

POL = Policy(name="credit-pd", version="4", alpha=0.05, pd_upper_allow_max=0.12)
CLEAN = dict(credibility=0.61, prediction_set=(0,), predicted_label=0, bracket=(0.02, 0.05), cell_n=412)


def test_row_6_allow_with_reasons():
    d = evaluate(POL, Envelope(**CLEAN))
    assert (d.route, d.row, d.reason_codes) == (ALLOW, 6, (CERTIFIED,))
    assert d.certificate_reasons == ("input within the scope of the guarantee", "one label left standing",
                                     "412 observations in the cell")
    assert d.trace_text() == "Route trace: row 6 matched, so every check above it passed."
    assert [t["row"] for t in d.route_trace] == [1, 2, 3, 4, 5, 6]
    assert not any(t["fired"] for t in d.route_trace[:-1])


def test_row_1_alarm_blocks_before_anything_else():
    d = evaluate(POL, Envelope(**{**CLEAN, "drift_state": "ALARM", "credibility": 0.0001, "prediction_set": ()}))
    assert (d.route, d.row, d.reason_codes) == (BLOCK, 1, (ENVELOPE_INVALID,))
    assert len(d.route_trace) == 1
    d2 = evaluate(POL, Envelope(**{**CLEAN, "guarantee_state": "suspended"}))
    assert (d2.route, d2.row) == (BLOCK, 1)


def test_row_1_does_not_fire_on_the_claimed_states():
    for state in ("under estimated covariate shift", "outcome recheck pending"):
        assert evaluate(POL, Envelope(**{**CLEAN, "guarantee_state": state})).route == ALLOW


def test_row_2_floor_and_extreme():
    d = evaluate(POL, Envelope(**{**CLEAN, "credibility": 0.005}))
    assert (d.route, d.row, d.reason_codes) == (REVIEW, 2, (OOD_INPUT,))
    d = evaluate(POL, Envelope(**{**CLEAN, "credibility": 0.0005}))
    assert (d.route, d.row, d.reason_codes) == (BLOCK, 2, (OOD_EXTREME,))
    pol = Policy(ood_action=BLOCK, ood_extreme_action=BLOCK)
    assert evaluate(pol, Envelope(**{**CLEAN, "credibility": 0.005})).route == BLOCK


def test_row_3_empty_and_ambiguous():
    d = evaluate(POL, Envelope(**{**CLEAN, "prediction_set": ()}))
    assert (d.route, d.row, d.reason_codes) == (REVIEW, 3, (EMPTY_SET,))
    d = evaluate(POL, Envelope(**{**CLEAN, "prediction_set": (0, 1)}))
    assert (d.route, d.row, d.reason_codes) == (REVIEW, 3, (SET_AMBIGUOUS,))


def test_row_4_ceiling_only_on_the_approve_side():
    d = evaluate(POL, Envelope(**{**CLEAN, "bracket": (0.04, 0.21)}))
    assert (d.route, d.row, d.reason_codes) == (REVIEW, 4, (PD_UPPER_EXCEEDS_CEILING,))
    # a decline-side prediction is not held to the approve ceiling
    d = evaluate(POL, Envelope(**{**CLEAN, "bracket": (0.4, 0.9), "predicted_label": 1}))
    assert d.route == ALLOW and d.route_trace[3]["fired"] is False
    # bracket off: row 4 is not read
    assert evaluate(Policy(bracket_on=False, pd_upper_allow_max=0.12), Envelope(**{**CLEAN, "bracket": (0.4, 0.9)})).route == ALLOW


def test_row_5_watch_by_policy():
    env = Envelope(**{**CLEAN, "drift_state": "WATCH"})
    d = evaluate(POL, env)
    assert d.route == ALLOW and "WATCH on the stream, noted, not routed on" in d.certificate_reasons
    d = evaluate(Policy(watch_action=REVIEW, pd_upper_allow_max=0.12), env)
    assert (d.route, d.row, d.reason_codes) == (REVIEW, 5, (DRIFT_WATCH,))


def test_replay_is_exact():
    a = evaluate(POL, Envelope(**CLEAN)); b = evaluate(Policy.from_dict(POL.as_dict()), Envelope.from_dict(Envelope(**CLEAN).as_dict()))
    assert a == b and a.as_dict() == b.as_dict()


def test_policy_and_envelope_validation():
    with pytest.raises(ValueError):
        Policy(invalid_action="DECLINE")
    with pytest.raises(ValueError):
        Policy(credibility_floor=0.0001, credibility_extreme=0.01)
    with pytest.raises(ValueError):
        Envelope(credibility=0.0, prediction_set=(0,))
    with pytest.raises(ValueError):
        Envelope(credibility=0.5, prediction_set=(0,), drift_state="PANIC")
