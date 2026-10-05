"""Fast-loop evaluator: routing on certified quantities only.

Six rows, read in a fixed order; the first row that fires sets the route
(technical documentation, section 2.11; envelope schema v16). Every row reads a
certified quantity or a threshold from the policy, never an advisory signal.
The result carries the row that matched, a trace of every row above it, the
machine reason codes and the certificate reasons in words.

Class of output: exact. The same envelope and the same policy give the same
route, row, trace and reasons, so a verifier can replay the route from the
record alone, without the platform.

    1  guarantee not active on this segment (ALARM on its stream, model or
       calibration-set mismatch or expiry)           -> policy.invalid_action (BLOCK)
    2  credibility p-value below the policy floor     -> policy.ood_action (REVIEW)
       credibility below the extreme floor            -> policy.ood_extreme_action (BLOCK)
    3  prediction set not a singleton                 -> REVIEW (empty or ambiguous)
    4  bracket on, approve-side prediction, p1 above
       the policy ceiling                             -> REVIEW
    5  WATCH on this segment's stream                 -> REVIEW if the policy says so,
                                                        otherwise noted and passed
    6  otherwise                                      -> ALLOW

BLOCK never means decline: a person decides, without leaning on the model.
REVIEW means a person decides with the model's answer in front of them. The
criticality score orders the REVIEW queue and is never a condition here.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

ALLOW, REVIEW, BLOCK = "ALLOW", "REVIEW", "BLOCK"
ROUTES = (ALLOW, REVIEW, BLOCK)

# validity states of the certificate (technical documentation, section 8.1)
ACTIVE = "active"
SUSPENDED = "suspended"
UNDER_ESTIMATED_SHIFT = "under estimated covariate shift"
OUTCOME_RECHECK_PENDING = "outcome recheck pending"
GUARANTEE_STATES = (ACTIVE, SUSPENDED, UNDER_ESTIMATED_SHIFT, OUTCOME_RECHECK_PENDING)

DRIFT_STATES = ("OK", "WATCH", "ALARM")

# reason codes, one vocabulary for the record and the queue
ENVELOPE_INVALID = "ENVELOPE_INVALID"
OOD_EXTREME = "OOD_EXTREME"
OOD_INPUT = "OOD_INPUT"
EMPTY_SET = "EMPTY_SET"
SET_AMBIGUOUS = "SET_AMBIGUOUS"
PD_UPPER_EXCEEDS_CEILING = "PD_UPPER_EXCEEDS_CEILING"
DRIFT_WATCH = "DRIFT_WATCH"
CERTIFIED = "CERTIFIED"
REASON_CODES = (ENVELOPE_INVALID, OOD_EXTREME, OOD_INPUT, EMPTY_SET, SET_AMBIGUOUS,
                PD_UPPER_EXCEEDS_CEILING, DRIFT_WATCH, CERTIFIED)


@dataclass(frozen=True)
class Policy:
    """The rules a deployer sets, kept as a signed policy version.

    Thresholds are configuration in the deployer's units. Nothing here is
    tuned by Reliax at run time; a change is a new policy version.
    """
    name: str = "policy"
    version: str = "1"
    alpha: float = 0.05
    credibility_floor: float = 0.01        # row 2
    credibility_extreme: float = 0.001     # row 2, the extreme floor
    invalid_action: str = BLOCK            # row 1
    ood_action: str = REVIEW               # row 2, below the floor
    ood_extreme_action: str = BLOCK        # row 2, below the extreme floor
    watch_action: str = "INFO"             # row 5: REVIEW, or INFO to note and pass
    bracket_on: bool = True                # row 4 is read only when the bracket is on
    pd_upper_allow_max: float | None = None  # row 4 ceiling on p1; None disables the row
    approve_label: int | None = 0          # the approve-side label for row 4; None disables the row

    def __post_init__(self):
        for name in ("invalid_action", "ood_action", "ood_extreme_action"):
            if getattr(self, name) not in ROUTES:
                raise ValueError(f"{name} must be one of {ROUTES}")
        if self.watch_action not in (REVIEW, "INFO"):
            raise ValueError("watch_action must be REVIEW or INFO")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if not 0.0 <= self.credibility_extreme <= self.credibility_floor <= 1.0:
            raise ValueError("need 0 <= credibility_extreme <= credibility_floor <= 1")

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Policy":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class Envelope:
    """The certified quantities of one decision, as the engine computed them."""
    credibility: float                       # label-free conformal p-value of the input (credibility.py)
    prediction_set: tuple                    # labels left standing (conformal.py, fairness.py)
    predicted_label: int | None = None       # the model's answer, for row 4
    bracket: tuple | None = None             # (p0, p1) from venn_abers.py, or None when off
    drift_state: str = "OK"                  # OK, WATCH or ALARM on this segment's stream
    guarantee_state: str = ACTIVE            # section 8.1 state
    cell_n: int | None = None                # calibration observations in the score cell, for the reasons

    def __post_init__(self):
        if self.drift_state not in DRIFT_STATES:
            raise ValueError(f"drift_state must be one of {DRIFT_STATES}")
        if self.guarantee_state not in GUARANTEE_STATES:
            raise ValueError(f"guarantee_state must be one of {GUARANTEE_STATES}")
        if not 0.0 < self.credibility <= 1.0:
            raise ValueError("credibility must lie in (0, 1]")
        object.__setattr__(self, "prediction_set", tuple(int(v) for v in self.prediction_set))
        if self.bracket is not None:
            p0, p1 = self.bracket
            object.__setattr__(self, "bracket", (float(p0), float(p1)))

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Envelope":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class Decision:
    route: str
    row: int
    reason_codes: tuple
    certificate_reasons: tuple
    route_trace: tuple = field(default_factory=tuple)   # one entry per row read, in order

    def trace_text(self) -> str:
        """The sentence written on the certificate."""
        if self.row == 1:
            return "Route trace: row 1 matched."
        return f"Route trace: row {self.row} matched, so every check above it passed."

    def as_dict(self) -> dict:
        return {"route": self.route, "row": self.row,
                "reason_codes": list(self.reason_codes),
                "certificate_reasons": list(self.certificate_reasons),
                "route_trace": [dict(t) for t in self.route_trace]}


def _fmt(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".") if x < 0.01 else f"{x:.2f}"


def evaluate(policy: Policy, env: Envelope) -> Decision:
    """Route one decision. First row that fires wins; rows below it are not read."""
    trace = []

    def read(row: int, check: str, value, fired: bool):
        trace.append({"row": row, "check": check, "value": value, "fired": fired})

    # row 1: is the guarantee claimed on this segment right now?
    suspended = env.guarantee_state == SUSPENDED or env.drift_state == "ALARM"
    read(1, "guarantee active on this segment", env.guarantee_state if env.drift_state != "ALARM" else "ALARM", suspended)
    if suspended:
        why = "drift ALARM on this segment's stream" if env.drift_state == "ALARM" else "guarantee suspended on this segment"
        return Decision(policy.invalid_action, 1, (ENVELOPE_INVALID,),
                        (f"{why}; the guarantee is not claimed for this decision",), tuple(trace))

    # row 2: does the guarantee cover this input?
    below_extreme = env.credibility < policy.credibility_extreme
    below_floor = env.credibility < policy.credibility_floor
    read(2, f"credibility p-value at or above the floor {policy.credibility_floor:g}", env.credibility, below_floor)
    if below_extreme:
        return Decision(policy.ood_extreme_action, 2, (OOD_EXTREME,),
                        (f"input outside the scope of the guarantee: credibility {_fmt(env.credibility)} "
                         f"below the extreme floor {policy.credibility_extreme:g}",), tuple(trace))
    if below_floor:
        return Decision(policy.ood_action, 2, (OOD_INPUT,),
                        (f"input outside the scope of the guarantee: credibility {_fmt(env.credibility)} "
                         f"below the floor {policy.credibility_floor:g}",), tuple(trace))

    # row 3: did the certified set single out one answer?
    k = len(env.prediction_set)
    read(3, "prediction set is a singleton", list(env.prediction_set), k != 1)
    if k == 0:
        return Decision(REVIEW, 3, (EMPTY_SET,),
                        ("no label left standing: this input does not look like anything in the calibration data",),
                        tuple(trace))
    if k > 1:
        return Decision(REVIEW, 3, (SET_AMBIGUOUS,),
                        (f"{k} labels left standing: the model cannot separate them for this case",), tuple(trace))

    # row 4: on the approve side, is the upper end of the bracket within the ceiling?
    row4_on = (policy.bracket_on and env.bracket is not None and policy.pd_upper_allow_max is not None
               and policy.approve_label is not None and env.predicted_label == policy.approve_label)
    if row4_on:
        p1 = env.bracket[1]
        fired = p1 > policy.pd_upper_allow_max
        read(4, f"bracket upper end at or below the ceiling {policy.pd_upper_allow_max:g}", p1, fired)
        if fired:
            return Decision(REVIEW, 4, (PD_UPPER_EXCEEDS_CEILING,),
                            (f"calibrated bracket reaches {_fmt(p1)}, above the approve ceiling {policy.pd_upper_allow_max:g}",),
                            tuple(trace))
    else:
        read(4, "bracket ceiling (not read: bracket off, no ceiling, or not an approve-side prediction)", None, False)

    # row 5: is the stream on WATCH?
    watch = env.drift_state == "WATCH"
    fired = watch and policy.watch_action == REVIEW
    read(5, "no WATCH on this segment's stream", env.drift_state, fired)
    if fired:
        return Decision(REVIEW, 5, (DRIFT_WATCH,),
                        ("WATCH on this segment's stream and the policy sends WATCH to a person",), tuple(trace))

    # row 6: every check above passed
    read(6, "every check above passed", True, True)
    reasons = ["input within the scope of the guarantee", "one label left standing"]
    if env.cell_n is not None:
        reasons.append(f"{env.cell_n} observations in the cell")
    if watch:
        reasons.append("WATCH on the stream, noted, not routed on")
    return Decision(ALLOW, 6, (CERTIFIED,), tuple(reasons), tuple(trace))
