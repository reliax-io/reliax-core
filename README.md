# reliax-core

The method behind the Reliax certificate, as a plain Python package: conformal
prediction sets, the calibrated bracket, the drift test, the credibility
p-value and the routing rule. Pure, stateless functions and small classes: no
I/O, no configuration, no network. Apache-2.0.

This is the code behind every number in the Reliax whitepaper. The evaluation
that produces those numbers, with the datasets, the runners, the
pre-registration and the negative results, is
[reliax-evaluation](https://github.com/reliax-io/reliax-evaluation), which
pins this package. Terms used here are defined in the
[glossary](https://github.com/reliax-io#terms).

## Install

```
pip install reliax-core
```

Or from a tagged release on GitHub:
`pip install "reliax-core @ git+https://github.com/reliax-io/reliax-core.git@v0.2.1"`.

The import name is `reliax_core`. Python 3.10 or later; numpy and
scikit-learn are the only dependencies.

## Quick start

Train a model as usual, hold out a calibration split, and route one decision.
Nothing here needs the Reliax platform.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from reliax_core import ConformalCalibrator, KNNOODDetector, CredibilityReference, Policy, Envelope, evaluate

rng = np.random.default_rng(0)
X = rng.normal(size=(3000, 4)); y = (X @ [1.2, -0.8, 0.5, 0.3] + rng.normal(size=3000) > 0).astype(int)
model = LogisticRegression().fit(X[:2000], y[:2000])          # your model, trained as usual
X_cal, y_cal = X[2000:], y[2000:]                              # a held-out calibration split

sets = ConformalCalibrator(model.predict_proba(X_cal), y_cal)  # the coverage guarantee
detector = KNNOODDetector().fit(X_cal)
credibility = CredibilityReference.from_detector(detector)     # "does the guarantee cover this input?"

x = X[0]
env = Envelope(credibility=credibility.p_value(detector.distance(x)),
               prediction_set=sets.prediction_set(model.predict_proba([x])[0], alpha=0.05))
decision = evaluate(Policy(alpha=0.05), env)
print(decision.route, decision.reason_codes, decision.trace_text())
# REVIEW ('SET_AMBIGUOUS',) Route trace: row 3 matched, so every check above it passed.
```

Both labels were left standing for this input at the 95% level, so the rule
sends it to a person. `Policy` holds the thresholds a deployer sets; `Envelope`
holds the certified quantities of one decision; `evaluate` returns the route,
the row that matched, the route trace and the reason codes. Every value is
deterministic, so the same inputs give the same route later.

The same steps on a stream add `ConformalMartingale` for the drift test,
`VennAbersCalibrator` for the bracket and `MondrianConformal` for per-segment
coverage; the signatures are in the table below and the full use is in the
runners of reliax-evaluation.

## What is in it

Every output carries one of three classes. A **guarantee** is a theorem that
holds on exchangeable data at the stated level, with no assumption on the
model or on the data distribution. An **exact** output is recomputed bit for
bit from its inputs, so a verifier can replay it from the record. A **signal**
is a measurement without such a proof. Routing rules should read guarantees;
signals order the review queue and inform recalibration.

| Module | What it computes | Class of output |
|---|---|---|
| `conformal.py` | `ConformalCalibrator`: split conformal prediction set from a calibration split. `P(Y in C(X)) >= 1 - alpha` marginally on exchangeable data (Vovk, Gammerman and Shafer, 2005). | guarantee |
| `fairness.py` | `MondrianConformal`: the same coverage per declared segment, with the segment attribute used for calibration only, never shown to the model. `coverage_audit` compares marginal and per-segment coverage. | guarantee |
| `venn_abers.py` | `VennAbersCalibrator`: inductive Venn-Abers bracket `[p0, p1]` around the base model's stated probability (Vovk and Petej, 2014). One of the two isotonic predictors is calibrated; the width of the bracket is itself a signal. | guarantee |
| `martingale.py` | `ConformalMartingale`: conformal test martingale on a label-free nonconformity score, a mixture over betting exponents; anytime-valid, WATCH at 20, ALARM at 100 (Vovk, Nouretdinov and Gammerman, 2003). | guarantee |
| `credibility.py` | `CredibilityReference`, `credibility_p_value`: the label-free conformal p-value of an input's kNN distance against the calibration distances, `(#{a_i >= a} + 1) / (n + 1)`. Valid under exchangeability; answers "does the guarantee cover this input?". Deterministic, so it replays. | guarantee |
| `evaluator.py` | `Policy`, `Envelope`, `evaluate`: the per-decision routing rule, six rows in a fixed order, first match wins, on certified quantities only; returns the route, the row, the route trace (every row read before the match), the reason codes and the certificate reasons in words. | exact |
| `ood.py` | `KNNOODDetector`: distance to the k nearest calibration rows in standardised feature space, as a percentile of the calibration set's own leave-one-out distances. | signal |
| `auditor.py` | `ErrorAuditor`: a second model, trained on the calibration split, that predicts when the base model is wrong. | signal |
| `drift.py` | `PSIMonitor`: population stability index per feature and on the score, over a rolling window. | signal |
| `scoring.py` | `reliability_score`: the 0 to 100 reliability composite. The product shows it as the criticality score, 100 minus this value, so higher means a person should look sooner. | signal |
| `sl_fusion.py` | `fuse_signals`, `averaging_fusion`: subjective-logic fusion of the signals into (belief, disbelief, uncertainty). | signal |
| `fairness.py` | `ImpactMonitor`: rolling ALLOW rate per segment and the four-fifths ratio. | signal |
| `calibration_trust.py` | `CalibrationTrust`: calibration opinion per (segment x score-bin) cell and fused per segment. A closed form of the binned calibration error and the sample size, with a Hoeffding envelope on the disbelief. Informs recalibration; never a route. | signal |

The theory behind the calibration opinion, with its finite-sample
propositions, is in
[CALIBRATION_TRUST.md](https://github.com/reliax-io/reliax-evaluation/blob/main/CALIBRATION_TRUST.md)
in the evaluation repository.

## What the guarantees say, and what they do not

- Coverage is a property of the procedure over exchangeable data, not a
  probability about any one prediction. Distribution-free per-instance
  conditional coverage is not attainable (Barber, Candès, Ramdas and
  Tibshirani, 2021). No output of this
  package is a probability that a given decision is right; how the outputs
  may and may not be read is set out once, in
  [Read this correctly](https://github.com/reliax-io#read-this-correctly).
- Per-segment coverage holds for each declared segment that has its own
  calibration rows.
- The martingale's false-alarm bound holds when calibration and production
  rows are exchangeable and the nonconformity score is computed the same way
  on both. The evaluation measured it breaking on sparse one-hot inputs with
  a small calibration set (TableShift hospital readmission), so as shipped the
  tripwire is valid on dense numeric inputs and on sparse inputs with a large
  calibration set. A distance that is exchangeable by construction on sparse
  inputs is pre-registered work, not in this release.

## Determinism

Every component is deterministic given its inputs and a seed.
`ConformalMartingale(seed=...)` draws its smoothing uniforms from a seeded
generator and `ErrorAuditor(seed=...)` seeds its gradient-boosting fit. The
credibility p-value and the evaluator use no randomness at all.
Nothing reads the clock, the environment or a file, so a result can be
recomputed later from the same inputs.

## Evidence

Every number in the whitepaper and the deck is produced by a runner in
[reliax-evaluation](https://github.com/reliax-io/reliax-evaluation) that
imports this package, and every results file is committed there, including
the negative ones: under a severe distribution shift the fused score ranks
worse than referring cases at random. The pre-registered target of twice the
wrong-approval capture of model confidence was withdrawn once that was
measured, because under that shift confidence itself falls below random
referral.

## The routing rule

`evaluate(policy, envelope)` reads six rows in this order and stops at the
first that fires:

1. The guarantee is not active on this segment (drift ALARM, model or cohort
   mismatch): the policy's `invalid_action`, BLOCK by default.
2. The credibility p-value is below the policy floor: `ood_action`, REVIEW by
   default; below the extreme floor: BLOCK.
3. The prediction set is not a singleton: REVIEW.
4. The bracket is on, the prediction is on the approve side and the bracket's
   upper end is above the policy ceiling: REVIEW.
5. WATCH on this segment's stream: REVIEW if the policy says so.
6. Otherwise: ALLOW.

The result carries the row, a trace of every row read, the reason codes and
the certificate reasons in words. The criticality score orders the REVIEW
queue and is never a condition.

## Not in this package

The Reliax platform, which holds all state: the calibration builder, the
reliability engine (drift state per segment, outcome verdicts, recalibration
tickets, signed policy versions), review queues, the audit service, the
readable view and the dashboard. The certificate format, the hash chain and
the verifier are [`reliax-certificate`](https://github.com/reliax-io/reliax-certificate);
the client is [`reliax-sdk`](https://github.com/reliax-io/reliax-python).
Not yet written: differentially private noise on the sorted
calibration scores (coverage is proven for sets; the p-values are roadmap), and
the surrogate scorer for replayable reason codes.

This package was extracted from reliax-evaluation on 25 September 2026 with
its history; the pre-registered freeze commits remain in that repository. The
dates are in [CHANGELOG.md](CHANGELOG.md).

## Development

```
git clone https://github.com/reliax-io/reliax-core.git && cd reliax-core
python -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest
```

A change to a threshold, a score or a guarantee goes through an issue first:
see [CONTRIBUTING.md](https://github.com/reliax-io/.github/blob/main/CONTRIBUTING.md).
A CLA is required before a first merge.

## Licence

Apache-2.0, with its patent grant. See [LICENSE](LICENSE).

## About this documentation

The documentation in this repository was written with the help of AI and
reviewed by the Reliax team.
