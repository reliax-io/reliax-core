# Changelog

## 0.4.0 (19 October 2026)

Added `cutoff.py`, the lender's cut-off. `break_even_pd(margin, loss)` is
PD* = M / (M + L). `guaranteed_cutoff(scores, outcomes, target, delta)` returns
the loosest cut-off whose default rate among approvals is at most `target`
with probability at least 1 - `delta` (Learn-then-Test with fixed-sequence
binomial tests over 41 candidates, starting at the first that approves enough
cases to reject; a guarantee on exchangeable data, which does not reach below
the cut-off under which the calibration outcomes were observed).
`bracket_route(p0, p1, pd_star)` approves below the cut-off, declines above
it and reviews only a bracket that straddles it. `scipy` is now listed as a
dependency (scikit-learn already required it). No change to any existing
component or number.

## 0.3.0 (6 October 2026)

The routing rules are called rules, not rows. `Decision.rule` replaces
`Decision.row`, each route-trace entry carries `rule` instead of `row`, and
`Decision.as_dict()` writes `rule`. `trace_text()` now names what happened
("Route trace: rules 1 to 5 passed; rule 6 allows." where it said "row 6
matched, so every check above it passed."), and is also exported as a
module function. `Decision.row` and `entry["row"]` remain readable as
aliases until 1.0. No change to any route, reason code or certificate
reason, and no change to any frozen component. The rule's docstring says
calibration set where it said cohort.

## 0.2.1 (30 September 2026)

Packaging and documentation only; no code change. First release on PyPI
(`pip install reliax-core`). The publish workflow runs the tests before
building.

## 0.2.0 (28 September 2026)

Added `credibility.py` (`CredibilityReference`, `credibility_p_value`: the
label-free conformal p-value of an input's kNN distance, a guarantee) and
`evaluator.py` (`Policy`, `Envelope`, `Decision`, `evaluate`: the fast-loop
routing rule, six rows in a fixed order on certified quantities only, with the
route trace, the reason codes and the certificate reasons; exact and
replayable). `ConformalCalibrator` gained `label_p_values` and `confidence`.
The composite score is described as the product shows it: the criticality
score is 100 minus it. No change to any frozen component or to any number in
the evaluation.

## Provenance

Until 25 September 2026 this code was the `reliax_core/` directory of
reliax-evaluation, and its history is kept here. The pre-registration in that
repository names commits of that repository as the frozen method code; those
commits remain there. Version 0.1.0 is the code as it stood on 25 September
2026, with no numerical change since the freeze of 13 September 2026. The one
change since is wording in the certificate line written by
`CalibrationTrust.certificate_line` (24 September 2026). Version 0.2.0 adds
the credibility p-value and the evaluator and changes nothing in the frozen
components.

## 0.1.0 (25 September 2026)

First release as a package. The code is the `reliax_core/` directory of
[reliax-evaluation](https://github.com/reliax-io/reliax-evaluation) as it
stood on 25 September 2026, moved here with its history. No numerical change
to any component since the methods were frozen for the pre-registered
expansion (13 September 2026). Since then: the certificate line written by
`CalibrationTrust.certificate_line` says "calibration error" where it said
"disbelief" (24 September 2026).

Added in the move: `pyproject.toml` (package name `reliax-core`, import name
`reliax_core`), `__version__`, `CalibrationTrust` and `MARGINAL` exported from
the package root, a package-level test, CI on GitHub Actions, and a PyPI
publishing workflow (Trusted Publishing; not yet configured on PyPI).
