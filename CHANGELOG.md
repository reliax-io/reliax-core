# Changelog

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
