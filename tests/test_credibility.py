"""The credibility p-value is valid under exchangeability and deterministic."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from reliax_core.credibility import CredibilityReference, credibility_p_value  # noqa: E402
from reliax_core.ood import KNNOODDetector  # noqa: E402
from reliax_core.conformal import ConformalCalibrator  # noqa: E402


def test_p_value_formula_and_range():
    calib = np.array([0.1, 0.2, 0.3, 0.4])
    assert credibility_p_value(0.25, calib) == (2 + 1) / 5
    assert credibility_p_value(10.0, calib) == 1 / 5          # farther than everything: smallest possible p
    assert credibility_p_value(0.0, calib) == 1.0             # closer than everything
    ref = CredibilityReference(calib)
    assert ref.p_value(0.25) == credibility_p_value(0.25, calib)
    assert list(ref.p_values_batch([0.25, 10.0, 0.0])) == [3 / 5, 1 / 5, 1.0]


def test_validity_under_exchangeability():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 6))
    det = KNNOODDetector(k=10).fit(X[:1500])
    ref = CredibilityReference.from_detector(det)
    p = ref.p_values_batch([det.distance(x) for x in X[1500:]])
    for t in (0.01, 0.05, 0.10):
        assert np.mean(p <= t) <= t + 0.02, t          # P(p <= t) <= t, up to sampling noise
    far = X[1500:] + 6.0
    p_far = ref.p_values_batch([det.distance(x) for x in far])
    assert np.mean(p_far <= 0.01) > 0.95               # a shifted population is outside the scope


def test_determinism_and_assess():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3))
    det = KNNOODDetector(k=5).fit(X)
    ref = CredibilityReference.from_detector(det)
    a, b = ref.assess(det, X[0]), ref.assess(det, X[0])
    assert a == b and a["covered"] and a["calibration_n"] == 300


def test_conformal_confidence_and_label_p_values():
    rng = np.random.default_rng(2)
    probs = rng.dirichlet([1, 1], size=500)
    labels = (rng.random(500) < probs[:, 1]).astype(int)
    cal = ConformalCalibrator(probs, labels)
    pv = cal.label_p_values(np.array([0.9, 0.1]))
    assert len(pv) == 2 and all(0 < v <= 1 for v in pv)
    assert cal.confidence(np.array([0.9, 0.1])) == 1.0 - sorted(pv, reverse=True)[1]
    assert cal.confidence(np.array([0.5, 0.5])) < cal.confidence(np.array([0.99, 0.01]))
