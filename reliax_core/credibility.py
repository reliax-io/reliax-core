"""Input credibility: a label-free conformal p-value on a distance.

Take the distance a(x) of an input to its k nearest calibration rows, and the
calibration rows' own leave-one-out distances a_1 .. a_n. The credibility

    p(x) = ( #{ i : a_i >= a(x) } + 1 ) / (n + 1)

is a valid p-value under exchangeability: P(p <= t) <= t for every t in (0, 1),
whatever the data distribution (Vovk, Gammerman and Shafer, 2005). It answers
"does the guarantee cover this input?". A small p says the input sits farther
from the calibration data than almost all of the calibration data sits from
itself, so the certificate has no evidence about it either way, and a person
decides. It says nothing about whether the model is right on that input.

Class of output: guarantee. The value is deterministic (no smoothing), so a
replay from the recorded distance gives the same number. The test martingale
keeps its own smoothed p-values for the stream (martingale.py); the two are
not interchangeable.
"""
import numpy as np


def credibility_p_value(distance: float, calib_distances: np.ndarray) -> float:
    """p = (#{a_i >= a} + 1) / (n + 1) against the calibration distances (any order)."""
    a = np.asarray(calib_distances, dtype=float)
    n = int(a.size)
    if n == 0:
        raise ValueError("calibration distances are empty")
    n_ge = int(np.count_nonzero(a >= float(distance)))
    return (n_ge + 1) / (n + 1)


class CredibilityReference:
    """The calibration distances, kept sorted, with the p-value lookup."""

    def __init__(self, calib_distances: np.ndarray):
        self.calib = np.sort(np.asarray(calib_distances, dtype=float))
        self.n = int(self.calib.size)
        if self.n == 0:
            raise ValueError("calibration distances are empty")

    @classmethod
    def from_detector(cls, detector) -> "CredibilityReference":
        """From a fitted KNNOODDetector: its leave-one-out calibration distances."""
        return cls(detector.calib_dists)

    def p_value(self, distance: float) -> float:
        n_lt = int(np.searchsorted(self.calib, float(distance), side="left"))
        n_ge = self.n - n_lt
        return (n_ge + 1) / (self.n + 1)

    def p_values_batch(self, distances: np.ndarray) -> np.ndarray:
        d = np.asarray(distances, dtype=float)
        n_lt = np.searchsorted(self.calib, d, side="left")
        return (self.n - n_lt + 1) / (self.n + 1)

    def assess(self, detector, x: np.ndarray, floor: float = 0.01) -> dict:
        """Distance and credibility of one input against a fitted KNNOODDetector."""
        d = detector.distance(x)
        p = self.p_value(d)
        return {"distance": round(d, 6), "credibility": round(p, 6), "covered": p >= floor,
                "floor": floor, "calibration_n": self.n}
