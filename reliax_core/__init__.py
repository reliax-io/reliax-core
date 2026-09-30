"""Reliax core: method components of the reliability envelope.

Pure, stateless functions and small classes with no I/O, no configuration and
no network: split and Mondrian conformal prediction, Venn-Abers calibration,
the conformal test martingale, the label-free credibility p-value, the
fast-loop evaluator with its route trace and certificate reasons, kNN distance,
PSI drift, the error auditor, the composite reliability score (shown by the
product as the criticality score, 100 minus it), subjective-logic fusion and
calibration opinions.

Four components carry proofs on exchangeable data: the conformal set, the
Venn-Abers bracket, the test martingale and the credibility p-value. The
evaluator, the route trace and the certificate reasons are exact: the same
inputs give the same route. The others are signals and say so in their module
docstrings. The Reliax platform (calibration builder, reliability engine,
queues, audit service, API) holds all state and is not part of this package.
"""
__version__ = "0.2.1"

from .conformal import ConformalCalibrator
from .venn_abers import VennAbersCalibrator
from .martingale import ConformalMartingale, WATCH_THRESHOLD, ALARM_THRESHOLD
from .ood import KNNOODDetector
from .auditor import ErrorAuditor
from .drift import PSIMonitor
from .fairness import MondrianConformal, ImpactMonitor, coverage_audit
from .scoring import reliability_score
from .sl_fusion import fuse_signals, averaging_fusion
from .calibration_trust import CalibrationTrust, MARGINAL
from .credibility import CredibilityReference, credibility_p_value
from .evaluator import Policy, Envelope, Decision, evaluate, ALLOW, REVIEW, BLOCK

__all__ = [
    "__version__",
    "ConformalCalibrator",
    "VennAbersCalibrator",
    "ConformalMartingale", "WATCH_THRESHOLD", "ALARM_THRESHOLD",
    "KNNOODDetector",
    "ErrorAuditor",
    "PSIMonitor",
    "MondrianConformal", "ImpactMonitor", "coverage_audit",
    "reliability_score",
    "fuse_signals", "averaging_fusion",
    "CalibrationTrust", "MARGINAL",
    "CredibilityReference", "credibility_p_value",
    "Policy", "Envelope", "Decision", "evaluate", "ALLOW", "REVIEW", "BLOCK",
]
