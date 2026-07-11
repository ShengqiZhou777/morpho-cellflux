"""Microalgae evaluation helpers kept in the active package.

Public API:
  - morphology: per-crop morphology feature extraction (FEATURES, extract_features,
    extract_population).
  - distribution_eval: distribution-level evaluation with identity baseline
    (run_distribution_eval, morphology_metrics).

Note: aggregate_microalgae.py (scalar fg-intensity Wasserstein PGC) is DEPRECATED,
superseded by distribution_eval, which sees full morphology and guards against
identity collapse.
"""
from phenoflux.eval.morphology import FEATURES, extract_features, extract_population
from phenoflux.eval.distribution_eval import (
    morphology_metrics,
    run_distribution_eval,
)

__all__ = [
    "FEATURES",
    "extract_features",
    "extract_population",
    "morphology_metrics",
    "run_distribution_eval",
]
