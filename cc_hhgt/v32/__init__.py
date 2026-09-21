"""V3.2 exact-pathway, patient-cross-fitted CC-HHGT pipeline.

Importing this package is deliberately CPU-only.  Torch and CUDA are imported
only after the explicit training authorization guard has passed.
"""

from .contracts import ANALYSIS_VERSION, CONTRACT_VERSION, PREDICTION_COLUMNS

__all__ = ["ANALYSIS_VERSION", "CONTRACT_VERSION", "PREDICTION_COLUMNS"]
