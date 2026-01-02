"""
Limited Amplitude - A Python package for ecology analysis.
"""

__version__ = "0.3.0"

# Import main functions/classes here to expose them at package level
from . import occurrence
from .distance_truncation import (
    estimate_distance_from_amplitude,
    estimate_distance_hawkears,
    create_amplitude_dataframe,
    apply_distance_truncation,
    convert_to_occurrence,
    convert_to_counts,
)

__all__ = [
    "__version__",
    "occurrence",
    "estimate_distance_from_amplitude",
    "estimate_distance_hawkears",
    "create_amplitude_dataframe",
    "apply_distance_truncation",
    "convert_to_occurrence",
    "convert_to_counts",
]
