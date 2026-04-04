"""Analysis utilities (metrics, performance profiles)."""

from .metrics import gap, success_rate
from .performance_profiles import performance_profile, plot_performance_profile

__all__ = [
    "gap",
    "success_rate",
    "performance_profile",
    "plot_performance_profile",
]

