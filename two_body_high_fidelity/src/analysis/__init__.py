"""
Analysis Package
================

Analysis and diagnostic tools for orbit determination and filter performance.
"""

from src.analysis.residual_diagnostics import (
    analyze_residual_behavior,
    print_residual_diagnostics,
)

__all__ = [
    'analyze_residual_behavior',
    'print_residual_diagnostics',
]
