"""
Orbit Propagation Package
=========================

Provides numerical integration utilities for spacecraft orbit propagation.
"""

from .propagator      import propagate_state_numerical_integration

__all__ = ['propagate_state_numerical_integration']
