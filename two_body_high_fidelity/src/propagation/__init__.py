"""
Orbit Propagation Package
=========================

Provides numerical integration utilities for spacecraft orbit propagation.
"""

from .propagator import propagate_orbit

__all__ = ['propagate_orbit']
