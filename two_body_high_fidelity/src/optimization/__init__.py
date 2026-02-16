"""
Trajectory optimization package.

This package provides tools for trajectory optimization using the
patched conic approximation. Currently supports Earth-to-Moon transfers
with minimum ΔV optimization.

Modules:
--------
  patched_conic    : Core patched conic math (SOI, frame transforms, propagation)
  lunar_transfer   : Earth-Moon transfer optimizer
"""

from src.optimization.patched_conic import (
  compute_soi_radius,
  compute_circular_velocity,
  compute_hohmann_estimates,
  propagate_circular_orbit,
  get_body_state,
  earth_to_moon_state,
  moon_to_earth_state,
  propagate_to_soi,
  propagate_to_periapsis,
  propagate_two_body,
)
from src.optimization.lunar_transfer import LunarTransferOptimizer

__all__ = [
  # Core functions
  'compute_soi_radius',
  'compute_circular_velocity',
  'compute_hohmann_estimates',
  'propagate_circular_orbit',
  'get_body_state',
  'earth_to_moon_state',
  'moon_to_earth_state',
  'propagate_to_soi',
  'propagate_to_periapsis',
  'propagate_two_body',
  # Optimizer
  'LunarTransferOptimizer',
]
