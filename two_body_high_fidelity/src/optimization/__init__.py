"""
Trajectory optimization package.

This package provides tools for trajectory optimization using the
patched conic approximation. Currently supports Earth-to-Moon transfers
with minimum ΔV optimization.

Modules:
--------
  initial_guess        : Initial trajectory guess generators (patched conic grid search)
  maneuver_optimizer   : General-purpose impulsive maneuver optimizer
"""

from src.model.orbital_mechanics import compute_circular_velocity, compute_hohmann_velocities, compute_soi_radius
from src.propagation.analytical_propagator import (
  propagate_to_soi,
  propagate_to_periapsis,
  propagate_two_body,
)
from src.optimization.initial_guess import PatchedConicGridSearch

__all__ = [
  # Core functions
  'compute_soi_radius',
  'compute_circular_velocity',
  'compute_hohmann_velocities',
  'propagate_to_soi',
  'propagate_to_periapsis',
  'propagate_two_body',
  # Optimizer
  'PatchedConicGridSearch',
]
