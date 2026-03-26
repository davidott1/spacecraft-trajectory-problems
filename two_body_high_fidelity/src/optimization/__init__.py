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

from src.model.orbital_mechanics import compute_circular_velocity, compute_hohmann_velocities
from src.model.frame_and_vector_converter import BodyVectorConverter
from src.optimization.patched_conic import (
  compute_soi_radius,
  propagate_to_soi,
  propagate_to_periapsis,
  propagate_two_body,
)
from src.optimization.lunar_transfer import LunarTransferOptimizer

__all__ = [
  # Core functions
  'compute_soi_radius',
  'compute_circular_velocity',
  'compute_hohmann_velocities',
  'BodyVectorConverter',
  'propagate_to_soi',
  'propagate_to_periapsis',
  'propagate_two_body',
  # Optimizer
  'LunarTransferOptimizer',
]
