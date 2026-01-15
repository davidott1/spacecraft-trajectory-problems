"""
Orbit determination package.

This package contains modules for orbit determination, including:
- Topocentric coordinate computation (observation simulation)
- Measurement modeling
- State estimation filters (Kalman, Extended Kalman, etc.)
"""

from src.orbit_determination.topocentric import compute_topocentric_coordinates

__all__ = [
  'compute_topocentric_coordinates',
]
