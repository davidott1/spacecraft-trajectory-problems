"""
Orbit determination package.

This package contains modules for orbit determination, including:
- Topocentric coordinate computation (observation simulation)
- Measurement modeling with rates (range_dot, az_dot, el_dot)
- State estimation filters (Kalman, Extended Kalman, etc.)
"""

from src.orbit_determination.topocentric import (
  compute_topocentric_coordinates,
  compute_topocentric_coordinates_with_rates,
)
from src.orbit_determination.measurement_simulator import MeasurementSimulator

__all__ = [
  'compute_topocentric_coordinates',
  'compute_topocentric_coordinates_with_rates',
  'MeasurementSimulator',
]
