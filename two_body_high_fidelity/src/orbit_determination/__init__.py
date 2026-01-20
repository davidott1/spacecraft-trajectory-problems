"""
Orbit determination package.

This package contains modules for orbit determination, including:
- Topocentric coordinate computation (observation simulation)
- Measurement modeling with rates (range_dot, az_dot, el_dot)
- State estimation filters (Extended Kalman Filter)
"""

from src.orbit_determination.topocentric import (
  compute_topocentric_coordinates,
  compute_topocentric_coordinates_with_rates,
)
from src.orbit_determination.measurement_simulator import MeasurementSimulator
from src.orbit_determination.extended_kalman_filter import (
  ExtendedKalmanFilter,
  EKFState,
  EKFMeasurement,
  EKFConfig,
)

__all__ = [
  'compute_topocentric_coordinates',
  'compute_topocentric_coordinates_with_rates',
  'MeasurementSimulator',
  'ExtendedKalmanFilter',
  'EKFState',
  'EKFMeasurement',
  'EKFConfig',
]
