"""
Orbit Determination Configuration Schema
======================================

Dataclasses for orbit determination configuration.
"""

from dataclasses import dataclass, field
from typing      import Optional


@dataclass
class OrbitDeterminationConfig:
  """
  Configuration for Orbit Determination processes.

  Attributes:
    enabled : bool
      Whether to run orbit determination.
    process_noise_pos : float
      Position process noise spectral density [m/s].
      The EKF computes Q_pos = (process_noise_pos * dt)^2 for each time step.
      This represents the rate of position uncertainty growth.
    process_noise_vel : float
      Velocity process noise spectral density [m/s^2].
      The EKF computes Q_vel = (process_noise_vel * sqrt(dt))^2 for each time step.
      This represents the continuous acceleration noise acting on the system.
  """
  enabled           : bool  = False
  process_noise_pos : float = 1e-4
  process_noise_vel : float = 1e-7
