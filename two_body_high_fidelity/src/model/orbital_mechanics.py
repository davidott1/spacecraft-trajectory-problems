"""
Orbital Mechanics
=================

Common closed-form expressions for Keplerian orbital mechanics.

Units:
------
  position : meters [m]
  velocity : meters per second [m/s]
  time     : seconds [s]
  gp       : m³/s²
"""
import numpy as np


def compute_circular_velocity(
  pos_mag : float,
  gp      : float,
) -> float:
  """
  Compute circular orbit velocity at radius r.

  vel_mag = sqrt(gp / r)

  Input:
  ------
    pos_mag : float
      Orbital radius [m].
    gp : float
      Gravitational parameter of central body [m³/s²].

  Output:
  -------
    vel_mag : float
      Circular orbital velocity [m/s].
  """
  return np.sqrt(gp / pos_mag)
