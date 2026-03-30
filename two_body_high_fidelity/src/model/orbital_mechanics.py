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


def compute_hohmann_velocities(
  pos_mag_o : float,
  pos_mag_f : float,
  gp        : float,
) -> dict:
  """
  Compute Hohmann transfer estimates between two circular orbits.

  Input:
  ------
    pos_mag_o : float
      Radius of departure (inner) circular orbit [m].
    pos_mag_f : float
      Radius of arrival (outer) circular orbit [m].
    gp : float
      Gravitational parameter of central body [m³/s²].

  Output:
  -------
    estimates : dicts
      Dictionary with keys:
        'delta_vel_mag_o'   : ΔV at departure [m/s]
        'delta_vel_mag_f'   : ΔV at arrival [m/s]
        'delta_vel_total'   : total ΔV [m/s]
        'delta_time_of'     : half-period of transfer ellipse [s]
        'sma_of'            : semi-major axis of transfer ellipse [m]
        'vel_mag_o_pls'     : velocity at departure on transfer orbit [m/s]
        'vel_mag_f_mns'     : velocity at arrival on transfer orbit [m/s]
  """
  sma_of = (pos_mag_o + pos_mag_f) / 2.0

  vel_mag_o_mns  = np.sqrt(gp / pos_mag_o)
  vel_mag_f_pls  = np.sqrt(gp / pos_mag_f)
  vel_mag_o_pls  = np.sqrt(gp * (2.0 / pos_mag_o - 1.0 / sma_of))
  vel_mag_f_mns  = np.sqrt(gp * (2.0 / pos_mag_f - 1.0 / sma_of))

  delta_vel_mag_o = vel_mag_o_pls - vel_mag_o_mns
  delta_vel_mag_f = vel_mag_f_pls - vel_mag_f_mns
  delta_vel_total = abs(delta_vel_mag_o) + abs(delta_vel_mag_f)

  delta_time_of = np.pi * np.sqrt(sma_of**3 / gp)

  return {
    'delta_vel_mag_o' : delta_vel_mag_o,
    'delta_vel_mag_f' : delta_vel_mag_f,
    'delta_vel_total' : delta_vel_total,
    'delta_time_of'   : delta_time_of,
    'sma_of'          : sma_of,
    'vel_mag_o_pls'   : vel_mag_o_pls,
    'vel_mag_f_mns'   : vel_mag_f_mns,
  }


def compute_soi_radius(
  sma          : float,
  gp_primary   : float,
  gp_secondary : float,
) -> float:
  """
  Compute sphere of influence radius using Laplace's formula.

  r_soi = a * (m_secondary / m_primary)^(2/5)

  Input:
  ------
    sma : float
      Semi-major axis of the secondary body's orbit around the primary [m].
    gp_primary : float
      Gravitational parameter of the primary body [m³/s²].
    gp_secondary : float
      Gravitational parameter of the secondary body [m³/s²].

  Output:
  -------
    r_soi : float
      Sphere of influence radius [m].
  """
  return sma * (gp_secondary / gp_primary) ** 0.4
