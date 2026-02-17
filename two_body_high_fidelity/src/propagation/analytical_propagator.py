"""
Analytical Propagator Module
=============================

Closed-form analytical orbit propagation methods. These are exact solutions
for specific orbit types (e.g., circular) and do not require numerical
integration.

Key Functions:
--------------
  propagate_circular_orbit - Analytical circular orbit propagation via
                             in-plane rotation

Units:
------
  Position : meters [m]
  Velocity : meters per second [m/s]
  Time     : seconds [s]
"""
import numpy as np


def propagate_circular_orbit(
  state_o    : np.ndarray,
  delta_time : float,
) -> np.ndarray:
  """
  Analytically propagate a circular orbit by time delta_time.

  Exact for circular orbits. Rotates the position and velocity vectors
  by the angle swept in time delta_time within the orbit plane.

  Input:
  ------
    state_o : np.ndarray (6,)
      Initial state [pos, vel] in m and m/s.
    delta_time : float
      Time to propagate [s].

  Output:
  -------
    state_f : np.ndarray (6,)
      Propagated state [pos, vel] in m and m/s.
  """
  pos_vec_o = state_o[0:3]
  vel_vec_o = state_o[3:6]
  pos_mag_o = np.linalg.norm(pos_vec_o)
  vel_mag_o = np.linalg.norm(vel_vec_o)

  # Unit vectors in orbit plane
  pos_dir_o = pos_vec_o / pos_mag_o
  vel_dir_o = vel_vec_o / vel_mag_o

  # Angular rate for circular orbit
  omega_o     = vel_mag_o / pos_mag_o
  delta_theta = omega_o * delta_time

  # Precompute trig functions
  cos_theta = np.cos(delta_theta)
  sin_theta = np.sin(delta_theta)

  # Rotate in the orbit plane
  pos_posdiro = pos_mag_o * cos_theta
  pos_veldiro = pos_mag_o * sin_theta
  pos_vec_f   = pos_posdiro * pos_dir_o + pos_veldiro * vel_dir_o
  
  vel_posdiro = vel_mag_o * -sin_theta
  vel_veldiro = vel_mag_o *  cos_theta
  vel_vec_f   = vel_posdiro * pos_dir_o + vel_veldiro * vel_dir_o

  return np.concatenate([pos_vec_f, vel_vec_f])
