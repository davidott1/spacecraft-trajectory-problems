"""
Analytical Propagator
=====================

Two-body (Keplerian) propagation with optional event detection.
Wraps scipy.integrate.solve_ivp using point-mass gravity for lightweight
trajectory computation (e.g. patched conic transfers).

Key Functions:
--------------
  propagate_to_soi       - Two-body propagation with SOI crossing event
  propagate_to_periapsis - Two-body propagation with periapsis event
  propagate_two_body     - Simple two-body numerical propagation

Units:
------
  Position : meters [m]
  Velocity : meters per second [m/s]
  Time     : seconds [s] (Ephemeris Time, seconds past J2000)
  GP       : m³/s²
"""
import numpy as np

from scipy.integrate import solve_ivp

from src.model.dynamics import AccelerationSTMDot, GeneralStateEquationsOfMotion
from src.model.frame_and_vector_converter import BodyVectorConverter
from src.schemas.gravity import GravityModelConfig
from src.schemas.spacecraft import SpacecraftProperties


# --------------------------------------------------------------------------
# Two-Body Dynamics
# --------------------------------------------------------------------------

def _build_two_body_eom(
  gp : float,
) -> GeneralStateEquationsOfMotion:
  """
  Build a two-body point-mass equations-of-motion object.

  Input:
  ------
    gp : float
      Gravitational parameter [m³/s²].

  Output:
  -------
    eom : GeneralStateEquationsOfMotion
      Equations-of-motion instance whose state_time_derivative(t, y)
      computes d/dt [r, v] = [v, -gp * r / |r|³].
  """
  gravity_config = GravityModelConfig(gp=gp)
  spacecraft     = SpacecraftProperties()
  acceleration   = AccelerationSTMDot(gravity_config=gravity_config, spacecraft=spacecraft)
  return GeneralStateEquationsOfMotion(acceleration)


# --------------------------------------------------------------------------
# Propagation with Event Detection
# --------------------------------------------------------------------------

def propagate_to_soi(
  state_o          : np.ndarray,
  time_et_o        : float,
  gp               : float,
  naif_id_secondary : int,
  naif_id_primary   : int,
  soi_radius       : float,
  max_time_s       : float,
  rtol             : float = 1e-12,
  atol             : float = 1e-12,
) -> dict:
  """
  Propagate under central body two-body gravity until SOI crossing.

  Uses scipy.integrate.solve_ivp with a terminal event function that
  detects when the spacecraft enters the target body's sphere of influence.

  Input:
  ------
    state_o : np.ndarray (6,)
      Initial state [pos, vel] in observer-centered J2000 [m, m/s].
    time_et_o : float
      Initial ephemeris time [s past J2000].
    gp : float
      Gravitational parameter of central body [m³/s²].
    naif_id_secondary : int
      NAIF ID of the secondary body (e.g., Moon = 301).
    naif_id_primary : int
      NAIF ID of the primary body (e.g., Earth = 399).
    soi_radius : float
      Sphere of influence radius of the target body [m].
    max_time_s : float
      Maximum propagation time [s].
    rtol : float
      Relative tolerance for integration.
    atol : float
      Absolute tolerance for integration.

  Output:
  -------
    result : dict
      Keys:
        'success'           : bool - Whether SOI crossing was detected
        'message'           : str  - Status message (on failure)
        'soi_time_et'       : float - ET at SOI crossing
        'soi_state'         : np.ndarray (6,) - State at SOI crossing
        'trajectory_times'  : np.ndarray (N,) - ET times of trajectory
        'trajectory_states' : np.ndarray (6, N) - States along trajectory
  """
  # SOI crossing event function
  def soi_event(time_et, object_state):
    secondary_state = BodyVectorConverter.get_body_state(naif_id_secondary, time_et, naif_id_primary)
    secondary_to_object_pos_mag = np.linalg.norm(object_state[0:3] - secondary_state[0:3])
    return secondary_to_object_pos_mag - soi_radius

  soi_event.terminal  = True
  soi_event.direction = -1  # trigger when entering SOI (distance decreasing)

  # Time span and evaluation points
  time_span   = (time_et_o, time_et_o + max_time_s)
  n_points = max(1000, int(max_time_s / 60))
  n_points = min(n_points, 50000)
  time_eval   = np.linspace(time_et_o, time_et_o + max_time_s, n_points)

  # Build EOM
  eom = _build_two_body_eom(gp)

  # Integrate
  sol = solve_ivp(
    fun          = eom.state_time_derivative,
    t_span       = time_span,
    y0           = state_o,
    method       = 'DOP853',
    rtol         = rtol,
    atol         = atol,
    events       = soi_event,
    t_eval       = time_eval,
    dense_output = True,
  )

  # Check if SOI crossing occurred
  if sol.t_events[0].size > 0:
    t_soi = sol.t_events[0][0]
    y_soi = sol.y_events[0][0]

    # Build trajectory up to and including SOI crossing point
    mask   = sol.t <= t_soi
    times  = np.append(sol.t[mask], t_soi)
    states = np.hstack([sol.y[:, mask], y_soi.reshape(6, 1)])

    return {
      'success'           : True,
      'soi_time_et'       : t_soi,
      'soi_state'         : y_soi,
      'trajectory_times'  : times,
      'trajectory_states' : states,
    }
  else:
    return {
      'success'           : False,
      'message'           : 'No SOI crossing detected within max propagation time',
      'trajectory_times'  : sol.t,
      'trajectory_states' : sol.y,
    }


def propagate_to_periapsis(
  state0     : np.ndarray,
  t0_et      : float,
  gp         : float,
  max_time_s : float,
  rtol       : float = 1e-12,
  atol       : float = 1e-12,
) -> dict:
  """
  Propagate under two-body gravity until periapsis passage.

  Detects periapsis as the point where r·v transitions from negative
  (approaching) to positive (receding).

  Input:
  ------
    state0 : np.ndarray (6,)
      Initial state [pos, vel] in body-centered frame [m, m/s].
    t0_et : float
      Initial ephemeris time [s past J2000].
    gp : float
      Gravitational parameter of central body [m³/s²].
    max_time_s : float
      Maximum propagation time [s].
    rtol : float
      Relative tolerance for integration.
    atol : float
      Absolute tolerance for integration.

  Output:
  -------
    result : dict
      Keys:
        'success'              : bool
        'message'              : str (on failure)
        'periapsis_time_et'    : float - ET at periapsis
        'periapsis_state'      : np.ndarray (6,) - State at periapsis
        'periapsis_radius'     : float - Periapsis distance from center [m]
        'periapsis_velocity'   : float - Speed at periapsis [m/s]
        'trajectory_times'     : np.ndarray (N,)
        'trajectory_states'    : np.ndarray (6, N)
  """
  # Periapsis event: r·v = 0 with direction from negative to positive
  def periapsis_event(t, y):
    return np.dot(y[0:3], y[3:6])

  periapsis_event.terminal  = True
  periapsis_event.direction = 1  # r·v: negative → positive at periapsis on approach

  # Time span and evaluation points
  t_span   = (t0_et, t0_et + max_time_s)
  n_points = max(500, int(max_time_s / 10))
  n_points = min(n_points, 10000)
  t_eval   = np.linspace(t0_et, t0_et + max_time_s, n_points)

  # Integrate
  eom = _build_two_body_eom(gp)
  sol = solve_ivp(
    fun          = eom.state_time_derivative,
    t_span       = t_span,
    y0           = state0,
    method       = 'DOP853',
    rtol         = rtol,
    atol         = atol,
    events       = periapsis_event,
    t_eval       = t_eval,
    dense_output = True,
  )

  # Check if periapsis was detected
  if sol.t_events[0].size > 0:
    t_peri = sol.t_events[0][0]
    y_peri = sol.y_events[0][0]

    # Build trajectory up to and including periapsis
    mask   = sol.t <= t_peri
    times  = np.append(sol.t[mask], t_peri)
    states = np.hstack([sol.y[:, mask], y_peri.reshape(6, 1)])

    peri_radius   = np.linalg.norm(y_peri[0:3])
    peri_velocity = np.linalg.norm(y_peri[3:6])

    return {
      'success'            : True,
      'periapsis_time_et'  : t_peri,
      'periapsis_state'    : y_peri,
      'periapsis_radius'   : peri_radius,
      'periapsis_velocity' : peri_velocity,
      'trajectory_times'   : times,
      'trajectory_states'  : states,
    }
  else:
    return {
      'success'           : False,
      'message'           : 'No periapsis detected within max propagation time',
      'trajectory_times'  : sol.t,
      'trajectory_states' : sol.y,
    }


def propagate_two_body(
  state0   : np.ndarray,
  t0_et    : float,
  tf_et    : float,
  gp       : float,
  n_points : int   = 1000,
  rtol     : float = 1e-12,
  atol     : float = 1e-12,
) -> dict:
  """
  Simple two-body numerical propagation.

  Input:
  ------
    state0 : np.ndarray (6,)
      Initial state [pos, vel] [m, m/s].
    t0_et : float
      Initial ephemeris time [s past J2000].
    tf_et : float
      Final ephemeris time [s past J2000].
    gp : float
      Gravitational parameter [m³/s²].
    n_points : int
      Number of output points.
    rtol : float
      Relative tolerance.
    atol : float
      Absolute tolerance.

  Output:
  -------
    result : dict
      Keys:
        'success' : bool
        'message' : str
        'times'   : np.ndarray (N,) - ET times
        'states'  : np.ndarray (6, N) - State history
  """
  t_eval = np.linspace(t0_et, tf_et, n_points)

  eom = _build_two_body_eom(gp)
  sol = solve_ivp(
    fun    = eom.state_time_derivative,
    t_span = (t0_et, tf_et),
    y0     = state0,
    method = 'DOP853',
    rtol   = rtol,
    atol   = atol,
    t_eval = t_eval,
  )

  return {
    'success' : sol.success,
    'message' : sol.message if hasattr(sol, 'message') else '',
    'times'   : sol.t,
    'states'  : sol.y,
  }
