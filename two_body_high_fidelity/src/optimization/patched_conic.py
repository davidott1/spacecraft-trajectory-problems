"""
Patched Conic Trajectory Module
================================

Core functions for patched conic trajectory computation between two gravitational
bodies. Provides sphere-of-influence detection, frame transformations, and
two-body propagation with event detection.

Key Functions:
--------------
  compute_soi_radius          - Sphere of influence radius from mass ratio
  compute_circular_velocity   - Circular orbit velocity at radius
  compute_hohmann_estimates   - Hohmann transfer ΔV and timing estimates
  propagate_circular_orbit    - Analytical circular orbit propagation
  get_body_state              - SPICE body state query
  earth_to_moon_state         - Earth-centered to Moon-centered frame transform
  moon_to_earth_state         - Moon-centered to Earth-centered frame transform
  propagate_to_soi            - Two-body propagation with SOI crossing event
  propagate_to_periapsis      - Two-body propagation with periapsis event
  propagate_two_body          - Simple two-body numerical propagation

Units:
------
  Position : meters [m]
  Velocity : meters per second [m/s]
  Time     : seconds [s] (Ephemeris Time, seconds past J2000)
  GP       : m³/s²
"""
import numpy as np
import spiceypy as spice

from scipy.integrate import solve_ivp

from src.model.constants import SOLARSYSTEMCONSTANTS, NAIFIDS, CONVERTER


# --------------------------------------------------------------------------
# Analytical Functions
# --------------------------------------------------------------------------

def compute_soi_radius(
  sma          : float,
  gp_secondary : float,
  gp_primary   : float,
) -> float:
  """
  Compute sphere of influence radius using Laplace's formula.

  r_soi = a * (m_secondary / m_primary)^(2/5)

  Input:
  ------
    sma : float
      Semi-major axis of the secondary body's orbit around the primary [m].
    gp_secondary : float
      Gravitational parameter of the secondary body [m³/s²].
    gp_primary : float
      Gravitational parameter of the primary body [m³/s²].

  Output:
  -------
    r_soi : float
      Sphere of influence radius [m].
  """
  return sma * (gp_secondary / gp_primary) ** 0.4


def compute_circular_velocity(
  r  : float,
  gp : float,
) -> float:
  """
  Compute circular orbit velocity at radius r.

  v_circ = sqrt(gp / r)

  Input:
  ------
    r : float
      Orbital radius [m].
    gp : float
      Gravitational parameter of central body [m³/s²].

  Output:
  -------
    v_circ : float
      Circular orbital velocity [m/s].
  """
  return np.sqrt(gp / r)


def compute_hohmann_estimates(
  r1 : float,
  r2 : float,
  gp : float,
) -> dict:
  """
  Compute Hohmann transfer estimates between two circular orbits.

  Input:
  ------
    r1 : float
      Radius of departure (inner) circular orbit [m].
    r2 : float
      Radius of arrival (outer) circular orbit [m].
    gp : float
      Gravitational parameter of central body [m³/s²].

  Output:
  -------
    estimates : dict
      Dictionary with keys:
        'dv1'               : ΔV at departure [m/s]
        'dv2'               : ΔV at arrival [m/s]
        'delta_vel_total'   : Total ΔV [m/s]
        'transfer_time' : Half-period of transfer ellipse [s]
        'a_transfer'    : Semi-major axis of transfer ellipse [m]
        'v_departure'   : Velocity at departure on transfer orbit [m/s]
        'v_arrival'     : Velocity at arrival on transfer orbit [m/s]
  """
  a_transfer = (r1 + r2) / 2.0

  v_circ_1    = np.sqrt(gp / r1)
  v_circ_2    = np.sqrt(gp / r2)
  v_departure = np.sqrt(gp * (2.0 / r1 - 1.0 / a_transfer))
  v_arrival   = np.sqrt(gp * (2.0 / r2 - 1.0 / a_transfer))

  dv1 = v_departure - v_circ_1
  dv2 = v_circ_2 - v_arrival

  transfer_time = np.pi * np.sqrt(a_transfer**3 / gp)

  return {
    'dv1'               : dv1,
    'dv2'               : dv2,
    'delta_vel_total'   : abs(dv1) + abs(dv2),
    'transfer_time' : transfer_time,
    'a_transfer'    : a_transfer,
    'v_departure'   : v_departure,
    'v_arrival'     : v_arrival,
  }


def propagate_circular_orbit(
  state0 : np.ndarray,
  dt     : float,
) -> np.ndarray:
  """
  Analytically propagate a circular orbit by time dt.

  Exact for circular orbits. Rotates the position and velocity vectors
  by the angle swept in time dt within the orbit plane.

  Input:
  ------
    state0 : np.ndarray (6,)
      Initial state [pos, vel] in meters and m/s.
    dt : float
      Time to propagate [s]. Can be positive or negative.

  Output:
  -------
    state_new : np.ndarray (6,)
      Propagated state [pos, vel] in meters and m/s.
  """
  r0    = state0[0:3]
  v0    = state0[3:6]
  r_mag = np.linalg.norm(r0)
  v_mag = np.linalg.norm(v0)

  # Unit vectors in orbit plane
  r_hat = r0 / r_mag
  v_hat = v0 / v_mag

  # Angular rate for circular orbit
  omega = v_mag / r_mag
  theta = omega * dt

  cos_t = np.cos(theta)
  sin_t = np.sin(theta)

  # Rotate in the orbit plane
  r_new = r_mag * (cos_t * r_hat + sin_t * v_hat)
  v_new = v_mag * (-sin_t * r_hat + cos_t * v_hat)

  return np.concatenate([r_new, v_new])


# --------------------------------------------------------------------------
# SPICE Wrappers
# --------------------------------------------------------------------------

def get_body_state(
  naif_id     : int,
  time_et     : float,
  observer_id : int = 399,
) -> np.ndarray:
  """
  Get body state from SPICE in J2000 frame.

  Input:
  ------
    naif_id : int
      NAIF ID of the target body.
    time_et : float
      Ephemeris time [s past J2000].
    observer_id : int
      NAIF ID of the observer body (default: Earth = 399).

  Output:
  -------
    state : np.ndarray (6,)
      State vector [pos, vel] in meters and m/s, J2000 frame.
  """
  state_km, _ = spice.spkez(naif_id, time_et, 'J2000', 'NONE', observer_id)
  return np.array(state_km) * CONVERTER.M_PER_KM


def earth_to_moon_state(
  state_earth_j2000 : np.ndarray,
  time_et           : float,
) -> np.ndarray:
  """
  Transform state from Earth-centered J2000 to Moon-centered J2000.

  Input:
  ------
    state_earth_j2000 : np.ndarray (6,)
      State in Earth-centered J2000 [m, m/s].
    time_et : float
      Ephemeris time [s past J2000].

  Output:
  -------
    state_moon : np.ndarray (6,)
      State in Moon-centered J2000 [m, m/s].
  """
  moon_state = get_body_state(NAIFIDS.MOON, time_et, NAIFIDS.EARTH)
  return state_earth_j2000 - moon_state


def moon_to_earth_state(
  state_moon : np.ndarray,
  time_et    : float,
) -> np.ndarray:
  """
  Transform state from Moon-centered J2000 to Earth-centered J2000.

  Input:
  ------
    state_moon : np.ndarray (6,)
      State in Moon-centered J2000 [m, m/s].
    time_et : float
      Ephemeris time [s past J2000].

  Output:
  -------
    state_earth_j2000 : np.ndarray (6,)
      State in Earth-centered J2000 [m, m/s].
  """
  moon_state = get_body_state(NAIFIDS.MOON, time_et, NAIFIDS.EARTH)
  return state_moon + moon_state


# --------------------------------------------------------------------------
# Two-Body Dynamics
# --------------------------------------------------------------------------

def _two_body_eom(
  t  : float,
  y  : np.ndarray,
  gp : float,
) -> np.ndarray:
  """
  Two-body equations of motion (right-hand side).

  d/dt [r, v] = [v, -gp * r / |r|³]

  Input:
  ------
    t : float
      Time (unused for autonomous system, but required by solve_ivp).
    y : np.ndarray (6,)
      State vector [pos, vel].
    gp : float
      Gravitational parameter [m³/s²].

  Output:
  -------
    dydt : np.ndarray (6,)
      State derivative [vel, acc].
  """
  r     = y[0:3]
  v     = y[3:6]
  r_mag = np.linalg.norm(r)
  a     = -gp * r / r_mag**3
  return np.concatenate([v, a])


# --------------------------------------------------------------------------
# Propagation with Event Detection
# --------------------------------------------------------------------------

def propagate_to_soi(
  state0       : np.ndarray,
  t0_et        : float,
  gp_central   : float,
  target_naif_id : int,
  observer_naif_id : int,
  soi_radius   : float,
  max_time_s   : float,
  rtol         : float = 1e-12,
  atol         : float = 1e-12,
) -> dict:
  """
  Propagate under central body two-body gravity until SOI crossing.

  Uses scipy.integrate.solve_ivp with a terminal event function that
  detects when the spacecraft enters the target body's sphere of influence.

  Input:
  ------
    state0 : np.ndarray (6,)
      Initial state [pos, vel] in observer-centered J2000 [m, m/s].
    t0_et : float
      Initial ephemeris time [s past J2000].
    gp_central : float
      Gravitational parameter of central body [m³/s²].
    target_naif_id : int
      NAIF ID of the target body (e.g., Moon = 301).
    observer_naif_id : int
      NAIF ID of the observer/central body (e.g., Earth = 399).
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
        'success'            : bool - Whether SOI crossing was detected
        'message'            : str  - Status message (on failure)
        'soi_time_et'        : float - ET at SOI crossing
        'soi_state'          : np.ndarray (6,) - State at SOI crossing
        'trajectory_times'   : np.ndarray (N,) - ET times of trajectory
        'trajectory_states'  : np.ndarray (6, N) - States along trajectory
  """
  # SOI crossing event function
  def soi_event(t, y):
    target_state = get_body_state(target_naif_id, t, observer_naif_id)
    dist = np.linalg.norm(y[0:3] - target_state[0:3])
    return dist - soi_radius

  soi_event.terminal  = True
  soi_event.direction = -1  # Trigger when entering SOI (distance decreasing)

  # Time span and evaluation points
  t_span   = (t0_et, t0_et + max_time_s)
  n_points = max(1000, int(max_time_s / 60))
  n_points = min(n_points, 50000)
  t_eval   = np.linspace(t0_et, t0_et + max_time_s, n_points)

  # Integrate
  sol = solve_ivp(
    fun          = lambda t, y: _two_body_eom(t, y, gp_central),
    t_span       = t_span,
    y0           = state0,
    method       = 'DOP853',
    rtol         = rtol,
    atol         = atol,
    events       = soi_event,
    t_eval       = t_eval,
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
  sol = solve_ivp(
    fun          = lambda t, y: _two_body_eom(t, y, gp),
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

  sol = solve_ivp(
    fun    = lambda t, y: _two_body_eom(t, y, gp),
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
