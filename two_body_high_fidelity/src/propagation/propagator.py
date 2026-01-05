"""
Orbit Propagator
================

Numerical integration of spacecraft equations of motion.
"""
import numpy as np

from datetime          import datetime, timedelta
from typing            import Optional, TYPE_CHECKING
from types             import SimpleNamespace
from pathlib           import Path
from scipy.integrate   import solve_ivp
from scipy.interpolate import interp1d
from sgp4.api          import jday

from src.model.dynamics        import GeneralStateEquationsOfMotion, Acceleration
from src.model.orbit_converter import OrbitConverter
from src.model.constants       import PRINTFORMATTER, SOLARSYSTEMCONSTANTS
from src.model.time_converter  import utc_to_et
from src.model.frame_converter import VectorConverter
from src.utility.tle_helper    import modify_tle_bstar, get_tle_satellite_and_tle_epoch
from src.schemas.gravity       import GravityModelConfig
from src.schemas.spacecraft    import SpacecraftProperties
from src.schemas.propagation   import PropagationConfig


FORMAT_NUMBER = ">19.12e"


def _get_harmonic_coefficients(
  gravity_harmonics_list : list,
) -> dict:
  """
  Map harmonic names to their coefficient values from constants.
  
  Input:
  ------
    gravity_harmonics_list : list
      List of harmonic names (e.g., ['J2', 'J3', 'C22', 'S22']).
      
  Output:
  -------
    coeffs : dict
      Dictionary mapping parameter names to values.
  """
  coeffs = {
    'j2'  : 0.0,
    'j3'  : 0.0,
    'j4'  : 0.0,
    'j5'  : 0.0,
    'j6'  : 0.0,
    'c21' : 0.0,
    's21' : 0.0,
    'c22' : 0.0,
    's22' : 0.0,
    'c31' : 0.0,
    's31' : 0.0,
    'c32' : 0.0,
    's32' : 0.0,
    'c33' : 0.0,
    's33' : 0.0,
  }
  
  # Map harmonic names to constants
  harmonic_map = {
    'J2'  : ('j2',  SOLARSYSTEMCONSTANTS.EARTH.J2),
    'J3'  : ('j3',  SOLARSYSTEMCONSTANTS.EARTH.J3),
    'J4'  : ('j4',  SOLARSYSTEMCONSTANTS.EARTH.J4),
    'J5'  : ('j5',  SOLARSYSTEMCONSTANTS.EARTH.J5),
    'J6'  : ('j6',  SOLARSYSTEMCONSTANTS.EARTH.J6),
    'C21' : ('c21', SOLARSYSTEMCONSTANTS.EARTH.C21),
    'S21' : ('s21', SOLARSYSTEMCONSTANTS.EARTH.S21),
    'C22' : ('c22', SOLARSYSTEMCONSTANTS.EARTH.C22),
    'S22' : ('s22', SOLARSYSTEMCONSTANTS.EARTH.S22),
    'C31' : ('c31', SOLARSYSTEMCONSTANTS.EARTH.C31),
    'S31' : ('s31', SOLARSYSTEMCONSTANTS.EARTH.S31),
    'C32' : ('c32', SOLARSYSTEMCONSTANTS.EARTH.C32),
    'S32' : ('s32', SOLARSYSTEMCONSTANTS.EARTH.S32),
    'C33' : ('c33', SOLARSYSTEMCONSTANTS.EARTH.C33),
    'S33' : ('s33', SOLARSYSTEMCONSTANTS.EARTH.S33),
  }
  
  for harmonic in gravity_harmonics_list:
    harmonic_upper = harmonic.upper()
    if harmonic_upper in harmonic_map:
      key, value = harmonic_map[harmonic_upper]
      coeffs[key] = value
  
  return coeffs


def propagate_tle(
  tle_line_1   : str,
  tle_line_2   : str,
  time_o       : Optional[float] = None,
  time_f       : Optional[float] = None,
  num_points   : int  = 100,
  time_eval    : Optional[np.ndarray] = None,
  to_j2000     : bool = False,
  disable_drag : bool = False,
) -> dict:
  """
  Propagate orbit from TLE using SGP4.
  
  Input:
  ------
    tle_line_1 : str
      First line of TLE.
    tle_line_2 : str
      Second line of TLE.
    time_o : float, optional
      Initial time in seconds from TLE epoch. Required if time_eval is not provided.
    time_f : float, optional
      Final time in seconds from TLE epoch. Required if time_eval is not provided.
    num_points : int
      Number of output time points (ignored if time_eval is provided).
    time_eval : np.ndarray, optional
      Specific times to evaluate at (in seconds from TLE epoch).
    to_j2000 : bool
      Convert from TEME to J2000 frame.
    disable_drag : bool
      If True, set B* drag term to zero.
      
  Output:
  -------
    result : dict
      Result dictionary with 'success', 'state', 'time', 'message', 'frame', 'coe'.
  """
  # Input validation
  if (time_o is not None or time_f is not None) and time_eval is not None:
    raise ValueError("Cannot provide both time_eval and time_o/time_f.")
  if (time_o is None or time_f is None) and time_eval is None:
    raise ValueError("Either time_eval or both time_o and time_f must be provided.")

  # Modify TLE to disable drag if requested
  if disable_drag:
    tle_line_1 = modify_tle_bstar(tle_line_1, 0.0)
  
  # Determine frame
  frame = 'J2000' if to_j2000 else 'TEME'

  # Propagation vectorized
  try:
    # Create satellite object and extract epoch
    epoch_datetime, satellite = get_tle_satellite_and_tle_epoch(tle_line_1, tle_line_2)
    
    # Generate time array
    if time_eval is not None:
      time       = time_eval
      num_points = len(time_eval)
    else:
      time = np.linspace(time_o, time_f, num_points) # type: ignore
    
    # Initialize arrays
    posvel_vec_array = np.zeros((6, num_points))
    coe_time_series = {
      'sma'  : np.zeros(num_points),
      'ecc'  : np.zeros(num_points),
      'inc'  : np.zeros(num_points),
      'raan' : np.zeros(num_points),
      'aop'  : np.zeros(num_points),
      'ma'   : np.zeros(num_points),
      'ta'   : np.zeros(num_points),
      'ea'   : np.zeros(num_points),
    }
    mee_time_series = {
      'p' : np.zeros(num_points),
      'f' : np.zeros(num_points),
      'g' : np.zeros(num_points),
      'h' : np.zeros(num_points),
      'k' : np.zeros(num_points),
      'L' : np.zeros(num_points),
    }

    # Get epoch Julian date
    jd_epoch, fr_epoch = jday(
      epoch_datetime.year,
      epoch_datetime.month,
      epoch_datetime.day,
      epoch_datetime.hour,
      epoch_datetime.minute,
      epoch_datetime.second + epoch_datetime.microsecond/1e6
    )
    
    # Create arrays for sgp4_array (time is in seconds, 86400 seconds/day)
    jd_arr = np.full(num_points, jd_epoch)
    fr_arr = fr_epoch + time / 86400.0
    
    # Run SGP4 (vectorized)
    error_code_arr, teme_pos_vec_arr, teme_vel_vec_arr = satellite.sgp4_array(jd_arr, fr_arr)
    
    # Check for errors
    if np.any(error_code_arr != 0):
      idx = np.where(error_code_arr != 0)[0][0]
      return {
        'success' : False,
        'message' : f'SGP4 error code: {error_code_arr[idx]} at index {idx}',
        'frame'   : frame,
        'time'    : time[:idx],
        'state'   : np.zeros((6, idx)),
        'coe'     : coe_time_series,
      }

    # Convert SGP4 output km, km/s to m, m/s
    teme_pos_vec_arr *= 1000.0
    teme_vel_vec_arr *= 1000.0

    # VectorConverter expects (3, N) for vectorized input.
    teme_pos_vec_arr_T = teme_pos_vec_arr.T # (3, N)
    teme_vel_vec_arr_T = teme_vel_vec_arr.T # (3, N)
    
    # Convert to desired frame
    if to_j2000:
      # Convert TEME to J2000/GCRS
      j2000_pos_vec_T, j2000_vel_vec_T = VectorConverter.teme_to_j2000(
        teme_pos_vec_arr_T,
        teme_vel_vec_arr_T,
        jd_arr + fr_arr,
        units_pos = 'm',
        units_vel = 'm/s',
      )
      pos_vec_arr = j2000_pos_vec_T
      vel_vec_arr = j2000_vel_vec_T
    else:
      # No frame conversion
      pos_vec_arr = teme_pos_vec_arr_T
      vel_vec_arr = teme_vel_vec_arr_T
      
    # Construct state array (6, N)
    posvel_vec_array = np.vstack((pos_vec_arr, vel_vec_arr))
    
    # Compute osculating elements (Loop)
    for i in range(num_points):
      coe = OrbitConverter.pv_to_coe(
        posvel_vec_array[0:3, i],
        posvel_vec_array[3:6, i],
        gp = SOLARSYSTEMCONSTANTS.EARTH.GP,
      )
      coe_time_series['sma'][i]  = coe.sma
      coe_time_series['ecc'][i]  = coe.ecc
      coe_time_series['inc'][i]  = coe.inc
      coe_time_series['raan'][i] = coe.raan
      coe_time_series['aop'][i]  = coe.aop
      coe_time_series['ta'][i]   = coe.ta if coe.ta is not None else 0.0
      coe_time_series['ea'][i]   = coe.ea if coe.ea is not None else 0.0
      coe_time_series['ma'][i]   = coe.ma if coe.ma is not None else 0.0
      
      # Compute MEE
      mee = OrbitConverter.pv_to_mee(
        posvel_vec_array[0:3, i],
        posvel_vec_array[3:6, i],
        gp = SOLARSYSTEMCONSTANTS.EARTH.GP,
      )
      mee_time_series['p'][i] = mee.p
      mee_time_series['f'][i] = mee.f
      mee_time_series['g'][i] = mee.g
      mee_time_series['h'][i] = mee.h
      mee_time_series['k'][i] = mee.k
      mee_time_series['L'][i] = mee.L
    
    # Return dict result
    return {
      'success'    : True,
      'message'    : 'SGP4 propagation successful',
      'frame'      : frame,
      'time'       : time,
      'state'      : posvel_vec_array,
      'coe'        : coe_time_series,
      'mee'        : mee_time_series,
    }
  except Exception as e:
    # Catch all exceptions and return failure
    return {
      'success'    : False,
      'message'    : str(e),
      'frame'      : frame,
      'time'       : [],
      'state'      : [],
      'coe'        : [],
      'mee'        : [],
    }


def get_tle_initial_state(
  tle_line_1   : str,
  tle_line_2   : str,
  disable_drag : bool = False,
  to_j2000     : bool = True,
) -> np.ndarray:
  """
  Extract initial position and velocity from TLE at epoch.
  
  Input:
  ------
    tle_line_1 : str
      First line of TLE.
    tle_line_2 : str
      Second line of TLE.
    disable_drag : bool
      If True, set B* drag term to zero.
    to_j2000 : bool
      If True, convert from TEME to J2000/GCRS.
  
  Output:
  -------
    initial_state : np.ndarray
      Initial state [x, y, z, vx, vy, vz] in m and m/s.
  """
  # Modify TLE to disable drag if requested
  if disable_drag:
    tle_line_1 = modify_tle_bstar(tle_line_1, 0.0)
  
  # Get satellite object and epoch
  epoch_datetime, satellite = get_tle_satellite_and_tle_epoch(tle_line_1, tle_line_2)
  
  # Get Julian date and fraction
  jd, fr = jday(
    epoch_datetime.year,
    epoch_datetime.month,
    epoch_datetime.day,
    epoch_datetime.hour,
    epoch_datetime.minute, 
    epoch_datetime.second + epoch_datetime.microsecond/1e6,
  )
  
  # Propagate to epoch
  error_code, teme_pos_vec, teme_vel_vec = satellite.sgp4(jd, fr)
  
  if error_code != 0:
    raise ValueError(f'SGP4 error code: {error_code}')
  
  # Transform TEME to J2000/GCRS if requested
  if to_j2000:
    j2000_pos_vec, j2000_vel_vec = VectorConverter.teme_to_j2000(teme_pos_vec, teme_vel_vec, jd + fr, units_pos='m', units_vel='m/s')
    pos_vec = np.array(j2000_pos_vec)
    vel_vec = np.array(j2000_vel_vec)
  else:
    # Convert from km to m and km/s to m/s
    pos_vec = np.array(teme_pos_vec) * 1000.0
    vel_vec = np.array(teme_vel_vec) * 1000.0

  return np.concatenate([pos_vec, vel_vec])


def propagate_state_numerical_integration(
  initial_state       : np.ndarray,
  time_o              : float,
  time_f              : float,
  dynamics            : Acceleration,
  method              : str                  = 'DOP853', # DOP853 RK45
  rtol                : float                = 1e-12,
  atol                : float                = 1e-15,
  dense_output        : bool                 = False,
  t_eval              : Optional[np.ndarray] = None,
  get_coe_time_series : bool                 = False,
  num_points          : Optional[int]        = None,
  gp                  : float                = SOLARSYSTEMCONSTANTS.EARTH.GP,
) -> dict:
  """
  Propagate an orbit from initial cartesian state using numerical integration.
  
  Input:
  ------
    initial_state : np.ndarray
      Initial state vector [pos, vel] in meters and m/s.
    time_o : float
      Initial time [s].
    time_f : float
      Final time [s].
    dynamics : Acceleration
      Acceleration model containing all force models.
    method : str
      Integration method for scipy.solve_ivp (default: 'DOP853').
    rtol : float
      Relative tolerance for integration.
    atol : float
      Absolute tolerance for integration.
    dense_output : bool
      Enable dense output for interpolation.
    t_eval : np.ndarray, optional
      Times at which to store the solution.
    get_coe_time_series : bool
      If True, convert states to classical orbital elements.
    num_points : int, optional
      Number of output points. If None, uses adaptive timesteps from solver.
      If specified, solution is evaluated at uniformly spaced times.
    gp : float, optional
      Gravitational parameter for orbital element conversion [m³/s²].
  
  Output:
  -------
    result : dict
      Dictionary containing:
      - success : bool - Integration success flag
      - message : str - Status message
      - time : np.ndarray - Time array [s]
      - state : np.ndarray - State history [6 x N]
      - state_f : np.ndarray - Final state vector
      - coe : dict - Classical orbital elements time series (if requested)
  """
  # Time span for integration
  time_span = (time_o, time_f)

  # Solve initial value problem
  solution = solve_ivp(
    fun          = GeneralStateEquationsOfMotion(dynamics).state_time_derivative,
    t_span       = time_span,
    y0           = initial_state,
    method       = method,
    rtol         = rtol,
    atol         = atol,
    dense_output = dense_output,
    t_eval       = t_eval,
  )
  
  # If num_points is specified, evaluate solution at uniform time grid
  if num_points is not None:
    t_eval     = np.linspace(time_o, time_f, num_points)
    y_eval     = solution.sol(t_eval)
    solution.t = t_eval
    solution.y = y_eval
  
  # Convert all states to classical orbital elements
  num_steps = solution.y.shape[1]
  coe_time_series = {
    'sma'  : np.zeros(num_steps),
    'ecc'  : np.zeros(num_steps),
    'inc'  : np.zeros(num_steps),
    'raan' : np.zeros(num_steps),
    'aop'  : np.zeros(num_steps),
    'ma'   : np.zeros(num_steps),
    'ta'   : np.zeros(num_steps),
    'ea'   : np.zeros(num_steps),
  }
  mee_time_series = {
    'p' : np.zeros(num_steps),
    'f' : np.zeros(num_steps),
    'g' : np.zeros(num_steps),
    'h' : np.zeros(num_steps),
    'k' : np.zeros(num_steps),
    'L' : np.zeros(num_steps),
  }
  
  if get_coe_time_series:
    for i in range(num_steps):
      pos = solution.y[0:3, i]
      vel = solution.y[3:6, i]
      
      # Compute COE
      coe = OrbitConverter.pv_to_coe(pos, vel, gp)
      coe_time_series['sma'][i]  = coe.sma
      coe_time_series['ecc'][i]  = coe.ecc
      coe_time_series['inc'][i]  = coe.inc
      coe_time_series['raan'][i] = coe.raan
      coe_time_series['aop'][i]  = coe.aop
      coe_time_series['ma'][i]   = coe.ma
      coe_time_series['ta'][i]   = coe.ta
      coe_time_series['ea'][i]   = coe.ea
      
      # Compute MEE
      mee = OrbitConverter.pv_to_mee(pos, vel, gp)
      mee_time_series['p'][i] = mee.p
      mee_time_series['f'][i] = mee.f
      mee_time_series['g'][i] = mee.g
      mee_time_series['h'][i] = mee.h
      mee_time_series['k'][i] = mee.k
      mee_time_series['L'][i] = mee.L
  
  return {
    'success' : solution.success,
    'message' : solution.message,
    'time'    : solution.t,
    'state'   : solution.y,
    'state_f' : solution.y[:, -1],
    'coe'     : coe_time_series,
    'mee'     : mee_time_series,
  }


def run_high_fidelity_propagation(
  initial_state                 : np.ndarray,
  propagation_config            : PropagationConfig,
  spacecraft                    : SpacecraftProperties,
  result_jpl_horizons_ephemeris : Optional[dict],
  compare_jpl_horizons          : bool,
  two_body_gravity_model        : GravityModelConfig,
) -> dict:
  """
  Configure and run the high-fidelity numerical propagator.
  
  Input:
  ------
    initial_state : np.ndarray
      Initial state vector [pos, vel] in meters and m/s.
    propagation_config : PropagationConfig
      Propagation time and integration settings.
    spacecraft : SpacecraftProperties
      Spacecraft physical properties and force model settings.
    result_jpl_horizons_ephemeris : dict | None
      JPL Horizons ephemeris result for comparison.
    compare_jpl_horizons : bool
      Flag to enable comparison with Horizons.
    two_body_gravity_model : GravityModelConfig
      Gravity model configuration containing harmonics and third-body settings.
      
  Output:
  -------
    result : dict
      Dictionary containing propagation results.
  """
  # Extract configuration from model
  include_third_body        = two_body_gravity_model.third_body.enabled
  third_bodies_list         = two_body_gravity_model.third_body.bodies
  include_gravity_harmonics = two_body_gravity_model.spherical_harmonics.enabled
  gravity_harmonics_list    = two_body_gravity_model.spherical_harmonics.coefficients

  # Print header
  print("\nHigh-Fidelity Model")

  # Calculate Ephemeris Times (ET) for integration
  time_et_o    = utc_to_et(propagation_config.time_o_dt)
  time_et_f    = utc_to_et(propagation_config.time_f_dt)
  delta_time_s = (propagation_config.time_f_dt - propagation_config.time_o_dt).total_seconds()

  # Extract the actual gravity model from the namespace
  spherical_harmonics_model = None
  if two_body_gravity_model.spherical_harmonics.model is not None:
    spherical_harmonics_model = two_body_gravity_model.spherical_harmonics.model

  # Configure gravity harmonics
  #   - If spherical_harmonics_model is provided (from file or explicit coefficients), we don't use the analytical TwoBodyGravity harmonics
  if spherical_harmonics_model is not None:
    harmonic_coeffs = _get_harmonic_coefficients([])  # All zeros - let spherical_harmonics_model handle it
  else:
    harmonic_coeffs = _get_harmonic_coefficients(gravity_harmonics_list)

  # Print configuration
  print(f"  Configuration")
  print(f"    Timespan")
  print(f"      Initial  : {propagation_config.time_o_dt} UTC / {time_et_o:.6f} ET")
  print(f"      Final    : {propagation_config.time_f_dt} UTC / {time_et_f:.6f} ET")
  print(f"      Duration : {delta_time_s} s")
  print(f"    Forces")
  print(f"      Gravity")
  print(f"        Earth")
  
  # Display gravity model info
  if spherical_harmonics_model is not None:
    # Check if this is an explicit coefficients model (has active_coefficients set)
    if hasattr(spherical_harmonics_model, 'active_coefficients') and spherical_harmonics_model.active_coefficients is not None:
      print(f"          Spherical Harmonics (Explicit Coefficients)")
      # Show which coefficients are active
      active_names = []
      for deg, ord, ctype in sorted(spherical_harmonics_model.active_coefficients):
        if ctype == 'J':
          active_names.append(f"J{deg}")
        elif ctype == 'C':
          active_names.append(f"C{deg}{ord}")
        elif ctype == 'S':
          active_names.append(f"S{deg}{ord}")
      print(f"            Active   : {', '.join(active_names)}")
      print(f"            GP       : {spherical_harmonics_model.gp:{PRINTFORMATTER.SCIENTIFIC_NOTATION}} m³/s²")
      print(f"            Radius   : {spherical_harmonics_model.radius:{PRINTFORMATTER.SCIENTIFIC_NOTATION}} m")
    else:
      print(f"          Spherical Harmonics")
      print(f"            Degree : {two_body_gravity_model.spherical_harmonics.degree}")
      print(f"            Order  : {two_body_gravity_model.spherical_harmonics.order}")
      print(f"            GP     : {two_body_gravity_model.spherical_harmonics.gp:{PRINTFORMATTER.SCIENTIFIC_NOTATION}} m³/s²")
      print(f"            Radius : {two_body_gravity_model.spherical_harmonics.radius:{PRINTFORMATTER.SCIENTIFIC_NOTATION}} m")
  elif include_gravity_harmonics:
    print(f"          Two-Body Point Mass")
    # Separate zonal and tesseral for display
    zonal_harmonics    = [h for h in gravity_harmonics_list if h.startswith('J')]
    tesseral_harmonics = [h for h in gravity_harmonics_list if h.startswith('C') or h.startswith('S')]
    if zonal_harmonics:
      print(f"          Zonal Harmonics    : {', '.join(zonal_harmonics)}")
    else:
      print(f"          Zonal Harmonics    : None")
    if tesseral_harmonics:
      print(f"          Tesseral Harmonics : {', '.join(tesseral_harmonics)}")
  else:
    print(f"          Two-Body Point Mass")
    print(f"          Zonal Harmonics    : None")
  
  if include_third_body:
    print(f"        Third-Body")
    print(f"          Bodies    : {', '.join(third_bodies_list)}")
    print(f"          Ephemeris : SPICE")
  
  if spacecraft.drag.enabled:
    print(f"      Atmospheric Drag")
    print(f"        Model : Exponential Atmosphere")
    print(f"        Coeff : {spacecraft.drag.cd:{PRINTFORMATTER.SCIENTIFIC_NOTATION}}")
    print(f"        Area  : {spacecraft.drag.area:{PRINTFORMATTER.SCIENTIFIC_NOTATION}} m²")
    print(f"        Mass  : {spacecraft.mass:{PRINTFORMATTER.SCIENTIFIC_NOTATION}} kg")
  
  if spacecraft.srp.enabled:
    print(f"      Solar Radiation Pressure")
    print(f"        Model : Spherical Earth & Cylindrical Shadow")
    print(f"        Coeff : {spacecraft.srp.cr:{PRINTFORMATTER.SCIENTIFIC_NOTATION}}")
    print(f"        Area  : {spacecraft.srp.area:{PRINTFORMATTER.SCIENTIFIC_NOTATION}} m²")
    print(f"        Mass  : {spacecraft.mass:{PRINTFORMATTER.SCIENTIFIC_NOTATION}} kg")

  # Initialize acceleration model
  acceleration = Acceleration(
    gp                        = SOLARSYSTEMCONSTANTS.EARTH.GP,
    j2                        = harmonic_coeffs['j2'],
    j3                        = harmonic_coeffs['j3'],
    j4                        = harmonic_coeffs['j4'],
    j5                        = harmonic_coeffs['j5'],
    j6                        = harmonic_coeffs['j6'],
    c21                       = harmonic_coeffs['c21'],
    s21                       = harmonic_coeffs['s21'],
    c22                       = harmonic_coeffs['c22'],
    s22                       = harmonic_coeffs['s22'],
    c31                       = harmonic_coeffs['c31'],
    s31                       = harmonic_coeffs['s31'],
    c32                       = harmonic_coeffs['c32'],
    s32                       = harmonic_coeffs['s32'],
    c33                       = harmonic_coeffs['c33'],
    s33                       = harmonic_coeffs['s33'],
    pos_ref                   = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR,
    mass                      = spacecraft.mass,
    enable_drag               = spacecraft.drag.enabled,
    cd                        = spacecraft.drag.cd,
    area_drag                 = spacecraft.drag.area,
    enable_third_body         = include_third_body,
    third_body_bodies         = third_bodies_list,
    enable_srp                = spacecraft.srp.enabled,
    cr                        = spacecraft.srp.cr,
    area_srp                  = spacecraft.srp.area,
    spherical_harmonics_model = spherical_harmonics_model,
  )
  
  # Get orbital period for grid density
  period = OrbitConverter.pv_to_period(
    initial_state[0:3],
    initial_state[3:6],
    SOLARSYSTEMCONSTANTS.EARTH.GP
  )
  
  if np.isfinite(period):
    # Elliptical orbit
    # Determine number of points
    
    # Calculate eccentricity to determine grid density
    coe = OrbitConverter.pv_to_coe(
      initial_state[0:3],
      initial_state[3:6],
      SOLARSYSTEMCONSTANTS.EARTH.GP
    )
    
    if coe.ecc > 0.1:
      points_per_period = 1000
    else:
      points_per_period = 100

    num_periods       = abs(delta_time_s) / period
    num_grid_points   = int(num_periods * points_per_period)
    
    # Ensure reasonable limits
    if num_grid_points < 1000:
      num_grid_points = 1000
    # Cap at reasonable max to prevent memory issues for long propagations
    if num_grid_points > 1000000:
      num_grid_points = 1000000
      print(f"    [WARNING] Grid points capped at {num_grid_points}")
      
  else:
    # Hyperbolic or parabolic - fallback to fixed number
    num_grid_points = 10000

  # Create equal-spaced time grid (in ET)
  t_eval_grid = np.linspace(time_et_o, time_et_f, num_grid_points)

  print("    Numerical Integration")
  print(f"      Method     : {propagation_config.method}")
  print(f"      Tolerances : rtol={propagation_config.rtol}, atol={propagation_config.atol}")
  print(f"      Grid       : {len(t_eval_grid)} points (equal-spaced)")

  print("\n  Compute")
  print("    Numerical Integration Running ... ", end='', flush=True)

  # Propagate on equal-spaced grid with dense output for interpolation
  result_high_fidelity = propagate_state_numerical_integration(
    initial_state       = initial_state,
    time_o              = time_et_o,
    time_f              = time_et_f,
    dynamics            = acceleration,
    method              = propagation_config.method,
    rtol                = propagation_config.rtol,
    atol                = propagation_config.atol,
    dense_output        = True,
    t_eval              = t_eval_grid,
    get_coe_time_series = True,
    gp                  = SOLARSYSTEMCONSTANTS.EARTH.GP,
  )

  print("Complete")
  
  if result_high_fidelity['success']:
    # Store integration time (ET)
    result_high_fidelity['integ_time_et'] = result_high_fidelity['time']
    # Create plotting time array (seconds from time_o)
    result_high_fidelity['plot_time_s'] = result_high_fidelity['time'] - time_et_o
    
    # If comparing to Horizons, interpolate to ephemeris times and store separately
    if compare_jpl_horizons and result_jpl_horizons_ephemeris and result_jpl_horizons_ephemeris.get('success'):
      ephem_times_s = result_jpl_horizons_ephemeris['plot_time_s']
      ephem_times_et = ephem_times_s + time_et_o
      
      print(f"    Interpolating to {len(ephem_times_et)} ephemeris time points ... ", end='', flush=True)
      
      # Interpolate state to ephemeris times
      state_at_ephem = np.zeros((6, len(ephem_times_et)))
      for i in range(6):
        interpolator = interp1d(
          result_high_fidelity['time'], 
          result_high_fidelity['state'][i, :], 
          kind='cubic', 
          fill_value='extrapolate'
        )
        state_at_ephem[i, :] = interpolator(ephem_times_et)
      
      # Compute COEs at ephemeris times
      coe_at_ephem = {
        'sma'  : np.zeros(len(ephem_times_et)),
        'ecc'  : np.zeros(len(ephem_times_et)),
        'inc'  : np.zeros(len(ephem_times_et)),
        'raan' : np.zeros(len(ephem_times_et)),
        'aop'  : np.zeros(len(ephem_times_et)),
        'ma'   : np.zeros(len(ephem_times_et)),
        'ta'   : np.zeros(len(ephem_times_et)),
        'ea'   : np.zeros(len(ephem_times_et)),
      }
      mee_at_ephem = {
        'p' : np.zeros(len(ephem_times_et)),
        'f' : np.zeros(len(ephem_times_et)),
        'g' : np.zeros(len(ephem_times_et)),
        'h' : np.zeros(len(ephem_times_et)),
        'k' : np.zeros(len(ephem_times_et)),
        'L' : np.zeros(len(ephem_times_et)),
      }
      for i in range(len(ephem_times_et)):
        coe = OrbitConverter.pv_to_coe(
          state_at_ephem[0:3, i],
          state_at_ephem[3:6, i],
          SOLARSYSTEMCONSTANTS.EARTH.GP
        )
        # for key in coe_at_ephem.keys():
        #   if coe[key] is not None:
        #     coe_at_ephem[key][i] = coe[key]
        coe_at_ephem['sma' ][i] = coe.sma
        coe_at_ephem['ecc' ][i] = coe.ecc
        coe_at_ephem['inc' ][i] = coe.inc
        coe_at_ephem['raan'][i] = coe.raan
        coe_at_ephem['aop' ][i] = coe.aop
        coe_at_ephem['ma'  ][i] = coe.ma
        coe_at_ephem['ta'  ][i] = coe.ta
        coe_at_ephem['ea'  ][i] = coe.ea
        
        # Compute MEE
        mee = OrbitConverter.pv_to_mee(
          state_at_ephem[0:3, i],
          state_at_ephem[3:6, i],
          SOLARSYSTEMCONSTANTS.EARTH.GP
        )
        # for key in mee_at_ephem.keys():
        #   mee_at_ephem[key][i] = mee[key]
        mee_at_ephem['p'][i] = mee.p
        mee_at_ephem['f'][i] = mee.f
        mee_at_ephem['g'][i] = mee.g
        mee_at_ephem['h'][i] = mee.h
        mee_at_ephem['k'][i] = mee.k
        mee_at_ephem['L'][i] = mee.L
      
      # Store ephemeris-time results
      result_high_fidelity['at_ephem_times'] = {
        'plot_time_s'   : ephem_times_s,
        'integ_time_et' : ephem_times_et,
        'state'         : state_at_ephem,
        'coe'           : coe_at_ephem,
        'mee'           : mee_at_ephem,
      }
      
      print("Complete")
  else:
    print(f"  Propagation failed: {result_high_fidelity['message']}")
  
  return result_high_fidelity


def run_sgp4_propagation(
  result_jpl_horizons_ephemeris : Optional[dict],
  tle_line_1                    : str,
  tle_line_2                    : str,
  propagation_config            : PropagationConfig,
  compare_jpl_horizons          : bool,
  time_eval_s                   : Optional[np.ndarray] = None,
) -> Optional[dict]:
  """
  Propagate SGP4 on equal-spaced grid.
  
  If comparing to Horizons, also stores results at ephemeris times.
  
  Input:
  ------
    result_jpl_horizons_ephemeris : dict | None
      JPL Horizons ephemeris result for comparison.
    tle_line_1 : str
      First line of TLE.
    tle_line_2 : str
      Second line of TLE.
    propagation_config : PropagationConfig
      Propagation time settings.
    compare_jpl_horizons : bool
      Flag to enable Horizons comparison.
    time_eval_s : np.ndarray | None
      Optional time grid (seconds from time_o) to evaluate SGP4 at.
      If provided, overrides the default 1000-point grid.
      
  Output:
  -------
    result : dict | None
      Result dictionary containing SGP4 propagation results, or None if failed.
  """
  print("\nSGP4 Model")

  if not tle_line_1 or not tle_line_2:
    print("  No TLE available, skipping SGP4 propagation")
    return None

  # Extract TLE epoch
  tle_epoch_dt, _ = get_tle_satellite_and_tle_epoch(tle_line_1, tle_line_2)
  
  # Calculate times relative to TLE epoch
  time_offset_o_s = (propagation_config.time_o_dt - tle_epoch_dt).total_seconds()
  time_offset_f_s = (propagation_config.time_f_dt - tle_epoch_dt).total_seconds()
  delta_time_s = time_offset_f_s - time_offset_o_s
  
  # Create time grid (seconds from TLE epoch)
  if time_eval_s is not None:
    sgp4_times_grid = time_eval_s + time_offset_o_s
    num_grid_points = len(sgp4_times_grid)
    grid_type_str   = f"{num_grid_points} points (from high-fidelity grid)"
  else:
    num_grid_points = 1000
    sgp4_times_grid = np.linspace(time_offset_o_s, time_offset_f_s, num_grid_points)
    grid_type_str   = f"{num_grid_points} points (equal-spaced)"

  # Calculate ET for display
  time_et_o = utc_to_et(propagation_config.time_o_dt)
  time_et_f = utc_to_et(propagation_config.time_f_dt)

  # Display propagation info
  print(f"  Configuration")
  print(f"    Timespan")
  print(f"      Initial  : {propagation_config.time_o_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC / {time_et_o:.6f} ET")
  print(f"      Final    : {propagation_config.time_f_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC / {time_et_f:.6f} ET")
  print(f"      Duration : {delta_time_s:.1f} s")
  print(f"      Grid     : {grid_type_str}")

  print("\n  Compute")
  print("    SGP4 Propagation Running ... ", end='', flush=True)
  
  result_sgp4 = propagate_tle(
    tle_line_1 = tle_line_1,
    tle_line_2 = tle_line_2,
    to_j2000   = True,
    time_eval  = sgp4_times_grid,
  )
  print("Complete")
  
  if not result_sgp4['success']:
    print(f"  SGP4 propagation failed: {result_sgp4['message']}")
    return None

  # Store integration time (seconds from TLE epoch)
  result_sgp4['integ_time_s'] = result_sgp4['time']
  
  # Create plotting time array (seconds from time_o)
  result_sgp4['plot_time_s'] = result_sgp4['time'] - time_offset_o_s
  
  # If comparing to Horizons, also propagate at ephemeris times
  if compare_jpl_horizons and result_jpl_horizons_ephemeris and result_jpl_horizons_ephemeris.get('success'):
    ephem_times_s = result_jpl_horizons_ephemeris['plot_time_s']  # seconds from time_o
    ephem_times_from_tle = ephem_times_s + time_offset_o_s  # seconds from TLE epoch
    
    print(f"    Propagating at {len(ephem_times_from_tle)} ephemeris time points ... ", end='', flush=True)
    
    result_sgp4_at_ephem = propagate_tle(
      tle_line_1 = tle_line_1,
      tle_line_2 = tle_line_2,
      to_j2000   = True,
      time_eval  = ephem_times_from_tle,
    )
    
    if result_sgp4_at_ephem['success']:
      # Store ephemeris-time results
      result_sgp4['at_ephem_times'] = {
        'plot_time_s'  : ephem_times_s,
        'integ_time_s' : ephem_times_from_tle,
        'state'        : result_sgp4_at_ephem['state'],
        'coe'          : result_sgp4_at_ephem['coe'],
        'mee'          : result_sgp4_at_ephem['mee'],
      }
      print("Complete")
    else:
      print(f"Failed: {result_sgp4_at_ephem['message']}")
  
  return result_sgp4


def run_propagations(
  initial_state                 : np.ndarray,
  propagation_config            : PropagationConfig,
  spacecraft                    : SpacecraftProperties,
  compare_tle                   : bool,
  compare_jpl_horizons          : bool,
  result_jpl_horizons_ephemeris : Optional[dict],
  tle_line_1                    : Optional[str],
  tle_line_2                    : Optional[str],
  two_body_gravity_model        : GravityModelConfig,
) -> tuple:
  """
  Run high-fidelity and SGP4 propagations.
  
  Input:
  ------
    initial_state : np.ndarray
      Initial state vector [pos, vel] in meters and m/s.
    propagation_config : PropagationConfig
      Propagation settings.
    spacecraft : SpacecraftProperties
      Spacecraft properties.
    compare_tle : bool
      Flag to enable TLE/SGP4 comparison.
    compare_jpl_horizons : bool
      Flag to enable Horizons comparison.
    result_jpl_horizons_ephemeris : dict | None
      JPL Horizons ephemeris result for comparison.
    tle_line_1 : str
      First line of TLE.
    tle_line_2 : str
      Second line of TLE.
    two_body_gravity_model : GravityModelConfig
      Gravity model configuration.
      
  Output:
  -------
    result_high_fidelity : dict
      High-fidelity propagation result.
    result_sgp4 : dict | None
      SGP4 propagation result or None.
  """
  # High-fidelity propagation
  result_high_fidelity = run_high_fidelity_propagation(
    initial_state                 = initial_state,
    propagation_config            = propagation_config,
    spacecraft                    = spacecraft,
    result_jpl_horizons_ephemeris = result_jpl_horizons_ephemeris,
    compare_jpl_horizons          = compare_jpl_horizons,
    two_body_gravity_model        = two_body_gravity_model,
  )
  
  # SGP4 propagation (if requested)
  result_sgp4 = None
  if compare_tle:
    # Use high-fidelity time grid if available for direct comparison
    time_eval_s = None
    if result_high_fidelity.get('success'):
      time_eval_s = result_high_fidelity.get('plot_time_s')
    
    result_sgp4 = run_sgp4_propagation(
      result_jpl_horizons_ephemeris = result_jpl_horizons_ephemeris,
      tle_line_1                    = tle_line_1,
      tle_line_2                    = tle_line_2,
      propagation_config            = propagation_config,
      compare_jpl_horizons          = compare_jpl_horizons,
      time_eval_s                   = time_eval_s,
    )

  return result_high_fidelity, result_sgp4

