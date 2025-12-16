"""
Orbit Propagator
================

Numerical integration of spacecraft equations of motion.
"""
import numpy as np

from datetime          import datetime, timedelta
from typing            import Optional
from pathlib           import Path
from scipy.integrate   import solve_ivp
from scipy.interpolate import interp1d
from sgp4.api          import jday

from src.model.dynamics        import GeneralStateEquationsOfMotion, Acceleration
from src.model.orbit_converter import OrbitConverter
from src.model.constants       import SOLARSYSTEMCONSTANTS
from src.model.time_converter  import utc_to_et
from src.model.frame_converter import FrameConverter
from src.utility.tle_helper    import modify_tle_bstar, get_tle_satellite_and_tle_epoch
from src.utility.time_helper   import format_time_offset


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

    # FrameConverter expects (3, N) for vectorized input.
    teme_pos_vec_arr_T = teme_pos_vec_arr.T # (3, N)
    teme_vel_vec_arr_T = teme_vel_vec_arr.T # (3, N)
    
    # Convert to desired frame
    if to_j2000:
      # Convert TEME to J2000/GCRS
      j2000_pos_vec_T, j2000_vel_vec_T = FrameConverter.teme_to_j2000(
        teme_pos_vec_arr_T,
        teme_vel_vec_arr_T,
        jd_arr + fr_arr,
        units_pos = 'm',
        units_vel = 'm/s'
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
      for key in coe_time_series.keys():
        if coe[key] is not None:
          coe_time_series[key][i] = coe[key]
    
    # Return dict result
    return {
      'success'    : True,
      'message'    : 'SGP4 propagation successful',
      'frame'      : frame,
      'time'       : time,
      'state'      : posvel_vec_array,
      'coe'        : coe_time_series,
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
    j2000_pos_vec, j2000_vel_vec = FrameConverter.teme_to_j2000(teme_pos_vec, teme_vel_vec, jd + fr, units_pos='m', units_vel='m/s')
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
  atol                : float                = 1e-12,
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
  
  if get_coe_time_series:
    for i in range(num_steps):
      pos = solution.y[0:3, i]
      vel = solution.y[3:6, i]
      
      coe = OrbitConverter.pv_to_coe(
        pos, vel, gp
      )
      for key in coe_time_series.keys():
        if coe[key] is not None:
          coe_time_series[key][i] = coe[key]
  
  return {
    'success' : solution.success,
    'message' : solution.message,
    'time'    : solution.t,
    'state'   : solution.y,
    'state_f' : solution.y[:, -1],
    'coe'     : coe_time_series,
  }


def run_high_fidelity_propagation(
  initial_state                 : np.ndarray,
  time_o_dt                     : datetime,
  time_f_dt                     : datetime,
  mass                          : float,
  include_drag                  : bool,
  cd                            : float,
  area_drag                     : float,
  cr                            : float,
  area_srp                      : float,
  include_third_body            : bool,
  third_bodies_list             : list,
  include_gravity_harmonics     : bool,
  gravity_harmonics_list        : list,
  include_srp                   : bool,
  spice_kernels_folderpath      : Path,
  result_jpl_horizons_ephemeris : Optional[dict],
  compare_jpl_horizons          : bool,
) -> dict:
  """
  Configure and run the high-fidelity numerical propagator.
  
  Input:
  ------
    initial_state : np.ndarray
      Initial state vector [pos, vel] in meters and m/s.
    time_o_dt : datetime
      Initial time as datetime object.
    time_f_dt : datetime
      Final time as datetime object.
    mass : float
      Spacecraft mass [kg].
    include_drag : bool
      Flag to enable atmospheric drag.
    cd : float
      Drag coefficient.
    area_drag : float
      Cross-sectional area for drag [m²].
    cr : float
      Radiation pressure coefficient.
    area_srp : float
      Cross-sectional area for SRP [m²].
    include_third_body : bool
      Flag to enable third-body gravity.
    third_bodies_list : list
      List of third bodies (e.g., ['SUN', 'MOON']).
    include_gravity_harmonics : bool
      Flag to enable gravity harmonics.
    gravity_harmonics_list : list
      List of gravity harmonics (e.g., ['J2', 'J3', 'J4']).
    include_srp : bool
      Flag to enable solar radiation pressure.
    spice_kernels_folderpath : Path
      Path to SPICE kernels folder.
    result_jpl_horizons_ephemeris : dict | None
      JPL Horizons ephemeris result for comparison.
    compare_jpl_horizons : bool
      Flag to enable comparison with Horizons.
      
  Output:
  -------
    result : dict
      Dictionary containing propagation results.
  """
  # Print header
  print("\nHigh-Fidelity Model")

  # Calculate Ephemeris Times (ET) for integration
  time_et_o = utc_to_et(time_o_dt)
  time_et_f = utc_to_et(time_f_dt)
  delta_time_s = (time_f_dt - time_o_dt).total_seconds()

  # Determine active zonal harmonics
  j2_val = 0.0
  j3_val = 0.0
  j4_val = 0.0
  active_harmonics = []
  if include_gravity_harmonics:
    if 'J2' in gravity_harmonics_list:
      j2_val = SOLARSYSTEMCONSTANTS.EARTH.J2
      active_harmonics.append('J2')
    if 'J3' in gravity_harmonics_list:
      j3_val = SOLARSYSTEMCONSTANTS.EARTH.J3
      active_harmonics.append('J3')
    if 'J4' in gravity_harmonics_list:
      j4_val = SOLARSYSTEMCONSTANTS.EARTH.J4
      active_harmonics.append('J4')

  # Print configuration
  print(f"  Configuration")
  print(f"    Timespan")
  print(f"      Initial  : {time_o_dt} UTC / {time_et_o:.6f} ET")
  print(f"      Final    : {time_f_dt} UTC / {time_et_f:.6f} ET")
  print(f"      Duration : {delta_time_s} s")
  print(f"    Forces")
  print(f"      Gravity")
  print(f"        Earth")
  print(f"          Two-Body Point Mass")
  if include_gravity_harmonics:
    print(f"          Zonal Harmonics : {', '.join(active_harmonics)}")
  else:
    print(f"          Zonal Harmonics : None")
  
  if include_third_body:
    print(f"        Third-Body")
    print(f"          Bodies    : {', '.join(third_bodies_list)}")
    print(f"          Ephemeris : SPICE")
  
  if include_drag:
    print(f"      Atmospheric Drag")
    print(f"        Model      : Exponential Atmosphere")
    print(f"        Parameters : Cd={cd}, Area_Drag={area_drag} m², Mass={mass} kg")
  
  if include_srp:
    print(f"      Solar Radiation Pressure")
    print(f"        Model      : Cylindrical Shadow (Spherical Earth)")
    print(f"        Parameters : Cr={cr}, Area_SRP={area_srp} m²")

  # Initialize acceleration model
  acceleration_model = Acceleration(
    gp                      = SOLARSYSTEMCONSTANTS.EARTH.GP,
    j2                      = j2_val,
    j3                      = j3_val,
    j4                      = j4_val,
    pos_ref                 = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR,
    mass                    = mass,
    enable_drag             = include_drag,
    cd                      = cd,
    area_drag               = area_drag,
    enable_third_body       = include_third_body,
    third_body_bodies       = third_bodies_list,
    spice_kernel_folderpath = str(spice_kernels_folderpath),
    enable_srp              = include_srp,
    cr                      = cr,
    area_srp                = area_srp,
  )
  
  # Get orbital period for grid density
  period = OrbitConverter.pv_to_period(
    initial_state[0:3],
    initial_state[3:6],
    SOLARSYSTEMCONSTANTS.EARTH.GP
  )
  
  if np.isfinite(period):
    # Elliptical orbit
    # Determine number of points (100 points per period)
    points_per_period = 100
    num_periods       = abs(delta_time_s) / period
    num_grid_points   = int(num_periods * points_per_period)
    
    # Ensure reasonable limits
    if num_grid_points < 100:
      num_grid_points = 100
    # Cap at reasonable max to prevent memory issues for long propagations
    if num_grid_points > 100000:
      num_grid_points = 100000
      print(f"    [WARNING] Grid points capped at {num_grid_points}")
      
  else:
    # Hyperbolic or parabolic - fallback to fixed number
    num_grid_points = 1000

  # Create equal-spaced time grid (in ET)
  t_eval_grid = np.linspace(time_et_o, time_et_f, num_grid_points)

  print("    Numerical Integration")
  print(f"      Method     : DOP853")
  print(f"      Tolerances : rtol=1e-12, atol=1e-12")
  print(f"      Grid       : {len(t_eval_grid)} points (equal-spaced)")

  print("\n  Compute")
  print("    Numerical Integration Running ... ", end='', flush=True)

  # Propagate on equal-spaced grid with dense output for interpolation
  result_high_fidelity = propagate_state_numerical_integration(
    initial_state       = initial_state,
    time_o              = time_et_o,
    time_f              = time_et_f,
    dynamics            = acceleration_model,
    method              = 'DOP853',
    rtol                = 1e-12,
    atol                = 1e-12,
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
      for i in range(len(ephem_times_et)):
        coe = OrbitConverter.pv_to_coe(
          state_at_ephem[0:3, i],
          state_at_ephem[3:6, i],
          SOLARSYSTEMCONSTANTS.EARTH.GP
        )
        for key in coe_at_ephem.keys():
          if coe[key] is not None:
            coe_at_ephem[key][i] = coe[key]
      
      # Store ephemeris-time results
      result_high_fidelity['at_ephem_times'] = {
        'plot_time_s' : ephem_times_s,
        'integ_time_et' : ephem_times_et,
        'state' : state_at_ephem,
        'coe' : coe_at_ephem,
      }
      
      print("Complete")
  else:
    print(f"  Propagation failed: {result_high_fidelity['message']}")
  
  return result_high_fidelity


def run_sgp4_propagation(
  result_jpl_horizons_ephemeris : Optional[dict],
  tle_line_1                    : str,
  tle_line_2                    : str,
  time_o_dt                     : datetime,
  time_f_dt                     : datetime,
  compare_jpl_horizons          : bool,
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
    time_o_dt : datetime
      Initial time as datetime object.
    time_f_dt : datetime
      Final time as datetime object.
    compare_jpl_horizons : bool
      Flag to enable Horizons comparison.
      
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
  time_offset_o_s = (time_o_dt - tle_epoch_dt).total_seconds()
  time_offset_f_s = (time_f_dt - tle_epoch_dt).total_seconds()
  delta_time_s = time_offset_f_s - time_offset_o_s
  
  # Create equal-spaced time grid (seconds from TLE epoch)
  num_grid_points = 1000
  sgp4_times_grid = np.linspace(time_offset_o_s, time_offset_f_s, num_grid_points)

  # Display propagation info
  print(f"  Configuration")
  print(f"    Timespan")
  print(f"      Initial  : {time_o_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
  print(f"      Final    : {time_f_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
  print(f"      Duration : {delta_time_s:.1f} s")
  print(f"      Grid     : {len(sgp4_times_grid)} points (equal-spaced)")

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
        'plot_time_s' : ephem_times_s,
        'integ_time_s' : ephemeris_times,
        'state' : result_sgp4_at_ephem['state'],
        'coe' : result_sgp4_at_ephem['coe'],
      }
      print("Complete")
    else:
      print(f"Failed: {result_sgp4_at_ephem['message']}")
  
  return result_sgp4


def run_propagations(
  initial_state                 : np.ndarray,
  time_o_dt                     : datetime,
  time_f_dt                     : datetime,
  mass                          : float,
  include_drag                  : bool,
  compare_tle                   : bool,
  compare_jpl_horizons          : bool,
  cd                            : float,
  area_drag                     : float,
  cr                            : float,
  area_srp                      : float,
  include_third_body            : bool,
  third_bodies_list             : list,
  include_gravity_harmonics     : bool,
  gravity_harmonics_list        : list,
  include_srp                   : bool,
  spice_kernels_folderpath      : Path,
  result_jpl_horizons_ephemeris : Optional[dict],
  tle_line_1                    : str,
  tle_line_2                    : str,
) -> tuple[dict, Optional[dict]]:
  """
  Run high-fidelity and SGP4 propagations.
  
  Input:
  ------
    initial_state : np.ndarray
      Initial state vector [pos, vel] in meters and m/s.
    time_o_dt : datetime
      Initial time as datetime object.
    time_f_dt : datetime
      Final time as datetime object.
    mass : float
      Spacecraft mass [kg].
    include_drag : bool
      Flag to enable atmospheric drag.
    compare_tle : bool
      Flag to enable TLE/SGP4 comparison.
    compare_jpl_horizons : bool
      Flag to enable Horizons comparison.
    cd : float
      Drag coefficient.
    area_drag : float
      Cross-sectional area for drag [m²].
    cr : float
      Radiation pressure coefficient.
    area_srp : float
      Cross-sectional area for SRP [m²].
    include_third_body : bool
      Flag to enable third-body gravity.
    third_bodies_list : list
      List of third bodies (e.g., ['SUN', 'MOON']).
    include_gravity_harmonics : bool
      Flag to enable gravity harmonics.
    gravity_harmonics_list : list
      List of gravity harmonics (e.g., ['J2', 'J3', 'J4']).
    include_srp : bool
      Flag to enable solar radiation pressure.
    spice_kernels_folderpath : Path
      Path to SPICE kernels folder.
    result_jpl_horizons_ephemeris : dict | None
      JPL Horizons ephemeris result for comparison.
    tle_line_1 : str
      First line of TLE.
    tle_line_2 : str
      Second line of TLE.
      
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
    time_o_dt                     = time_o_dt,
    time_f_dt                     = time_f_dt,
    mass                          = mass,
    include_drag                  = include_drag,
    cd                            = cd,
    area_drag                     = area_drag,
    cr                            = cr,
    area_srp                      = area_srp,
    include_third_body            = include_third_body,
    third_bodies_list             = third_bodies_list,
    include_gravity_harmonics     = include_gravity_harmonics,
    gravity_harmonics_list        = gravity_harmonics_list,
    include_srp                   = include_srp,
    spice_kernels_folderpath      = spice_kernels_folderpath,
    result_jpl_horizons_ephemeris = result_jpl_horizons_ephemeris,
    compare_jpl_horizons          = compare_jpl_horizons,
  )
  
  # SGP4 propagation (if requested)
  result_sgp4 = None
  if compare_tle:
    result_sgp4 = run_sgp4_propagation(
      result_jpl_horizons_ephemeris = result_jpl_horizons_ephemeris,
      tle_line_1                    = tle_line_1,
      tle_line_2                    = tle_line_2,
      time_o_dt                     = time_o_dt,
      time_f_dt                     = time_f_dt,
      compare_jpl_horizons          = compare_jpl_horizons,
    )

  return result_high_fidelity, result_sgp4

