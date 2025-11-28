"""
Orbit Propagator
================

Numerical integration of spacecraft equations of motion.
"""
import math
import numpy as np

from datetime        import datetime, timedelta
from typing          import Optional
from pathlib         import Path
from scipy.integrate import solve_ivp
from sgp4.api        import Satrec, jday

from src.model.dynamics        import GeneralStateEquationsOfMotion, Acceleration, OrbitConverter
from src.model.constants       import PHYSICALCONSTANTS
from src.model.time_converter  import utc_to_et
from src.model.frame_converter import FrameConverter
from src.utility.tle_helper    import modify_tle_bstar, get_tle_satellite_and_tle_epoch


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
    dict
      Result dictionary with 'success', 'state', 'time', 'message'.
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
      'argp' : np.zeros(num_points),
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
        gp = PHYSICALCONSTANTS.EARTH.GP,
      )
      for key in coe_time_series.keys():
        if coe[key] is not None:
          coe_time_series[key][i] = coe[key]
    
    # Return dict result
    return {
      'success' : True,
      'message' : 'SGP4 propagation successful',
      'frame'   : frame,
      'time'    : time,
      'state'   : posvel_vec_array,
      'coe'     : coe_time_series,
    }
  except Exception as e:
    # Catch all exceptions and return failure
    return {
      'success' : False,
      'message' : str(e),
      'frame'   : frame,
      'time'    : [],
      'state'   : [],
      'coe'     : [],
    }


def get_tle_initial_state(
  tle_line_1    : str,
  tle_line_2    : str,
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
    np.ndarray
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
  gp                  : float                = PHYSICALCONSTANTS.EARTH.GP,
) -> dict:
  """
  Propagate an orbit from initial cartesian state.
  
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
      If None, uses dynamics.gravity.two_body.gp.
  
  Output:
  -------
    dict
      Dictionary containing:
      - success : bool - Integration success flag
      - message : str - Status message
      - time : np.ndarray - Time array [s]
      - state : np.ndarray - State history [6 x N]
      - final_state : np.ndarray - Final state vector
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
    'argp' : np.zeros(num_steps),
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


def run_sgp4_propagation(
  result_jpl_horizons_ephemeris : Optional[dict],
  tle_line_1                    : str,
  tle_line_2                    : str,
  desired_time_o_dt             : datetime,
  desired_time_f_dt             : datetime,
  actual_time_o_dt              : datetime,
  actual_time_f_dt              : datetime,
) -> Optional[dict]:
  """
  Propagate SGP4 on the same time grid as the Horizons ephemeris.
  
  This function takes a Horizons result dictionary and propagates a TLE using
  SGP4 at the exact time points from the Horizons data. It then enriches the
  SGP4 result with time arrays and classical orbital elements.
  
  Input:
  ------
    result_jpl_horizons_ephemeris : dict | None
      The processed dictionary from JPL Horizons, containing 'success', 'plot_time_s'.
    tle_line_1 : str
      The first line of the TLE.
    tle_line_2 : str
      The second line of the TLE.
    desired_time_o_dt : datetime
      Desired initial datetime.
    desired_time_f_dt : datetime
      Desired final datetime.
    actual_time_o_dt : datetime
      Actual initial datetime (from Horizons or Desired).
    actual_time_f_dt : datetime
      Actual final datetime (from Horizons or Desired).
      
  Output:
  -------
    dict | None
      An enriched dictionary with SGP4 results, or None if propagation fails.
  """
  print("\nSGP4 Model")

  # Propagate SGP4 at Horizons time points for direct comparison
  if not (result_jpl_horizons_ephemeris and result_jpl_horizons_ephemeris.get('success')):
    return None
  
  # Extract TLE epoch to calculate time offset
  tle_epoch_dt, _ = get_tle_satellite_and_tle_epoch(tle_line_1, tle_line_2)

  # Calculate time offset: Horizons start relative to TLE epoch
  time_offset_s = (actual_time_o_dt - tle_epoch_dt).total_seconds()

  # Convert Horizons plot_time_s to integration times for SGP4 (seconds from TLE epoch)
  sgp4_eval_times = result_jpl_horizons_ephemeris['plot_time_s'] + time_offset_s
  
  # Calculate display values
  duration_actual_s  = (actual_time_f_dt - actual_time_o_dt).total_seconds()
  duration_desired_s = (desired_time_f_dt - desired_time_o_dt).total_seconds()
  num_points         = len(sgp4_eval_times)

  # Display propagation info
  print(f"  Configuration")
  print(f"    Timespan")
  print(f"      Desired")
  print(f"        Initial  : {desired_time_o_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC ({utc_to_et(desired_time_o_dt):.6f} ET)")
  print(f"        Final    : {desired_time_f_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC ({utc_to_et(desired_time_f_dt):.6f} ET)")
  print(f"        Duration : {duration_desired_s:.1f} s")
  print(f"      Actual")
  print(f"        Initial  : {actual_time_o_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC ({utc_to_et(actual_time_o_dt):.6f} ET)")
  print(f"        Final    : {actual_time_f_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC ({utc_to_et(actual_time_f_dt):.6f} ET)")
  print(f"        Duration : {duration_actual_s:.1f} s")
  print(f"        Grid     : {num_points} points")

  print("\n  Compute")
  print("    Numerical Integration Running ... ", end='', flush=True)
  result_sgp4 = propagate_tle(
    tle_line_1 = tle_line_1,
    tle_line_2 = tle_line_2,
    to_j2000   = True,
    time_eval  = sgp4_eval_times,
  )
  print("Complete")
  
  if not result_sgp4['success']:
    print(f"  SGP4 propagation at Horizons times failed: {result_sgp4['message']}")
    return None

  # Store integration time (seconds from TLE epoch)
  result_sgp4['integ_time_s'] = result_sgp4['time']
  
  # Create plotting time array (seconds from Actual start time)
  result_sgp4['plot_time_s'] = result_jpl_horizons_ephemeris['plot_time_s']
  
  # Compute COEs for SGP4 data
  num_points_sgp4 = result_sgp4['state'].shape[1]
  result_sgp4['coe'] = {
    'sma'  : np.zeros(num_points_sgp4),
    'ecc'  : np.zeros(num_points_sgp4),
    'inc'  : np.zeros(num_points_sgp4),
    'raan' : np.zeros(num_points_sgp4),
    'argp' : np.zeros(num_points_sgp4),
    'ma'   : np.zeros(num_points_sgp4),
    'ta'   : np.zeros(num_points_sgp4),
    'ea'   : np.zeros(num_points_sgp4),
  }
  
  for i in range(num_points_sgp4):
    coe = OrbitConverter.pv_to_coe(
      result_sgp4['state'][0:3, i],
      result_sgp4['state'][3:6, i],
      PHYSICALCONSTANTS.EARTH.GP
    )
    for key in result_sgp4['coe'].keys():
      result_sgp4['coe'][key][i] = coe[key]
  
  # Calculate display values
  duration_actual_s  = result_jpl_horizons_ephemeris['plot_time_s'][-1]
  grid_end_dt        = actual_time_o_dt + timedelta(seconds=duration_actual_s)
  duration_desired_s = (desired_time_f_dt - desired_time_o_dt).total_seconds()
  
  return result_sgp4


def run_high_fidelity_propagation(
  initial_state                 : np.ndarray,
  desired_time_o_dt             : datetime,
  desired_time_f_dt             : datetime,
  actual_time_o_dt              : datetime,
  actual_time_f_dt              : datetime,
  mass                          : float,
  include_drag                  : bool,
  cd                            : float,
  area_drag                     : float,
  cr                            : float,
  area_srp                      : float,
  use_spice                     : bool,
  include_third_body            : bool,
  third_bodies_list             : list,
  include_zonal_harmonics       : bool,
  zonal_harmonics_list          : list,
  include_srp                   : bool,
  spice_kernels_folderpath      : Path,
  result_jpl_horizons_ephemeris : Optional[dict],
) -> dict:
  """
  Configure and run the high-fidelity numerical propagator.
  
  Input:
  ------
    initial_state : np.ndarray
      Initial state vector [x, y, z, vx, vy, vz].
    desired_time_o_dt : datetime
      Desired initial datetime.
    desired_time_f_dt : datetime
      Desired final datetime.
    actual_time_o_dt : datetime
      Actual initial datetime (from Horizons or Desired).
    actual_time_f_dt : datetime
      Actual final datetime (from Horizons or Desired).
    mass : float
      Satellite mass [kg].
    include_drag : bool
      Whether to enable Atmospheric Drag.
    cd : float
      Drag coefficient.
    area_drag : float
      Drag area [m²].
    cr : float
      Reflectivity coefficient.
    area_srp : float
      SRP area [m²].
    use_spice : bool
      Whether to use SPICE for third-body ephemerides.
    include_third_body : bool
      Whether to enable third-body gravity.
    third_bodies_list : list
      List of third bodies to include.
    include_zonal_harmonics : bool
      Whether to enable zonal harmonics.
    zonal_harmonics_list : list
      List of zonal harmonics to include.
    include_srp : bool
      Whether to enable Solar Radiation Pressure.
    spice_kernels_folderpath : Path
      Path to SPICE kernels folder.
    result_jpl_horizons_ephemeris : dict | None
      Result from Horizons loader (used for time grid).
      
  Output:
  -------
    dict
      Propagation result dictionary.
  """

  # Display configuration
  print("\nHigh-Fidelity Model")

  # Calculate Ephemeris Times (ET) for integration
  time_et_o_actual = utc_to_et(actual_time_o_dt)
  time_et_f_actual = utc_to_et(actual_time_f_dt)
  
  # Calculate ETs for display
  time_et_o_desired = utc_to_et(desired_time_o_dt)
  time_et_f_desired = utc_to_et(desired_time_f_dt)

  # Determine active zonal harmonics
  j2_val = 0.0
  j3_val = 0.0
  j4_val = 0.0
  active_harmonics = []
  if include_zonal_harmonics:
    if 'J2' in zonal_harmonics_list:
      j2_val = PHYSICALCONSTANTS.EARTH.J2
      active_harmonics.append('J2')
    if 'J3' in zonal_harmonics_list:
      j3_val = PHYSICALCONSTANTS.EARTH.J3
      active_harmonics.append('J3')
    if 'J4' in zonal_harmonics_list:
      j4_val = PHYSICALCONSTANTS.EARTH.J4
      active_harmonics.append('J4')

  # Calculate duration and grid info
  delta_time  = (desired_time_f_dt - desired_time_o_dt).total_seconds()
  grid_points = 0
  if result_jpl_horizons_ephemeris and result_jpl_horizons_ephemeris.get('success'):
    grid_points = len(result_jpl_horizons_ephemeris['plot_time_s'])

  # Set up high-fidelity dynamics model
  print("  Configuration")
  print( "    Timespan")
  print(f"      Desired")
  print(f"        Initial  : {desired_time_o_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC ({time_et_o_desired:.6f} ET)")
  print(f"        Final    : {desired_time_f_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC ({time_et_f_desired:.6f} ET)")
  print(f"        Duration : {delta_time:.1f} s")
  
  if result_jpl_horizons_ephemeris and result_jpl_horizons_ephemeris.get('success'):
    duration_actual = (actual_time_f_dt - actual_time_o_dt).total_seconds()
    print(f"      Actual")
    print(f"        Initial  : {actual_time_o_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC ({time_et_o_actual:.6f} ET)")
    print(f"        Final    : {actual_time_f_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC ({time_et_f_actual:.6f} ET)")
    print(f"        Duration : {duration_actual:.1f} s")
    print(f"        Grid     : {grid_points} points")
  
  print("    Forces")
  print("      Gravity")
  print("        Earth")
  print("          Two-Body Point Mass")
  if active_harmonics:
    print(f"          Zonal Harmonics : {', '.join(active_harmonics)}")
  else:
    print("          Zonal Harmonics : None")
  
  if include_third_body:
    print("        Third-Body")
    print(f"          Bodies    : {', '.join(third_bodies_list)}")
    if use_spice:
      print("          Ephemeris : SPICE (High Accuracy)")
      print(f"          Note      : SPICE kernels loaded for third-body ephemerides.")
    else:
      print("          Ephemeris : Analytical (Approximate)")
      
  print("      Atmospheric Drag")
  if include_drag:
    print( "        Model      : Exponential Atmosphere")
    print(f"        Parameters : Cd={cd}, Area_Drag={area_drag} m², Mass={mass} kg")
  else:
    print("        Model      : None")

  if include_srp:
    print("      Solar Radiation Pressure")
    print( "        Model      : Conical Shadow (Spherical Earth)")
    print(f"        Parameters : Cr={cr}, Area_SRP={area_srp} m²")
  
  # Define acceleration model
  acceleration = Acceleration(
    gp                      = PHYSICALCONSTANTS.EARTH.GP,
    j2                      = j2_val,
    j3                      = j3_val,
    j4                      = j4_val,
    pos_ref                 = PHYSICALCONSTANTS.EARTH.RADIUS.EQUATOR,
    mass                    = mass,
    enable_drag             = include_drag,
    cd                      = cd,
    area_drag               = area_drag,
    enable_srp              = include_srp,
    cr                      = cr,
    area_srp                = area_srp,
    enable_third_body       = include_third_body,
    third_body_use_spice    = use_spice,
    third_body_bodies       = third_bodies_list,
    spice_kernel_folderpath = str(spice_kernels_folderpath),
  )
  
  # Propagate with high-fidelity model: use Horizons time grid
  if result_jpl_horizons_ephemeris and result_jpl_horizons_ephemeris['success']:
    # Convert Horizons plot_time_s (seconds from ACTUAL start) to ET
    horizons_et_times = result_jpl_horizons_ephemeris['plot_time_s'] + time_et_o_actual
    
    # Update integration start/end times to match grid exactly (avoids floating point errors)
    time_et_o_actual = horizons_et_times[0]
    time_et_f_actual = horizons_et_times[-1]
  
    # Print numerical integration settings
    print("    Numerical Integration")
    print(f"      Method     : DOP853")
    print(f"      Tolerances : rtol=1e-12, atol=1e-12")

    # Print completion message
    print("\n  Compute")
    print("    Numerical Integration Running ... ", end='', flush=True)

    # Propagate
    result_high_fidelity = propagate_state_numerical_integration(
      initial_state       = initial_state,
      time_o              = time_et_o_actual, # Use Actual Start ET
      time_f              = time_et_f_actual, # Use Actual End ET
      dynamics            = acceleration,
      method              = 'DOP853',
      rtol                = 1e-12,
      atol                = 1e-12,
      dense_output        = True,  # Enable dense output for exact time evaluation
      t_eval              = horizons_et_times,  # Evaluate at Horizons times (Actual ET)
      get_coe_time_series = True,
      gp                  = PHYSICALCONSTANTS.EARTH.GP,
    )

    # Print completion message
    print("Complete")
  else:
    # If Horizons data is not available, error analysis is not possible.
    raise RuntimeError("Horizons ephemeris is required for high-fidelity propagation and error analysis, but it failed to load.")
  
  if result_high_fidelity['success']:
    # Store integration time (ET)
    result_high_fidelity['integ_time_s'] = result_high_fidelity['time']
    
    # Create plotting time array (seconds from ACTUAL start time, to match Horizons plot_time_s)
    result_high_fidelity['plot_time_s'] = result_high_fidelity['time'] - time_et_o_actual
  else:
    # Print error message
    print(f"  Propagation failed: {result_high_fidelity['message']}")
  
  return result_high_fidelity


def run_propagations(
  initial_state                 : np.ndarray,
  desired_time_o_dt             : datetime,
  desired_time_f_dt             : datetime,
  actual_time_o_dt              : datetime,
  actual_time_f_dt              : datetime,
  mass                          : float,
  include_drag                  : bool,
  cd                            : float,
  area_drag                     : float,
  cr                            : float,
  area_srp                      : float,
  use_spice                     : bool,
  include_third_body            : bool,
  third_bodies_list             : list,
  include_zonal_harmonics       : bool,
  zonal_harmonics_list          : list,
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
      Initial state vector.
    desired_time_o_dt : datetime
      Desired initial time as a datetime.
    desired_time_f_dt : datetime
      Desired final time as a datetime.
    actual_time_o_dt : datetime
      Actual initial time as a datetime.
    actual_time_f_dt : datetime
      Actual final time as a datetime.
    mass : float
      Satellite mass.
    include_drag : bool
      Whether to enable Atmospheric Drag.
    cd : float
      Drag coefficient.
    area_drag : float
      Drag area.
    cr : float
      Reflectivity coefficient.
    area_srp : float
      SRP area.
    use_spice : bool
      Whether to use SPICE.
    include_third_body : bool
      Whether to enable third-body gravity.
    third_bodies_list : list
      List of third bodies to include.
    include_zonal_harmonics : bool
      Whether to enable zonal harmonics.
    zonal_harmonics_list : list
      List of zonal harmonics to include.
    include_srp : bool
      Whether to enable SRP.
    spice_kernels_folderpath : Path
      Path to SPICE kernels.
    result_jpl_horizons_ephemeris : dict | None
      JPL Horizons ephemeris result.
    tle_line_1 : str
      TLE line 1.
    tle_line_2 : str
      TLE line 2.
      
  Output:
  -------
    tuple[dict, dict | None]
      Tuple containing (result_high_fidelity, result_sgp4_at_horizons).
  """
  # Propagate: run high-fidelity propagation at Horizons time points for comparison
  result_high_fidelity = run_high_fidelity_propagation(
    initial_state                 = initial_state,
    desired_time_o_dt             = desired_time_o_dt,
    desired_time_f_dt             = desired_time_f_dt,
    actual_time_o_dt              = actual_time_o_dt,
    actual_time_f_dt              = actual_time_f_dt,
    mass                          = mass,
    include_drag                  = include_drag,
    cd                            = cd,
    area_drag                     = area_drag,
    cr                            = cr,
    area_srp                      = area_srp,
    use_spice                     = use_spice,
    include_third_body            = include_third_body,
    third_bodies_list             = third_bodies_list,
    include_zonal_harmonics       = include_zonal_harmonics,
    zonal_harmonics_list          = zonal_harmonics_list,
    include_srp                   = include_srp,
    spice_kernels_folderpath      = spice_kernels_folderpath,
    result_jpl_horizons_ephemeris = result_jpl_horizons_ephemeris,
  )
  
  # Propagate: run SGP4 at Horizons time points for comparison
  result_sgp4_at_horizons = run_sgp4_propagation(
    result_jpl_horizons_ephemeris = result_jpl_horizons_ephemeris,
    tle_line_1                    = tle_line_1,
    tle_line_2                    = tle_line_2,
    desired_time_o_dt             = desired_time_o_dt,
    desired_time_f_dt             = desired_time_f_dt,
    actual_time_o_dt              = actual_time_o_dt,
    actual_time_f_dt              = actual_time_f_dt,
  )

  return result_high_fidelity, result_sgp4_at_horizons

