"""
Orbit Propagator
================

Numerical integration of spacecraft equations of motion.
"""
import numpy as np

from datetime          import datetime, timedelta
from typing            import Optional
from scipy.integrate   import solve_ivp
from scipy.interpolate import interp1d
from sgp4.api          import jday

from src.model.dynamics        import GeneralStateEquationsOfMotion, AccelerationSTMDot
from src.model.orbit_converter import OrbitConverter
from src.model.constants       import PRINTFORMATTER, SOLARSYSTEMCONSTANTS, CONVERTER
from src.model.time_converter  import utc_to_et, et_to_utc
from src.model.frame_converter import VectorConverter
from src.utility.tle_helper    import modify_tle_bstar, get_tle_satellite_and_tle_epoch
from src.schemas.gravity       import GravityModelConfig
from src.schemas.spacecraft    import SpacecraftProperties
from src.schemas.propagation   import PropagationConfig, PropagationResult, TimeGrid
from src.schemas.state         import ClassicalOrbitalElements, ModifiedEquinoctialElements


FORMAT_NUMBER = ">19.12e"


def propagate_tle(
  tle_line_1     : str,
  tle_line_2     : str,
  time_o         : Optional[float] = None,
  time_f         : Optional[float] = None,
  num_points     : int  = 100,
  time_eval      : Optional[np.ndarray] = None,
  to_j2000       : bool = False,
  disable_drag   : bool = False,
  initial_dt     : Optional['datetime'] = None,
  final_dt       : Optional['datetime'] = None,
) -> PropagationResult:
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
    initial_dt : datetime, optional
      Initial time as datetime (UTC) for creating TimeGrid.
    final_dt : datetime, optional
      Final time as datetime (UTC) for creating TimeGrid.

  Output:
  -------
    result : PropagationResult
      Result object with 'success', 'state', 'time_grid', 'message', 'coe', 'mee'.
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
    
    # Initialize COE arrays
    coe_sma  = np.zeros(num_points)
    coe_ecc  = np.zeros(num_points)
    coe_inc  = np.zeros(num_points)
    coe_raan = np.zeros(num_points)
    coe_aop  = np.zeros(num_points)
    coe_ma   = np.zeros(num_points)
    coe_ta   = np.zeros(num_points)
    coe_ea   = np.zeros(num_points)

    # Initialize MEE arrays
    mee_p = np.zeros(num_points)
    mee_f = np.zeros(num_points)
    mee_g = np.zeros(num_points)
    mee_h = np.zeros(num_points)
    mee_k = np.zeros(num_points)
    mee_L = np.zeros(num_points)

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
      return PropagationResult(
        success = False,
        message = f'SGP4 error code: {error_code_arr[idx]} at index {idx}',
        state   = np.zeros((6, idx)),
      )

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
      coe_sma[i]  = coe.sma
      coe_ecc[i]  = coe.ecc
      coe_inc[i]  = coe.inc
      coe_raan[i] = coe.raan
      coe_aop[i]  = coe.aop
      coe_ta[i]   = coe.ta if coe.ta is not None else 0.0
      coe_ea[i]   = coe.ea if coe.ea is not None else 0.0
      coe_ma[i]   = coe.ma if coe.ma is not None else 0.0
      
      # Compute MEE
      mee = OrbitConverter.pv_to_mee(
        posvel_vec_array[0:3, i],
        posvel_vec_array[3:6, i],
        gp = SOLARSYSTEMCONSTANTS.EARTH.GP,
      )
      mee_p[i] = mee.p
      mee_f[i] = mee.f
      mee_g[i] = mee.g
      mee_h[i] = mee.h
      mee_k[i] = mee.k
      mee_L[i] = mee.L
    
    # Construct state objects
    coe_time_series = ClassicalOrbitalElements(
      sma=coe_sma, ecc=coe_ecc, inc=coe_inc, raan=coe_raan, aop=coe_aop, ma=coe_ma, ta=coe_ta, ea=coe_ea
    )
    mee_time_series = ModifiedEquinoctialElements(
      p=mee_p, f=mee_f, g=mee_g, h=mee_h, k=mee_k, L=mee_L
    )

    # Create TimeGrid if datetime parameters provided
    time_grid = None
    if initial_dt is not None and final_dt is not None:
      # time array is in seconds from TLE epoch, convert to deltas from initial_dt
      time_offset_o_s = (initial_dt - epoch_datetime).total_seconds()
      deltas = time - time_offset_o_s
      time_grid = TimeGrid(
        initial = initial_dt,
        final   = final_dt,
        deltas  = deltas,
      )

    # Return result object with time_grid
    return PropagationResult(
      success   = True,
      message   = 'SGP4 propagation successful',
      time_grid = time_grid,
      state     = posvel_vec_array,
      coe       = coe_time_series,
      mee       = mee_time_series,
    )
  except Exception as e:
    # Catch all exceptions and return failure
    return PropagationResult(
      success = False,
      message = str(e),
      state   = np.array([]),
    )


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
  initial_dt          : datetime,
  final_dt            : datetime,
  dynamics            : AccelerationSTMDot,
  method              : str                  = 'DOP853', # DOP853 RK45
  rtol                : float                = 1e-12,
  atol                : float                = 1e-15,
  dense_output        : bool                 = False,
  t_eval              : Optional[np.ndarray] = None,
  get_coe_time_series : bool                 = False,
  num_points          : Optional[int]        = None,
  gp                  : float                = SOLARSYSTEMCONSTANTS.EARTH.GP,
) -> PropagationResult:
  """
  Propagate an orbit from initial cartesian state using numerical integration.

  Input:
  ------
    initial_state : np.ndarray
      Initial state vector [pos, vel] in meters and m/s.
    initial_dt : datetime
      Initial time as datetime (UTC).
    final_dt : datetime
      Final time as datetime (UTC).
    dynamics : AccelerationSTMDot
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
      Times at which to store the solution (in ET seconds).
    get_coe_time_series : bool
      If True, convert states to classical orbital elements.
    num_points : int, optional
      Number of output points. If None, uses adaptive timesteps from solver.
      If specified, solution is evaluated at uniformly spaced times.
    gp : float, optional
      Gravitational parameter for orbital element conversion [m³/s²].

  Output:
  -------
    result : PropagationResult
      Object containing:
      - success   : bool - Integration success flag
      - message   : str - Status message
      - time_grid : TimeGrid - Time grid with initial, final, and deltas
      - state     : np.ndarray - State history [6 x N]
      - coe       : ClassicalOrbitalElements - Classical orbital elements time series (if requested)
      - mee       : ModifiedEquinoctialElements - Modified equinoctial elements time series (if requested)
  """
  # Convert datetime to ephemeris time for integration
  time_o = utc_to_et(initial_dt)
  time_f = utc_to_et(final_dt)

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
  
  coe_time_series = None
  mee_time_series = None

  if get_coe_time_series:
    # Initialize arrays
    coe_sma  = np.zeros(num_steps)
    coe_ecc  = np.zeros(num_steps)
    coe_inc  = np.zeros(num_steps)
    coe_raan = np.zeros(num_steps)
    coe_aop  = np.zeros(num_steps)
    coe_ma   = np.zeros(num_steps)
    coe_ta   = np.zeros(num_steps)
    coe_ea   = np.zeros(num_steps)

    mee_p = np.zeros(num_steps)
    mee_f = np.zeros(num_steps)
    mee_g = np.zeros(num_steps)
    mee_h = np.zeros(num_steps)
    mee_k = np.zeros(num_steps)
    mee_L = np.zeros(num_steps)

    for i in range(num_steps):
      # Unpack position and velocity vectors
      pos_vec = solution.y[0:3, i]
      vel_vec = solution.y[3:6, i]
      
      # Compute COE
      coe = OrbitConverter.pv_to_coe(pos_vec, vel_vec, gp)
      coe_sma[i]  = coe.sma
      coe_ecc[i]  = coe.ecc
      coe_inc[i]  = coe.inc
      coe_raan[i] = coe.raan
      coe_aop[i]  = coe.aop
      coe_ma[i]   = coe.ma
      coe_ta[i]   = coe.ta
      coe_ea[i]   = coe.ea
      
      # Compute MEE
      mee = OrbitConverter.pv_to_mee(pos_vec, vel_vec, gp)
      mee_p[i] = mee.p
      mee_f[i] = mee.f
      mee_g[i] = mee.g
      mee_h[i] = mee.h
      mee_k[i] = mee.k
      mee_L[i] = mee.L
    
    # Construct objects
    coe_time_series = ClassicalOrbitalElements(
      sma  = coe_sma,
      ecc  = coe_ecc,
      inc  = coe_inc,
      raan = coe_raan,
      aop  = coe_aop,
      ma   = coe_ma,
      ta   = coe_ta,
      ea   = coe_ea,
    )
    mee_time_series = ModifiedEquinoctialElements(
      p = mee_p,
      f = mee_f,
      g = mee_g,
      h = mee_h,
      k = mee_k,
      L = mee_L,
    )

  # Create TimeGrid
  # solution.t contains the time array (in ET seconds)
  # Convert to deltas from time_o
  deltas = solution.t - time_o
  time_grid = TimeGrid(
    initial = initial_dt,
    final   = final_dt,
    deltas  = deltas,
  )

  return PropagationResult(
    success   = solution.success,
    message   = solution.message,
    time_grid = time_grid,
    state     = solution.y,
    coe       = coe_time_series,
    mee       = mee_time_series,
  )


def run_high_fidelity_propagation(
  initial_state                 : np.ndarray,
  propagation_config            : PropagationConfig,
  spacecraft                    : SpacecraftProperties,
  result_jpl_horizons_ephemeris : Optional[PropagationResult],
  compare_jpl_horizons          : bool,
  two_body_gravity_model        : GravityModelConfig,
) -> PropagationResult:
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
    result_jpl_horizons_ephemeris : PropagationResult | None
      JPL Horizons ephemeris result for comparison.
    compare_jpl_horizons : bool
      Flag to enable comparison with Horizons.
    two_body_gravity_model : GravityModelConfig
      Gravity model configuration containing harmonics and third-body settings.
      
  Output:
  -------
    result : PropagationResult
      Object containing propagation results.
  """
  # Extract configuration from model
  include_third_body        = two_body_gravity_model.third_body.enabled
  third_bodies_list         = two_body_gravity_model.third_body.bodies
  include_gravity_harmonics = two_body_gravity_model.spherical_harmonics.enabled
  gravity_harmonics_list    = two_body_gravity_model.spherical_harmonics.coefficients

  # Print header
  title = "High-Fidelity Model"
  print("\n" + "-" * len(title))
  print(title)
  print("-" * len(title))

  # Calculate Ephemeris Times (ET) for integration
  time_et_o    = utc_to_et(propagation_config.time_o_dt)
  time_et_f    = utc_to_et(propagation_config.time_f_dt)
  delta_time_epoch = (propagation_config.time_f_dt - propagation_config.time_o_dt).total_seconds()

  # Extract the actual gravity model from the namespace
  spherical_harmonics_model = two_body_gravity_model.spherical_harmonics.model

  print()
  print("  Progress")
  print("    Initialize acceleration model")

  # Initialize acceleration model
  acceleration = AccelerationSTMDot(
    gravity_config = two_body_gravity_model,
    spacecraft     = spacecraft,
  )

  print("    Calculate orbital period for grid density")

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

    num_periods       = abs(delta_time_epoch) / period
    num_grid_points   = int(num_periods * points_per_period)

    # Ensure reasonable limits
    if num_grid_points < 1000:
      num_grid_points = 1000
    # Cap at reasonable max to prevent memory issues for long propagations
    if num_grid_points > 1000000:
      num_grid_points = 1000000
      print(f"      [WARNING] Grid points capped at {num_grid_points}")

  else:
    # Hyperbolic or parabolic - fallback to fixed number
    num_grid_points = 10000

  print("    Create equal-spaced time grid")

  # Create equal-spaced time grid (in ET)
  t_eval_grid = np.linspace(time_et_o, time_et_f, num_grid_points)

  print("    Run numerical integration")

  # Propagate on equal-spaced grid with dense output for interpolation
  result_high_fidelity = propagate_state_numerical_integration(
    initial_state       = initial_state,
    initial_dt          = propagation_config.time_o_dt,
    final_dt            = propagation_config.time_f_dt,
    dynamics            = acceleration,
    method              = propagation_config.method,
    rtol                = propagation_config.rtol,
    atol                = propagation_config.atol,
    dense_output        = True,
    t_eval              = t_eval_grid,
    get_coe_time_series = True,
    gp                  = SOLARSYSTEMCONSTANTS.EARTH.GP,
  )

  if result_high_fidelity.success:
    # t_eval_grid is the time array we passed in (absolute ET values)
    time_et_array = t_eval_grid

    # If comparing to Horizons, interpolate to ephemeris times and store separately
    if compare_jpl_horizons and result_jpl_horizons_ephemeris and result_jpl_horizons_ephemeris.success:
      print("    Interpolate to ephemeris time points")

      ephem_times_s = result_jpl_horizons_ephemeris.time_grid.deltas
      ephem_times_et = ephem_times_s + time_et_o

      # Interpolate state to ephemeris times (use time_et_array from above)
      state_at_ephem = np.zeros((6, len(ephem_times_et)))
      for i in range(6):
        interpolator = interp1d(
          time_et_array,
          result_high_fidelity.state[i, :],
          kind='cubic',
          fill_value='extrapolate'
        )
        state_at_ephem[i, :] = interpolator(ephem_times_et)

      # Compute COEs at ephemeris times
      n_ephem = len(ephem_times_et)
      coe_sma  = np.zeros(n_ephem)
      coe_ecc  = np.zeros(n_ephem)
      coe_inc  = np.zeros(n_ephem)
      coe_raan = np.zeros(n_ephem)
      coe_aop  = np.zeros(n_ephem)
      coe_ma   = np.zeros(n_ephem)
      coe_ta   = np.zeros(n_ephem)
      coe_ea   = np.zeros(n_ephem)

      mee_p = np.zeros(n_ephem)
      mee_f = np.zeros(n_ephem)
      mee_g = np.zeros(n_ephem)
      mee_h = np.zeros(n_ephem)
      mee_k = np.zeros(n_ephem)
      mee_L = np.zeros(n_ephem)

      for i in range(len(ephem_times_et)):
        coe = OrbitConverter.pv_to_coe(
          state_at_ephem[0:3, i],
          state_at_ephem[3:6, i],
          SOLARSYSTEMCONSTANTS.EARTH.GP
        )
        coe_sma[i]  = coe.sma
        coe_ecc[i]  = coe.ecc
        coe_inc[i]  = coe.inc
        coe_raan[i] = coe.raan
        coe_aop[i]  = coe.aop
        coe_ma[i]   = coe.ma
        coe_ta[i]   = coe.ta
        coe_ea[i]   = coe.ea

        # Compute MEE
        mee = OrbitConverter.pv_to_mee(
          state_at_ephem[0:3, i],
          state_at_ephem[3:6, i],
          SOLARSYSTEMCONSTANTS.EARTH.GP
        )
        mee_p[i] = mee.p
        mee_f[i] = mee.f
        mee_g[i] = mee.g
        mee_h[i] = mee.h
        mee_k[i] = mee.k
        mee_L[i] = mee.L

      # Construct objects
      coe_at_ephem = ClassicalOrbitalElements(
        sma=coe_sma, ecc=coe_ecc, inc=coe_inc, raan=coe_raan, aop=coe_aop, ma=coe_ma, ta=coe_ta, ea=coe_ea
      )
      mee_at_ephem = ModifiedEquinoctialElements(
        p=mee_p, f=mee_f, g=mee_g, h=mee_h, k=mee_k, L=mee_L
      )

      # Store ephemeris-time results as PropagationResult
      # Create time grid for interpolated ephemeris times
      if result_jpl_horizons_ephemeris.time_grid is not None:
        ephem_time_grid = TimeGrid(
          initial = result_jpl_horizons_ephemeris.time_grid.initial,
          final   = result_jpl_horizons_ephemeris.time_grid.final,
          deltas  = ephem_times_s,
        )
        result_high_fidelity.at_ephem_times = PropagationResult(
          success   = True,
          message   = "Interpolated to ephemeris times",
          time_grid = ephem_time_grid,
          state     = state_at_ephem,
          coe       = coe_at_ephem,
          mee       = mee_at_ephem,
        )

    print()
    print("  Summary")

    # Print configuration
    print(f"    Configuration")
    print(f"      Timespan")
    print(f"        Initial  : {propagation_config.time_o_dt} UTC / {time_et_o:.6f} ET")
    print(f"        Final    : {propagation_config.time_f_dt} UTC / {time_et_f:.6f} ET")
    print(f"        Duration : {delta_time_epoch} s")
    print(f"      Forces")
    print(f"        Gravity")
    print(f"          Earth")

    # Display gravity model info
    if spherical_harmonics_model is not None:
      # Check if this is an explicit coefficients model (has active_coefficients set)
      if hasattr(spherical_harmonics_model, 'active_coefficients') and spherical_harmonics_model.active_coefficients is not None:
        print(f"            Spherical Harmonics (Explicit Coefficients)")
        # Show which coefficients are active
        active_names = []
        for deg, ord, ctype in sorted(spherical_harmonics_model.active_coefficients):
          if ctype == 'J':
            active_names.append(f"J{deg}")
          elif ctype == 'C':
            active_names.append(f"C{deg}{ord}")
          elif ctype == 'S':
            active_names.append(f"S{deg}{ord}")
        print(f"              Active   : {', '.join(active_names)}")
        print(f"              GP       : {spherical_harmonics_model.gp:{PRINTFORMATTER.SCIENTIFIC_NOTATION}} m³/s²")
        print(f"              Radius   : {spherical_harmonics_model.radius:{PRINTFORMATTER.SCIENTIFIC_NOTATION}} m")
      else:
        print(f"            Spherical Harmonics")
        print(f"              Degree : {two_body_gravity_model.spherical_harmonics.degree}")
        print(f"              Order  : {two_body_gravity_model.spherical_harmonics.order}")
        print(f"              GP     : {two_body_gravity_model.spherical_harmonics.gp:{PRINTFORMATTER.SCIENTIFIC_NOTATION}} m³/s²")
        print(f"              Radius : {two_body_gravity_model.spherical_harmonics.radius:{PRINTFORMATTER.SCIENTIFIC_NOTATION}} m")
    elif include_gravity_harmonics:
      print(f"            Two-Body Point Mass")
      # Separate zonal and tesseral for display
      zonal_harmonics    = [h for h in gravity_harmonics_list if h.startswith('J')]
      tesseral_harmonics = [h for h in gravity_harmonics_list if h.startswith('C') or h.startswith('S')]
      if zonal_harmonics:
        print(f"            Zonal Harmonics    : {', '.join(zonal_harmonics)}")
      else:
        print(f"            Zonal Harmonics    : None")
      if tesseral_harmonics:
        print(f"            Tesseral Harmonics : {', '.join(tesseral_harmonics)}")
    else:
      print(f"            Two-Body Point Mass")
      print(f"            Zonal Harmonics    : None")

    if include_third_body:
      print(f"          Third-Body")
      print(f"            Bodies    : {', '.join(third_bodies_list)}")
      print(f"            Ephemeris : SPICE")

    if spacecraft.drag.enabled:
      print(f"        Atmospheric Drag")
      print(f"          Model : Exponential Atmosphere")
      print(f"          Coeff : {spacecraft.drag.cd:{PRINTFORMATTER.SCIENTIFIC_NOTATION}}")
      print(f"          Area  : {spacecraft.drag.area:{PRINTFORMATTER.SCIENTIFIC_NOTATION}} m²")
      print(f"          Mass  : {spacecraft.mass:{PRINTFORMATTER.SCIENTIFIC_NOTATION}} kg")

    if spacecraft.srp.enabled:
      print(f"        Solar Radiation Pressure")
      print(f"          Model : Spherical Earth & Cylindrical Shadow")
      print(f"          Coeff : {spacecraft.srp.cr:{PRINTFORMATTER.SCIENTIFIC_NOTATION}}")
      print(f"          Area  : {spacecraft.srp.area:{PRINTFORMATTER.SCIENTIFIC_NOTATION}} m²")
      print(f"          Mass  : {spacecraft.mass:{PRINTFORMATTER.SCIENTIFIC_NOTATION}} kg")

    print(f"      Numerical Integration")
    print(f"        Method     : {propagation_config.method}")
    print(f"        Tolerances : rtol={propagation_config.rtol}, atol={propagation_config.atol}")
    print(f"        Grid       : {len(t_eval_grid)} points (equal-spaced)")
    print()

    # Print initial state
    print(f"    Initial State")
    print(f"      Epoch : {propagation_config.time_o_dt} UTC / {time_et_o:.6f} ET")
    print(f"      Frame : J2000")
    print(f"      Cartesian State")
    print(f"        Position : {initial_state[0]:>19.12e}  {initial_state[1]:>19.12e}  {initial_state[2]:>19.12e} m")
    print(f"        Velocity : {initial_state[3]:>19.12e}  {initial_state[4]:>19.12e}  {initial_state[5]:>19.12e} m/s")

    # Compute initial COE
    coe_o = OrbitConverter.pv_to_coe(
      initial_state[0:3],
      initial_state[3:6],
      SOLARSYSTEMCONSTANTS.EARTH.GP
    )
    print(f"      Classical Orbital Elements")
    print(f"        SMA  : { coe_o.sma:>19.12e} m")
    print(f"        ECC  : { coe_o.ecc:>19.12e} -")
    print(f"        INC  : { coe_o.inc * CONVERTER.DEG_PER_RAD:>19.12e} deg")
    print(f"        RAAN : {coe_o.raan * CONVERTER.DEG_PER_RAD:>19.12e} deg")
    print(f"        AOP  : { coe_o.aop * CONVERTER.DEG_PER_RAD:>19.12e} deg")
    if coe_o.ta is not None:
      print(f"        TA   : {  coe_o.ta * CONVERTER.DEG_PER_RAD:>19.12e} deg")
    if coe_o.ea is not None:
      print(f"        EA   : {  coe_o.ea * CONVERTER.DEG_PER_RAD:>19.12e} deg")
    if coe_o.ma is not None:
      print(f"        MA   : {  coe_o.ma * CONVERTER.DEG_PER_RAD:>19.12e} deg")
    print()

    # Print final state and orbital elements
    time_et_f_final = result_high_fidelity.time_grid.times_et[-1]
    time_utc_f_str = f"{et_to_utc(time_et_f_final)} UTC / {time_et_f_final:.6f} ET"

    pos_vec_f = result_high_fidelity.state[0:3, -1]
    vel_vec_f = result_high_fidelity.state[3:6, -1]

    coe_f = result_high_fidelity.coe
    sma  = coe_f.sma[-1]
    ecc  = coe_f.ecc[-1]
    inc  = coe_f.inc[-1] * CONVERTER.DEG_PER_RAD
    raan = coe_f.raan[-1] * CONVERTER.DEG_PER_RAD
    aop  = coe_f.aop[-1] * CONVERTER.DEG_PER_RAD
    ta   = coe_f.ta[-1] * CONVERTER.DEG_PER_RAD
    ea   = coe_f.ea[-1] * CONVERTER.DEG_PER_RAD
    ma   = coe_f.ma[-1] * CONVERTER.DEG_PER_RAD

    print(f"    Final State")
    print(f"      Epoch : {time_utc_f_str}")
    print(f"      Frame : J2000")
    print(f"      Cartesian State")
    print(f"        Position : {pos_vec_f[0]:>19.12e}  {pos_vec_f[1]:>19.12e}  {pos_vec_f[2]:>19.12e} m")
    print(f"        Velocity : {vel_vec_f[0]:>19.12e}  {vel_vec_f[1]:>19.12e}  {vel_vec_f[2]:>19.12e} m/s")
    print(f"      Classical Orbital Elements")
    print(f"        SMA  : { sma:>19.12e} m")
    print(f"        ECC  : { ecc:>19.12e} -")
    print(f"        INC  : { inc:>19.12e} deg")
    print(f"        RAAN : {raan:>19.12e} deg")
    print(f"        AOP  : { aop:>19.12e} deg")
    print(f"        TA   : {  ta:>19.12e} deg")
    print(f"        EA   : {  ea:>19.12e} deg")
    print(f"        MA   : {  ma:>19.12e} deg")
  else:
    print(f"\n  [ERROR] Propagation failed: {result_high_fidelity.message}")
  
  return result_high_fidelity


def run_sgp4_propagation(
  result_jpl_horizons_ephemeris : Optional[PropagationResult],
  tle_line_1                    : str,
  tle_line_2                    : str,
  propagation_config            : PropagationConfig,
  compare_jpl_horizons          : bool,
  time_eval_s                   : Optional[np.ndarray] = None,
) -> Optional[PropagationResult]:
  """
  Propagate SGP4 on equal-spaced grid.
  
  If comparing to Horizons, also stores results at ephemeris times.
  
  Input:
  ------
    result_jpl_horizons_ephemeris : PropagationResult | None
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
    result : PropagationResult | None
      Result object containing SGP4 propagation results, or None if failed.
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
  delta_time_epoch = time_offset_f_s - time_offset_o_s
  
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
  print(f"      Duration : {delta_time_epoch:.1f} s")
  print(f"      Grid     : {grid_type_str}")

  print("\n  Compute")
  print("    SGP4 Propagation Running ... ", end='', flush=True)

  result_sgp4 = propagate_tle(
    tle_line_1 = tle_line_1,
    tle_line_2 = tle_line_2,
    to_j2000   = True,
    time_eval  = sgp4_times_grid,
    initial_dt = propagation_config.time_o_dt,
    final_dt   = propagation_config.time_f_dt,
  )
  print("Complete")

  if not result_sgp4.success:
    print(f"  SGP4 propagation failed: {result_sgp4.message}")
    return None

  # If comparing to Horizons, also propagate at ephemeris times
  if compare_jpl_horizons and result_jpl_horizons_ephemeris and result_jpl_horizons_ephemeris.success and result_jpl_horizons_ephemeris.time_grid is not None:
    ephem_times_s = result_jpl_horizons_ephemeris.time_grid.deltas  # seconds from time_o
    ephem_times_from_tle = ephem_times_s + time_offset_o_s  # seconds from TLE epoch

    print(f"    Propagating at {len(ephem_times_from_tle)} ephemeris time points ... ", end='', flush=True)

    result_sgp4_at_ephem = propagate_tle(
      tle_line_1 = tle_line_1,
      tle_line_2 = tle_line_2,
      to_j2000   = True,
      time_eval  = ephem_times_from_tle,
      initial_dt = result_jpl_horizons_ephemeris.time_grid.initial,
      final_dt   = result_jpl_horizons_ephemeris.time_grid.final,
    )

    if result_sgp4_at_ephem.success:
      # Store ephemeris-time results directly (time_grid already created by propagate_tle)
      result_sgp4.at_ephem_times = result_sgp4_at_ephem
      print("Complete")
    else:
      print(f"Failed: {result_sgp4_at_ephem.message}")
  
  return result_sgp4


def run_propagations(
  initial_state                 : np.ndarray,
  propagation_config            : PropagationConfig,
  spacecraft                    : SpacecraftProperties,
  compare_tle                   : bool,
  compare_jpl_horizons          : bool,
  result_jpl_horizons_ephemeris : Optional[PropagationResult],
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
    result_jpl_horizons_ephemeris : PropagationResult | None
      JPL Horizons ephemeris result for comparison.
    tle_line_1 : str
      First line of TLE.
    tle_line_2 : str
      Second line of TLE.
    two_body_gravity_model : GravityModelConfig
      Gravity model configuration.
      
  Output:
  -------
    result_high_fidelity : PropagationResult
      High-fidelity propagation result.
    result_sgp4 : PropagationResult | None
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
    if result_high_fidelity.success and result_high_fidelity.time_grid is not None:
      time_eval_s = result_high_fidelity.time_grid.deltas
    
    result_sgp4 = run_sgp4_propagation(
      result_jpl_horizons_ephemeris = result_jpl_horizons_ephemeris,
      tle_line_1                    = tle_line_1,
      tle_line_2                    = tle_line_2,
      propagation_config            = propagation_config,
      compare_jpl_horizons          = compare_jpl_horizons,
      time_eval_s                   = time_eval_s,
    )

  return result_high_fidelity, result_sgp4

