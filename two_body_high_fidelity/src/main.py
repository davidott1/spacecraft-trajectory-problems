"""
High-Fidelity Orbit Propagator

Description:
  This script propagates the orbit of a satellite using a high-fidelity numerical
  integration model. It takes a NORAD ID, a start time, and an end time as input.
  The initial state is derived from a hardcoded Two-Line Element (TLE) set.

  The propagation includes the following forces:
  - Earth's gravity (including J2, J3, J4 zonal harmonics)
  - Atmospheric drag
  - Third-body gravity from the Sun and Moon

  The script performs the following steps:
  1. Loads a reference ephemeris from JPL Horizons (if available).
  2. Derives an initial state from the TLE for the specified start time.
  3. Propagates the orbit using the high-fidelity model.
  4. Propagates the orbit using the SGP4 model for comparison.
  5. Generates and saves plots comparing the trajectories and their errors.

Usage:
  python -m src.main <norad_id> <start_time> <end_time>

Example:
  python -m src.main 25544 2025-10-01T00:00:00 2025-10-02T00:00:00
"""
import argparse
import matplotlib.pyplot as plt
import numpy             as np
import spiceypy          as spice

from pathlib         import Path
from scipy.integrate import solve_ivp
from datetime        import datetime, timedelta
from sgp4.api        import Satrec

from src.plot.trajectory             import plot_3d_trajectories, plot_time_series, plot_3d_error, plot_time_series_error
from src.propagation.propagator      import propagate_state_numerical_integration
from src.propagation.tle_propagator  import propagate_tle
from src.propagation.horizons_loader import load_horizons_ephemeris
from src.model.dynamics              import Acceleration, OrbitConverter
from src.model.constants             import PHYSICALCONSTANTS, CONVERTER

# Define supported objects and their properties
SUPPORTED_OBJECTS = {
  '25544': {
    'name'      : 'ISS',
    'tle_line1' : "1 25544U 98067A   25274.11280702  .00018412  00000-0  33478-3 0  9995",
    'tle_line2' : "2 25544  51.6324 142.0598 0001038 182.7689 177.3294 15.49574764531593",
    'mass'      : 420000.0,    # ISS mass [kg] (approximate)
    'cd'        : 2.2,         # drag coefficient [-]
    'area_drag' : 1000.0,      # cross-sectional area [m²] (approximate)
  }
}


def parse_and_validate_inputs(
  norad_id       : str,
  start_time_str : str,
  end_time_str   : str,
) -> dict:
  """
  Parse and validate input parameters for orbit propagation.
  
  Input:
  ------
    norad_id : str
      NORAD catalog ID of the satellite.
    start_time_str : str
      Start time in ISO format (e.g., '2025-10-01T00:00:00').
    end_time_str : str
      End time in ISO format (e.g., '2025-10-02T00:00:00').
  
  Output:
  -------
    dict
      A dictionary containing parsed and calculated propagation parameters.
  
  Raises:
  -------
    ValueError
      If NORAD ID is not supported.
  """
  # Validate norad id input
  if norad_id not in SUPPORTED_OBJECTS:
    raise ValueError(f"NORAD ID {norad_id} is not supported. Supported IDs: {list(SUPPORTED_OBJECTS.keys())}")

  # Get object properties
  obj_props = SUPPORTED_OBJECTS[norad_id]

  # Extract TLE lines
  tle_line1 = obj_props['tle_line1']
  tle_line2 = obj_props['tle_line2']

  # Parse TLE epoch
  satellite    = Satrec.twoline2rv(tle_line1, tle_line2)
  tle_epoch_jd = satellite.jdsatepoch + satellite.jdsatepochF
  tle_epoch_dt = datetime(2000, 1, 1, 12, 0, 0) + timedelta(days=tle_epoch_jd - 2451545.0)
  
  # Target propagation start/end times from arguments
  target_start_dt = datetime.fromisoformat(start_time_str)
  target_end_dt   = datetime.fromisoformat(end_time_str)
  delta_time      = (target_end_dt - target_start_dt).total_seconds()
  
  # Integration time bounds (seconds from TLE epoch)
  integ_time_o     = (target_start_dt - tle_epoch_dt).total_seconds()
  integ_time_f     = integ_time_o + delta_time
  delta_integ_time = integ_time_f - integ_time_o
  
  print(f"\nPropagation Time Span:")
  print(f"  TLE epoch                  : {tle_epoch_dt.isoformat()} UTC")
  print(f"  Target start               : {target_start_dt.isoformat()} UTC")
  print(f"  Target end                 : {target_end_dt.isoformat()} UTC")
  print(f"  Time offset from TLE epoch : {integ_time_o/3600:.2f} hours ({integ_time_o:.1f} seconds)")
  print(f"  Propagation duration       : {delta_integ_time/3600:.2f} hours")
  
  return {
    'obj_props'        : obj_props,
    'tle_line1'        : tle_line1,
    'tle_line2'        : tle_line2,
    'tle_epoch_dt'     : tle_epoch_dt,
    'tle_epoch_jd'     : tle_epoch_jd,
    'target_start_dt'  : target_start_dt,
    'target_end_dt'    : target_end_dt,
    'delta_time'       : delta_time,
    'integ_time_o'     : integ_time_o,
    'integ_time_f'     : integ_time_f,
    'delta_integ_time' : delta_integ_time,
    'mass'             : obj_props['mass'],
    'cd'               : obj_props['cd'],
    'area_drag'        : obj_props['area_drag'],
  }

def setup_paths_and_files(
  norad_id        : str,
  obj_name        : str,
  target_start_dt : datetime,
  target_end_dt   : datetime,
) -> dict:
  """
  Set up all required folder paths and file names for the propagation.
  
  Input:
  ------
    norad_id : str
      NORAD catalog ID of the satellite.
    obj_name : str
      Name of the object (e.g., 'ISS').
    target_start_dt : datetime
      Target start time as a datetime object.
    target_end_dt : datetime
      Target end time as a datetime object.
  
  Output:
  -------
    dict
      A dictionary containing paths to output, data, SPICE kernels,
      Horizons ephemeris, and leap seconds files.
  """
  # Output directory for figures
  output_folderpath = Path('./output/figures')
  output_folderpath.mkdir(parents=True, exist_ok=True)
  
  # Project and data paths
  project_root    = Path(__file__).parent.parent
  data_folderpath = project_root / 'data'
  
  # SPICE kernels path
  spice_kernels_folderpath = data_folderpath / 'spice_kernels'
  lsk_filepath             = spice_kernels_folderpath / 'naif0012.tls'
  
  # Horizons ephemeris file (dynamically named)
  start_str         = target_start_dt.strftime('%Y%m%dT%H%M%SZ')
  end_str           = target_end_dt.strftime('%Y%m%dT%H%M%SZ')
  horizons_filename = f"horizons_ephem_{norad_id}_{obj_name.lower()}_{start_str}_{end_str}_1m.csv"
  horizons_filepath = data_folderpath / 'ephems' / horizons_filename
  
  return {
    'output_folderpath'        : output_folderpath,
    'spice_kernels_folderpath' : spice_kernels_folderpath,
    'horizons_filepath'        : horizons_filepath,
    'lsk_filepath'             : lsk_filepath,
  }

def load_spice_files(
  lsk_filepath: Path,
):
  """
  Load required data files, e.g., SPICE kernels.
  
  Input:
  ------
    lsk_filepath : Path
      Path to the leap seconds kernel file.
  """
  # Load leap seconds kernel first (minimal kernel set for time conversion)
  spice.furnsh(str(lsk_filepath))

def process_horizons_result(
  result_horizons : dict,
) -> dict:
  """
  Processes and enriches the result dictionary from `load_horizons_ephemeris`.

  This function performs the following actions if the Horizons data loading was successful:
  1. Logs basic information about the loaded ephemeris (epoch, number of points, time span).
  2. Creates a `plot_time_s` array, which is a time vector in seconds starting from zero.
  3. Computes the classical orbital elements (COE) for each state vector in the ephemeris
     and adds them to the dictionary under the 'coe' key.

  If the loading was not successful, it prints a failure message.

  Input:
  ------
    result_horizons : dict
      The dictionary returned by `load_horizons_ephemeris`. It should contain
      'success', 'time', and 'state' keys.

  Output:
  -------
    dict | None
      An enriched dictionary with 'plot_time_s' and 'coe' keys if processing
      is successful. Returns `None` if the input indicates a failure.
  """
  if result_horizons and result_horizons.get('success'):
    print(f"  ✓ Horizons ephemeris loaded!")
    print(f"  Epoch            : {result_horizons['epoch'].isoformat()} UTC")
    print(f"  Number of points : {len(result_horizons['time'])}")
    print(f"  Time span        : {result_horizons['time'][0]:.1f} to {result_horizons['time'][-1]:.1f} seconds")

    # Create plot_time_s for seconds-based, zero-start plotting time
    result_horizons['plot_time_s'] = result_horizons['time']

    # Compute classical orbital elements for Horizons data
    num_points = result_horizons['state'].shape[1]
    result_horizons['coe'] = {
      'sma'  : np.zeros(num_points),
      'ecc'  : np.zeros(num_points),
      'inc'  : np.zeros(num_points),
      'raan' : np.zeros(num_points),
      'argp' : np.zeros(num_points),
      'ma'   : np.zeros(num_points),
      'ta'   : np.zeros(num_points),
      'ea'   : np.zeros(num_points),
    }

    for i in range(num_points):
      pos_vec = result_horizons['state'][0:3, i]
      vel_vec = result_horizons['state'][3:6, i]
      coe = OrbitConverter.pv_to_coe(
        pos_vec,
        vel_vec,
        PHYSICALCONSTANTS.EARTH.GP,
      )
      for key in result_horizons['coe'].keys():
        result_horizons['coe'][key][i] = coe[key]

    return result_horizons

  # Failure path
  msg = result_horizons.get('message') if isinstance(result_horizons, dict) else 'No result returned'
  print(f"  ✗ Horizons loading failed: {msg}")
  return None

def get_horizons_ephemeris(
  horizons_filepath : Path,
  target_start_dt   : datetime,
  target_end_dt     : datetime,
) -> dict | None:
  """
  Load and process JPL Horizons ephemeris.
  
  Input:
  ------
    horizons_filepath : Path
      Path to the Horizons ephemeris file.
    target_start_dt : datetime
      Start time for data request.
    target_end_dt : datetime
      End time for data request.
      
  Output:
  -------
    dict | None
      Processed Horizons result dictionary, or None if loading failed.
  """
  # Load Horizons ephemeris
  print("\n Loading JPL Horizons ephemeris ...")
  print(f"  File path   : {horizons_filepath}")
  print(f"  File exists : {horizons_filepath.exists()}")
  print(f"  Requesting data from {target_start_dt} to {target_end_dt}")
  
  # Load Horizons data
  result_horizons = load_horizons_ephemeris(
    filepath = str(horizons_filepath),
    start_dt = target_start_dt,
    end_dt   = target_end_dt,
  )

  # Process Horizons data
  result_horizons = process_horizons_result(result_horizons)
  
  return result_horizons

def get_initial_state(
  tle_line1            : str,
  tle_line2            : str,
  integ_time_o         : float,
  result_horizons      : dict | None,
  use_horizons_initial : bool = True,
  to_j2000             : bool = True,
) -> np.ndarray:
  """
  Get initial Cartesian state from Horizons (if available) or TLE.
  
  Input:
  ------
    tle_line1 : str
      The first line of the TLE.
    tle_line2 : str
      The second line of the TLE.
    integ_time_o : float
      The start time of the integration in seconds from the TLE epoch.
    result_horizons : dict | None
      The dictionary containing Horizons ephemeris data.
    use_horizons_initial : bool
      If True and Horizons data is available, use it. Otherwise use TLE.
    to_j2000 : bool
      Flag to indicate if the output state should be in the J2000 frame.
  
  Output:
  -------
    np.ndarray
      A 6x1 state vector [m, m, m, m/s, m/s, m/s].
  """
  print("\n Determining initial Cartesian state ...")
  
  # 1. Use Horizons if available and requested
  if use_horizons_initial and result_horizons and result_horizons.get('success'):
    horizons_initial_state = result_horizons['state'][:, 0]
    print(f"  ✓ Using Horizons initial state")
    print(f"  Horizons initial position: [{horizons_initial_state[0]:.3f}, {horizons_initial_state[1]:.3f}, {horizons_initial_state[2]:.3f}] m")
    print(f"  Horizons initial velocity: [{horizons_initial_state[3]:.3f}, {horizons_initial_state[4]:.3f}, {horizons_initial_state[5]:.3f}] m/s")
    return horizons_initial_state

  # 2. Fallback to TLE
  print(f"  Using TLE-derived initial state")
  print(f"  TLE Line 1: {tle_line1}")
  print(f"  TLE Line 2: {tle_line2}")

  result_tle_initial = propagate_tle(
    tle_line1  = tle_line1,
    tle_line2  = tle_line2,
    time_o     = integ_time_o,
    time_f     = integ_time_o,
    num_points = 1,
    to_j2000   = to_j2000,
  )
  if not result_tle_initial['success']:
    raise RuntimeError(f"Failed to get initial state from TLE: {result_tle_initial['message']}")

  tle_initial_state = result_tle_initial['state'][:, 0]
  print(f"  TLE Initial Pos {integ_time_o} : [{tle_initial_state[0]:.3f}, {tle_initial_state[1]:.3f}, {tle_initial_state[2]:.3f}] m")
  print(f"  TLE Initial Vel {integ_time_o} : [{tle_initial_state[3]:.3f}, {tle_initial_state[4]:.3f}, {tle_initial_state[5]:.3f}] m/s")

  return tle_initial_state

def get_et_j2000_from_utc(
  utc_dt : datetime,
) -> float:
  """
  Convert a UTC datetime object to Ephemeris Time (ET) in seconds past J2000.
  
  Input:
  ------
    utc_dt : datetime
      The UTC datetime to convert.
    
  Output:
  -------
    et_float : float
      The corresponding Ephemeris Time (ET) in seconds past J2000.
  """

  utc_str = utc_dt.strftime('%Y-%m-%dT%H:%M:%S')
  et_float = spice.str2et(utc_str)
  print(f"  Target start UTC : {utc_str}")
  print(f"  Target start ET  : {et_float:.3f} seconds past J2000")

  return et_float

def propagate_sgp4_at_horizons_grid(
  result_horizons : dict,
  integ_time_o    : float,
  tle_line1       : str,
  tle_line2       : str,
) -> dict | None:
  """
  Propagate SGP4 on the same time grid as the Horizons ephemeris.
  
  This function takes a Horizons result dictionary and propagates a TLE using
  SGP4 at the exact time points from the Horizons data. It then enriches the
  SGP4 result with time arrays and classical orbital elements.
  
  Input:
  ------
    result_horizons : dict
      The processed dictionary from JPL Horizons, containing 'success', 'plot_time_s'.
    integ_time_o : float
      The start time of the integration in seconds from the TLE epoch.
    tle_line1 : str
      The first line of the TLE.
    tle_line2 : str
      The second line of the TLE.
      
  Output:
  -------
    dict | None
      An enriched dictionary with SGP4 results, or None if propagation fails.
  """
  # Propagate SGP4 at Horizons time points for direct comparison
  if not (result_horizons and result_horizons.get('success')):
    return None
    
  print("  Propagating SGP4 at Horizons ephemeris time points ...")
  
  # Convert Horizons plot_time_s to integration times for SGP4
  sgp4_eval_times = result_horizons['plot_time_s'] + integ_time_o
  
  result_sgp4_at_horizons = propagate_tle(
    tle_line1  = tle_line1,
    tle_line2  = tle_line2,
    to_j2000   = True,
    t_eval     = sgp4_eval_times,
  )
  
  if not result_sgp4_at_horizons['success']:
    print(f"  ✗ SGP4 propagation at Horizons times failed: {result_sgp4_at_horizons['message']}")
    return None

  # Store integration time (seconds from TLE epoch)
  result_sgp4_at_horizons['integ_time_s'] = result_sgp4_at_horizons['time']
  
  # Create plotting time array (seconds from target start time)
  result_sgp4_at_horizons['plot_time_s'] = result_sgp4_at_horizons['time'] - integ_time_o
  
  # Compute COEs for SGP4 data
  num_points_sgp4 = result_sgp4_at_horizons['state'].shape[1]
  result_sgp4_at_horizons['coe'] = {
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
      result_sgp4_at_horizons['state'][0:3, i],
      result_sgp4_at_horizons['state'][3:6, i],
      PHYSICALCONSTANTS.EARTH.GP
    )
    for key in result_sgp4_at_horizons['coe'].keys():
      result_sgp4_at_horizons['coe'][key][i] = coe[key]
  
  print(f"  ✓ SGP4 propagation at Horizons times successful!")
  print(f"  Number of points: {num_points_sgp4}")
  print(f"  Time span: {result_sgp4_at_horizons['plot_time_s'][0]:.1f} to {result_sgp4_at_horizons['plot_time_s'][-1]:.1f} seconds")
  
  return result_sgp4_at_horizons

def run_high_fidelity_propagation(
  initial_state            : np.ndarray,
  integ_time_o             : float,
  integ_time_f             : float,
  target_start_dt          : datetime,
  target_end_dt            : datetime,
  mass                     : float,
  cd                       : float,
  area_drag                : float,
  use_spice                : bool,
  spice_kernels_folderpath : Path,
  result_horizons          : dict,
) -> dict:
  """
  Configure and run the high-fidelity numerical propagator.
  
  Input:
  ------
    initial_state : np.ndarray
      Initial state vector [x, y, z, vx, vy, vz].
    integ_time_o : float
      Integration start time (seconds from TLE epoch).
    integ_time_f : float
      Integration end time (seconds from TLE epoch).
    target_start_dt : datetime
      Target start datetime.
    target_end_dt : datetime
      Target end datetime.
    mass : float
      Satellite mass [kg].
    cd : float
      Drag coefficient.
    area_drag : float
      Drag area [m^2].
    use_spice : bool
      Whether to use SPICE for third-body ephemerides.
    spice_kernels_folderpath : Path
      Path to SPICE kernels folder.
    result_horizons : dict
      Result from Horizons loader (used for time grid).
      
  Output:
  -------
    dict
      Propagation result dictionary.
  """
  # Set up high-fidelity dynamics model
  print("\n Setting up high-fidelity dynamics model ...")
  print(f"  Including: Two-body gravity, J2, J3, J4, Atmospheric drag, Third-body (Sun/Moon)")
  
  # Convert UTC datetime to ET seconds past J2000 if SPICE is enabled
  time_et_o = 0.0
  if use_spice:
    time_et_o = get_et_j2000_from_utc(target_start_dt)
  
  # Define acceleration model
  acceleration = Acceleration(
    gp                      = PHYSICALCONSTANTS.EARTH.GP,
    time_et_o               = time_et_o,
    time_o                  = integ_time_o,
    j2                      = PHYSICALCONSTANTS.EARTH.J2,
    j3                      = PHYSICALCONSTANTS.EARTH.J3,
    j4                      = PHYSICALCONSTANTS.EARTH.J4,
    pos_ref                 = PHYSICALCONSTANTS.EARTH.RADIUS.EQUATOR,
    mass                    = mass,
    enable_drag             = True,
    cd                      = cd,
    area_drag               = area_drag,
    enable_third_body       = True,
    third_body_use_spice    = use_spice,
    third_body_bodies       = ['SUN', 'MOON'],
    spice_kernel_folderpath = str(spice_kernels_folderpath),
  )
  
  # Propagate with high-fidelity model
  delta_time = (target_end_dt - target_start_dt).total_seconds()
  print("\n Propagating orbit with high-fidelity model using numerical integration ...")
  print(f"  Time span: {target_start_dt} to {target_end_dt} UTC ({delta_time/3600:.1f} hours)")
  
  # Use Horizons time grid for high-fidelity propagation
  if result_horizons and result_horizons['success']:
    # Convert Horizons plot_time_s (seconds from target_start) to integration time (seconds from TLE epoch)
    horizons_integ_times = result_horizons['plot_time_s'] + integ_time_o
    print(f"  Using Horizons time grid: {len(horizons_integ_times)} points")
    print(f"  Time step: {result_horizons['plot_time_s'][1] - result_horizons['plot_time_s'][0]:.1f} seconds")
    
    result_hifi = propagate_state_numerical_integration(
      initial_state       = initial_state,
      time_o              = integ_time_o,
      time_f              = integ_time_f,
      dynamics            = acceleration,
      method              = 'DOP853',
      rtol                = 1e-12,
      atol                = 1e-12,
      dense_output        = True,  # Enable dense output for exact time evaluation
      t_eval              = horizons_integ_times,  # Evaluate at Horizons times
      get_coe_time_series = True,
      gp                  = PHYSICALCONSTANTS.EARTH.GP,
    )
  else:
    # If Horizons data is not available, error analysis is not possible.
    raise RuntimeError("Horizons ephemeris is required for high-fidelity propagation and error analysis, but it failed to load.")
  
  if result_hifi['success']:
    print(f"  ✓ Propagation successful!")
    print(f"  Number of time steps: {len(result_hifi['time'])}")
    
    # Store integration time (seconds from TLE epoch)
    result_hifi['integ_time_s'] = result_hifi['time']
    print(f"  Integration time range (from TLE epoch): {result_hifi['integ_time_s'][0]:.1f} to {result_hifi['integ_time_s'][-1]:.1f} seconds")
    
    # Create plotting time array (seconds from target start time)
    result_hifi['plot_time_s'] = result_hifi['time'] - integ_time_o
    print(f"  Plotting time range (from Oct 1 00:00): {result_hifi['plot_time_s'][0]:.1f} to {result_hifi['plot_time_s'][-1]:.1f} seconds")
  else:
    print(f"  ✗ Propagation failed: {result_hifi['message']}")
  
  return result_hifi

def run_propagations(
  initial_state            : np.ndarray,
  integ_time_o             : float,
  integ_time_f             : float,
  target_start_dt          : datetime,
  target_end_dt            : datetime,
  mass                     : float,
  cd                       : float,
  area_drag                : float,
  use_spice                : bool,
  spice_kernels_folderpath : Path,
  result_horizons          : dict,
  tle_line1                : str,
  tle_line2                : str,
) -> tuple[dict, dict | None]:
  """
  Run high-fidelity and SGP4 propagations.
  
  Input:
  ------
    initial_state : np.ndarray
      Initial state vector.
    integ_time_o : float
      Integration start time.
    integ_time_f : float
      Integration end time.
    target_start_dt : datetime
      Target start datetime.
    target_end_dt : datetime
      Target end datetime.
    mass : float
      Satellite mass.
    cd : float
      Drag coefficient.
    area_drag : float
      Drag area.
    use_spice : bool
      Whether to use SPICE.
    spice_kernels_folderpath : Path
      Path to SPICE kernels.
    result_horizons : dict
      Horizons ephemeris result.
    tle_line1 : str
      TLE line 1.
    tle_line2 : str
      TLE line 2.
      
  Output:
  -------
    tuple[dict, dict | None]
      Tuple containing (result_hifi, result_sgp4_at_horizons).
  """
  # Propagate: run high-fidelity propagation at Horizons time points for comparison
  result_hifi = run_high_fidelity_propagation(
    initial_state            = initial_state,
    integ_time_o             = integ_time_o,
    integ_time_f             = integ_time_f,
    target_start_dt          = target_start_dt,
    target_end_dt            = target_end_dt,
    mass                     = mass,
    cd                       = cd,
    area_drag                = area_drag,
    use_spice                = use_spice,
    spice_kernels_folderpath = spice_kernels_folderpath,
    result_horizons          = result_horizons,
  )
  
  # Propagate: run SGP4 at Horizons time points for comparison
  print("\nPropagating with SGP4 for comparison ...")
  result_sgp4_at_horizons = propagate_sgp4_at_horizons_grid(
    result_horizons = result_horizons,
    integ_time_o    = integ_time_o,
    tle_line1       = tle_line1,
    tle_line2       = tle_line2,
  )
  
  return result_hifi, result_sgp4_at_horizons

def get_simulation_paths(
  norad_id        : str,
  obj_name        : str,
  target_start_dt : datetime,
  target_end_dt   : datetime,
) -> tuple[Path, Path, Path, Path]:
  """
  Get paths for output, SPICE kernels, Horizons ephemeris, and leap seconds.
  
  Input:
  ------
    norad_id : str
      NORAD ID.
    obj_name : str
      Object name.
    target_start_dt : datetime
      Start time.
    target_end_dt : datetime
      End time.
      
  Output:
  -------
    tuple[Path, Path, Path, Path]
      (output_folderpath, spice_kernels_folderpath, horizons_filepath, lsk_filepath)
  """
  # Set up paths and files
  folderpaths_filepaths = setup_paths_and_files(
    norad_id        = norad_id,
    obj_name        = obj_name,
    target_start_dt = target_start_dt,
    target_end_dt   = target_end_dt,
  )
  
  return (
    folderpaths_filepaths['output_folderpath'],
    folderpaths_filepaths['spice_kernels_folderpath'],
    folderpaths_filepaths['horizons_filepath'],
    folderpaths_filepaths['lsk_filepath'],
  )

def main(
  norad_id       : str,
  start_time_str : str,
  end_time_str   : str,
) -> None:
  """
  Main function to run the high-fidelity orbit propagation.
  
  This function propagates an orbit using a high-fidelity dynamics model. The
  initial state is derived from a TLE, then propagated with detailed force
  models. The result is compared with SGP4 and JPL Horizons ephemeris.
  
  Input:
  ------
    norad_id : str
      NORAD catalog ID of the satellite.
    start_time_str : str
      Start time for propagation in ISO format.
    end_time_str : str
      End time for propagation in ISO format.
  
  Output:
  -------
    None
  """
  # Configuration
  use_spice = True  # Master flag to enable/disable all SPICE-related functionality

  # Process inputs and setup
  inputs = parse_and_validate_inputs(norad_id, start_time_str, end_time_str)
  
  obj_props        = inputs['obj_props']
  tle_line1_object = inputs['tle_line1']
  tle_line2_object = inputs['tle_line2']
  tle_epoch_dt     = inputs['tle_epoch_dt']
  target_start_dt  = inputs['target_start_dt']
  target_end_dt    = inputs['target_end_dt']
  delta_time       = inputs['delta_time']
  integ_time_o     = inputs['integ_time_o']
  integ_time_f     = inputs['integ_time_f']
  delta_integ_time = inputs['delta_integ_time']
  mass             = inputs['mass']
  cd               = inputs['cd']
  area_drag        = inputs['area_drag']

  # Set up paths and files
  output_folderpath, spice_kernels_folderpath, horizons_filepath, lsk_filepath = get_simulation_paths(
    norad_id        = norad_id,
    obj_name        = obj_props['name'],
    target_start_dt = target_start_dt,
    target_end_dt   = target_end_dt,
  )

  # Load spice files if SPICE is enabled
  if use_spice:
    load_spice_files(lsk_filepath)

  # Get Horizons ephemeris
  result_horizons = get_horizons_ephemeris(
    horizons_filepath = horizons_filepath,
    target_start_dt   = target_start_dt,
    target_end_dt     = target_end_dt,
  )

  # Determine initial state (from Horizons if available, else TLE)
  initial_state = get_initial_state(
    tle_line1            = tle_line1_object,
    tle_line2            = tle_line2_object,
    integ_time_o         = integ_time_o,
    result_horizons      = result_horizons,
    use_horizons_initial = True,
    to_j2000             = True,
  )

  # Run propagations: high-fidelity and SGP4 at Horizons times
  result_hifi, result_sgp4_at_horizons = run_propagations(
    initial_state            = initial_state,
    integ_time_o             = integ_time_o,
    integ_time_f             = integ_time_f,
    target_start_dt          = target_start_dt,
    target_end_dt            = target_end_dt,
    mass                     = mass,
    cd                       = cd,
    area_drag                = area_drag,
    use_spice                = use_spice,
    spice_kernels_folderpath = spice_kernels_folderpath,
    result_horizons          = result_horizons, # type: ignore
    tle_line1                = tle_line1_object,
    tle_line2                = tle_line2_object,
  )
  
  # Display results and create plots
  print("\n" + "="*60)
  print("Results Summary")
  print("="*60)
  
  print("\nFinal time ranges for plotting:")
  if result_horizons and result_horizons['success']:
    print(f"  Horizons:      {result_horizons['plot_time_s'][0]:.1f} to {result_horizons['plot_time_s'][-1]:.1f} seconds")
  print(f"  High-fidelity: {result_hifi['plot_time_s'][0]:.1f} to {result_hifi['plot_time_s'][-1]:.1f} seconds")
  if result_sgp4_at_horizons and result_sgp4_at_horizons['success']:
    print(f"  SGP4:          {result_sgp4_at_horizons['plot_time_s'][0]:.1f} to {result_sgp4_at_horizons['plot_time_s'][-1]:.1f} seconds")
  
  # Print final orbital elements (high-fidelity)
  final_alt_km = (np.linalg.norm(result_hifi['state'][0:3, -1]) - PHYSICALCONSTANTS.EARTH.RADIUS.EQUATOR) / 1e3
  print(f"\nFinal altitude (high-fidelity): {final_alt_km:.2f} km")
  print(f"Final semi-major axis: {result_hifi['coe']['sma'][-1]/1e3:.2f} km")
  print(f"Final eccentricity: {result_hifi['coe']['ecc'][-1]:.6f}")
  print(f"Final inclination: {np.rad2deg(result_hifi['coe']['inc'][-1]):.4f}°")
  
  # Create plots
  print("\nGenerating and saving plots...")
  
  # Horizons plots (first)
  if result_horizons and result_horizons['success']:
    fig1 = plot_3d_trajectories(result_horizons)
    fig1.suptitle('ISS Orbit - JPL Horizons Ephemeris', fontsize=16)
    fig1.savefig(output_folderpath / 'iss_horizons_3d.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_folderpath / 'iss_horizons_3d.png'}")
    
    fig2 = plot_time_series(result_horizons, epoch=target_start_dt)
    fig2.suptitle('ISS Orbit - JPL Horizons Time Series', fontsize=16)
    fig2.savefig(output_folderpath / 'iss_horizons_timeseries.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_folderpath / 'iss_horizons_timeseries.png'}")
  
  # High-fidelity plots (second)
  fig3 = plot_3d_trajectories(result_hifi)
  fig3.suptitle('ISS Orbit - High-Fidelity Propagation', fontsize=16)
  fig3.savefig(output_folderpath / 'iss_hifi_3d.png', dpi=300, bbox_inches='tight')
  print(f"  Saved: {output_folderpath / 'iss_hifi_3d.png'}")
  
  fig4 = plot_time_series(result_hifi, epoch=target_start_dt)
  fig4.suptitle('ISS Orbit - High-Fidelity Time Series', fontsize=16)
  fig4.savefig(output_folderpath / 'iss_hifi_timeseries.png', dpi=300, bbox_inches='tight')
  print(f"  Saved: {output_folderpath / 'iss_hifi_timeseries.png'}")
  
  # SGP4 at Horizons time points plots
  if result_sgp4_at_horizons and result_sgp4_at_horizons['success']:
    print("\nGenerating SGP4 at Horizons time points plots...")
    
    # 3D trajectory plot
    fig_sgp4_hz_3d = plot_3d_trajectories(result_sgp4_at_horizons)
    fig_sgp4_hz_3d.suptitle('ISS Orbit - SGP4 at Horizons Times', fontsize=16)
    fig_sgp4_hz_3d.savefig(output_folderpath / 'iss_sgp4_at_horizons_3d.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_folderpath / 'iss_sgp4_at_horizons_3d.png'}")
    
    # Time series plot
    fig_sgp4_hz_ts = plot_time_series(result_sgp4_at_horizons, epoch=target_start_dt)
    fig_sgp4_hz_ts.suptitle('ISS Orbit - SGP4 at Horizons Times - Time Series', fontsize=16)
    fig_sgp4_hz_ts.savefig(output_folderpath / 'iss_sgp4_at_horizons_timeseries.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_folderpath / 'iss_sgp4_at_horizons_timeseries.png'}")
    
    # Error plots comparing SGP4 to Horizons
    if result_horizons and result_horizons['success']:
      print("\nGenerating SGP4 vs Horizons error comparison plots...")
      
      # 3D error plot
      fig_sgp4_err_3d = plot_3d_error(result_horizons, result_sgp4_at_horizons)
      fig_sgp4_err_3d.suptitle('ISS Orbit Error: Horizons vs SGP4', fontsize=16)
      fig_sgp4_err_3d.savefig(output_folderpath / 'iss_sgp4_error_3d.png', dpi=300, bbox_inches='tight')
      print(f"  Saved: {output_folderpath / 'iss_sgp4_error_3d.png'}")
      
      # Time series error plot (RIC frame)
      fig_sgp4_err_ts = plot_time_series_error(result_horizons, result_sgp4_at_horizons, epoch=target_start_dt, use_ric=False)
      fig_sgp4_err_ts.suptitle('ISS XYZ Position/Velocity Errors: Horizons vs SGP4', fontsize=16)
      fig_sgp4_err_ts.savefig(output_folderpath / 'iss_sgp4_error_timeseries.png', dpi=300, bbox_inches='tight')
      print(f"  Saved: {output_folderpath / 'iss_sgp4_error_timeseries.png'}")
      
      # Compute and display SGP4 error statistics
      pos_error_sgp4_km = np.linalg.norm(result_sgp4_at_horizons['state'][0:3, :] - result_horizons['state'][0:3, :], axis=0) / 1e3
      vel_error_sgp4_ms = np.linalg.norm(result_sgp4_at_horizons['state'][3:6, :] - result_horizons['state'][3:6, :], axis=0)
      sma_error_sgp4_km = (result_sgp4_at_horizons['coe']['sma'] - result_horizons['coe']['sma']) / 1e3
      
      print("\nError Statistics (SGP4 vs Horizons):")
      print(f"  Position error - Mean: {np.mean(pos_error_sgp4_km):.3f} km, Max: {np.max(pos_error_sgp4_km):.3f} km")
      print(f"  Velocity error - Mean: {np.mean(vel_error_sgp4_ms):.3f} m/s, Max: {np.max(vel_error_sgp4_ms):.3f} m/s")
      print(f"  SMA error - Mean: {np.mean(np.abs(sma_error_sgp4_km)):.3f} km, Max: {np.max(np.abs(sma_error_sgp4_km)):.3f} km")
      
      # Analyze error growth by removing initial offset
      print(f"\nSGP4 Error Growth Analysis:")
      print(f"  Initial position error (at Oct 1 00:00): {pos_error_sgp4_km[0]:.3f} km")
      print(f"  Final position error (at Oct 2 00:00): {pos_error_sgp4_km[-1]:.3f} km")
      print(f"  Error growth over 24 hours: {pos_error_sgp4_km[-1] - pos_error_sgp4_km[0]:.3f} km")
      print(f"  Initial velocity error: {vel_error_sgp4_ms[0]:.3f} m/s")
      print(f"  Final velocity error: {vel_error_sgp4_ms[-1]:.3f} m/s")
      print(f"  Velocity error growth: {vel_error_sgp4_ms[-1] - vel_error_sgp4_ms[0]:.3f} m/s")
      
      # Create plot showing error growth (initial offset removed)
      fig_sgp4_growth = plt.figure(figsize=(14, 8))
      
      # Position error growth subplot
      ax1 = fig_sgp4_growth.add_subplot(2, 1, 1)
      pos_error_growth = pos_error_sgp4_km - pos_error_sgp4_km[0]  # Remove initial offset
      ax1.plot(result_horizons['plot_time_s']/3600, pos_error_growth, 'b-', linewidth=2)
      ax1.set_ylabel('Position Error Growth [km]')
      ax1.set_title('SGP4 vs Horizons: Error Growth (Initial Offset Removed)')
      ax1.grid(True, alpha=0.3)
      ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
      
      # Velocity error growth subplot
      ax2 = fig_sgp4_growth.add_subplot(2, 1, 2)
      vel_error_growth = vel_error_sgp4_ms - vel_error_sgp4_ms[0]  # Remove initial offset
      ax2.plot(result_horizons['plot_time_s']/3600, vel_error_growth, 'r-', linewidth=2)
      ax2.set_xlabel('Time [hours]')
      ax2.set_ylabel('Velocity Error Growth [m/s]')
      ax2.grid(True, alpha=0.3)
      ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
      
      fig_sgp4_growth.tight_layout()
      fig_sgp4_growth.savefig(output_folderpath / 'iss_sgp4_error_growth.png', dpi=300, bbox_inches='tight')
      print(f"  Saved: {output_folderpath / 'iss_sgp4_error_growth.png'}")
  
  # Create error comparison plots if both Horizons and high-fidelity are available
  if result_horizons and result_horizons['success'] and result_hifi['success']:
    print("\nGenerating error comparison plots...")
    
    # Position and velocity error plots
    fig_err_3d = plot_3d_error(result_horizons, result_hifi)
    fig_err_3d.suptitle('ISS Orbit Error: Horizons vs High-Fidelity', fontsize=16)
    fig_err_3d.savefig(output_folderpath / 'iss_error_3d.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_folderpath / 'iss_error_3d.png'}")
    
    # Time series error plots
    fig_err_ts = plot_time_series_error(result_horizons, result_hifi, epoch=target_start_dt)
    fig_err_ts.suptitle('ISS RIC Position/Velocity Errors: Horizons vs High-Fidelity', fontsize=16)
    fig_err_ts.savefig(output_folderpath / 'iss_error_timeseries.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_folderpath / 'iss_error_timeseries.png'}")
    
    # Compute and display error statistics
    pos_error_km = np.linalg.norm(result_hifi['state'][0:3, :] - result_horizons['state'][0:3, :], axis=0) / 1e3
    vel_error_ms = np.linalg.norm(result_hifi['state'][3:6, :] - result_horizons['state'][3:6, :], axis=0)
    sma_error_km = (result_hifi['coe']['sma'] - result_horizons['coe']['sma']) / 1e3
    
    print("\nError Statistics (High-Fidelity vs Horizons):")
    print(f"  Position error - Mean: {np.mean(pos_error_km):.3f} km, Max: {np.max(pos_error_km):.3f} km")
    print(f"  Velocity error - Mean: {np.mean(vel_error_ms):.3f} m/s, Max: {np.max(vel_error_ms):.3f} m/s")
    print(f"  SMA error - Mean: {np.mean(np.abs(sma_error_km)):.3f} km, Max: {np.max(np.abs(sma_error_km)):.3f} km")
    
    # Also compute argument of latitude error statistics
    if all(k in result_horizons['coe'] and k in result_hifi['coe'] for k in ['raan', 'argp', 'ta']):
        u_ref = result_horizons['coe']['raan'] + result_horizons['coe']['argp'] + result_horizons['coe']['ta']
        u_comp = result_hifi['coe']['raan'] + result_hifi['coe']['argp'] + result_hifi['coe']['ta']
        u_error_rad = np.arctan2(np.sin(u_ref - u_comp), np.cos(u_ref - u_comp))
        u_error_deg = u_error_rad * CONVERTER.DEG_PER_RAD
        print(f"  Arg of latitude error - Mean: {np.mean(u_error_deg):.3f}°, RMS: {np.sqrt(np.mean(u_error_deg**2)):.3f}°, Max: {np.max(np.abs(u_error_deg)):.3f}°")

  print(f"\nAll figures saved to: {output_folderpath}")
  plt.show()
  
  # Unload all SPICE kernels if they were loaded
  if use_spice:
    spice.kclear()
  
  return result_hifi


def parse_command_line_arguments() -> argparse.Namespace:
  """
  Parse command-line arguments for the orbit propagation script.
  
  Output:
  -------
    argparse.Namespace
      An object containing the parsed arguments (norad_id, start_time, end_time).
  """
  parser = argparse.ArgumentParser(description="Run high-fidelity orbit propagation.")
  parser.add_argument('norad_id'   , type=str, help="NORAD Catalog ID of the satellite (e.g., '25544' for ISS).")
  parser.add_argument('start_time' , type=str, help="Start time for propagation in ISO format (e.g., '2025-10-01T00:00:00').")
  parser.add_argument('end_time'   , type=str, help="End time for propagation in ISO format (e.g., '2025-10-02T00:00:00').")
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_command_line_arguments()
  main(
    args.norad_id,
    args.start_time,
    args.end_time,
  )