"""
High-Fidelity Orbit Propagator

Description:
  This script propagates the orbit of a satellite using a high-fidelity numerical
  integration model. It takes a NORAD ID, a start time, and an end time as input.
  The initial state is derived from a hardcoded Two-Line Element (TLE) set.

  The propagation includes the following forces:
  - Earth's gravity (including J2, J3, J4 zonal harmonics)
  - Atmospheric drag
  - Solar Radiation Pressure (SRP)
  - Third-body gravity from the Sun and Moon

  The script performs the following steps:
  1. Loads a reference ephemeris from JPL Horizons (if available).
  2. Derives an initial state from the TLE for the specified start time.
  3. Propagates the orbit using the high-fidelity model.
  4. Propagates the orbit using the SGP4 model for comparison.
  5. Generates and saves plots comparing the trajectories and their errors.

Usage:

  Argument                     Required   Description
  ---------------------------  --------   --------------------------------------------------
  --input-object-type          Yes        Type of input object (e.g., norad-id)
  --norad-id                   Yes*       NORAD ID (required for norad-id type)
  --timespan                   Yes        Start and end time (ISO format)
  --include-zonal-harmonics    No         Enable zonal harmonics
  --zonal-harmonics            No         List of zonal harmonics: J2 (default), J3, and J4
  --include-spice              No         Enable SPICE functionality
  --include-third-body         No         Enable third-body gravity
  --include-srp                No         Enable Solar Radiation Pressure

  Example Commands:
    python -m src.main \
      --input-object-type <type> \
      --norad-id <id> \
      --timespan <start> <end> \
      [--include-spice] \
      [--include-third-body] \
      [--include-srp] \
      [--include-zonal-harmonics] \
      [--zonal-harmonics <J2|J3|J4>]

    python -m src.main \
      --input-object-type norad-id \
      --norad-id 25544 \
      --timespan 2025-10-01T00:00:00 2025-10-02T00:00:00 \
      --include-zonal-harmonics \
      --zonal-harmonics J2 J3 J4 \
      --include-third-body \
      --include-srp \
      --include-spice


    
"""
import argparse
import sys
import matplotlib.pyplot as plt
import numpy             as np
import spiceypy          as spice

from pathlib         import Path
from scipy.integrate import solve_ivp
from datetime        import datetime, timedelta
from sgp4.api        import Satrec
from types           import SimpleNamespace
from typing          import Optional

from src.plot.trajectory             import plot_3d_trajectories, plot_time_series, plot_3d_error, plot_time_series_error, generate_error_plots, generate_3d_and_time_series_plots, generate_plots
from src.propagation.propagator      import propagate_state_numerical_integration
from src.utility.tle_propagator      import propagate_tle
from src.utility.loader              import load_supported_objects, load_spice_files, unload_spice_files
from src.utility.printer             import print_results_summary
from src.config.parser               import parse_time, parse_and_validate_inputs, get_config, setup_paths_and_files
from src.propagation.horizons_loader import load_horizons_ephemeris
from src.model.dynamics              import Acceleration, OrbitConverter
from src.model.constants             import PHYSICALCONSTANTS, CONVERTER
from src.model.time_converter        import utc_to_et


def process_horizons_result(
  result_horizons : dict,
) -> Optional[dict]:
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
    actual_start = result_horizons['time_o']
    actual_end   = actual_start + timedelta(seconds=result_horizons['delta_time'][-1])
    
    try:
      start_et = f"{utc_to_et(actual_start):.6f} ET"
      end_et   = f"{utc_to_et(actual_end):.6f} ET"
    except:
      start_et = "N/A ET"
      end_et   = "N/A ET"

    duration_s = result_horizons['delta_time'][-1] - result_horizons['delta_time'][0]

    print(f"      Actual")
    print(f"        Start    : {actual_start.strftime('%Y-%m-%d %H:%M:%S')} UTC ({start_et})")
    print(f"        End      : {actual_end.strftime('%Y-%m-%d %H:%M:%S')} UTC ({end_et})")
    print(f"        Duration : {duration_s:.1f} s")
    print(f"        Grid     : {len(result_horizons['delta_time'])} points")

    # Create plot_time_s for seconds-based, zero-start plotting time
    result_horizons['plot_time_s'] = result_horizons['delta_time'] - result_horizons['delta_time'][0]
    
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
  print(f"    - Horizons loading failed: {msg}")
  return None


def get_horizons_ephemeris(
  horizons_filepath : Path,
  target_start_dt   : datetime,
  target_end_dt     : datetime,
) -> Optional[dict]:
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
  try:
    rel_path = horizons_filepath.relative_to(Path.cwd())
    display_path = f"<project_folderpath>/{rel_path}"
  except ValueError:
    display_path = horizons_filepath

  print("  JPL Horizons Ephemeris")
  print(f"    Filepath : {display_path}")
  print(f"    Timespan")
  print(f"      Desired")
  
  try:
    start_et = f"{utc_to_et(target_start_dt):.6f} ET"
    end_et   = f"{utc_to_et(target_end_dt):.6f} ET"
  except:
    start_et = "N/A ET"
    end_et   = "N/A ET"

  duration_s = (target_end_dt - target_start_dt).total_seconds()

  print(f"        Start    : {target_start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC ({start_et})")
  print(f"        End      : {target_end_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC ({end_et})")
  print(f"        Duration : {duration_s:.1f} s")
  
  # Load Horizons data
  result_horizons = load_horizons_ephemeris(
    filepath      = str(horizons_filepath),
    time_start_dt = target_start_dt,
    time_end_dt   = target_end_dt,
  )

  # Process Horizons data
  result_horizons = process_horizons_result(result_horizons)
  
  return result_horizons








def propagate_sgp4_at_horizons_grid(
  result_horizons : dict,
  integ_time_o    : float,
  tle_line1       : str,
  tle_line2       : str,
  target_start_dt : datetime,
  target_end_dt   : datetime,
) -> Optional[dict]:
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
    
  # Calculate time offset between Horizons data start and requested target start
  horizons_start_dt = result_horizons['time_o']
  time_offset_s     = (horizons_start_dt - target_start_dt).total_seconds()

  # Convert Horizons plot_time_s to integration times for SGP4
  # Shift by time_offset_s to align SGP4 evaluation with Actual Horizons times
  sgp4_eval_times = result_horizons['plot_time_s'] + integ_time_o + time_offset_s
  
  # Calculate display values
  duration_actual_s = result_horizons['plot_time_s'][-1]
  grid_end_dt       = horizons_start_dt + timedelta(seconds=duration_actual_s)
  duration_desired_s = (target_end_dt - target_start_dt).total_seconds()
  num_points = len(sgp4_eval_times)

  # Helper for ET string
  def get_et_str(dt):
      try:
          return f"{utc_to_et(dt):.6f} ET"
      except:
          return "N/A ET"

  print(f"  Configuration")
  print(f"    Timespan")
  print(f"      Desired")
  print(f"        Start    : {target_start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC ({get_et_str(target_start_dt)})")
  print(f"        End      : {target_end_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC ({get_et_str(target_end_dt)})")
  print(f"        Duration : {duration_desired_s:.1f} s")
  print(f"      Actual")
  print(f"        Start    : {horizons_start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC ({get_et_str(horizons_start_dt)})")
  print(f"        End      : {grid_end_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC ({get_et_str(grid_end_dt)})")
  print(f"        Duration : {duration_actual_s:.1f} s")
  print(f"        Grid     : {num_points} points")

  print("\n  Compute")
  print("    Numerical Integration Running ... ", end='', flush=True)
  result_sgp4_at_horizons = propagate_tle(
    tle_line1  = tle_line1,
    tle_line2  = tle_line2,
    to_j2000   = True,
    time_eval  = sgp4_eval_times,
  )
  print("Complete")
  
  if not result_sgp4_at_horizons['success']:
    print(f"  SGP4 propagation at Horizons times failed: {result_sgp4_at_horizons['message']}")
    return None

  # Store integration time (seconds from TLE epoch)
  result_sgp4_at_horizons['integ_time_s'] = result_sgp4_at_horizons['time']
  
  # Create plotting time array (seconds from Actual start time)
  result_sgp4_at_horizons['plot_time_s'] = result_horizons['plot_time_s']
  
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
  
  # Calculate display values
  duration_actual_s = result_horizons['plot_time_s'][-1]
  grid_end_dt       = horizons_start_dt + timedelta(seconds=duration_actual_s)
  duration_desired_s = (target_end_dt - target_start_dt).total_seconds()
  
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
  cr                       : float,
  area_srp                 : float,
  use_spice                : bool,
  include_third_body       : bool,
  include_zonal_harmonics  : bool,
  zonal_harmonics_list     : list,
  include_srp              : bool,
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
      Drag area [m²].
    cr : float
      Reflectivity coefficient.
    area_srp : float
      SRP area [m²].
    use_spice : bool
      Whether to use SPICE for third-body ephemerides.
    include_third_body : bool
      Whether to enable third-body gravity.
    include_zonal_harmonics : bool
      Whether to enable zonal harmonics.
    zonal_harmonics_list : list
      List of zonal harmonics to include.
    include_srp : bool
      Whether to enable Solar Radiation Pressure.
    spice_kernels_folderpath : Path
      Path to SPICE kernels folder.
    result_horizons : dict
      Result from Horizons loader (used for time grid).
      
  Output:
  -------
    dict
      Propagation result dictionary.
  """

  # Display configuration
  print("\nHigh-Fidelity Model")

  # Determine Actual times if Horizons is available (for grid alignment)
  actual_start_dt = target_start_dt
  actual_end_dt   = target_end_dt
  
  if result_horizons and result_horizons.get('success'):
      actual_start_dt = result_horizons['time_o']
      duration_horizons = result_horizons['plot_time_s'][-1]
      actual_end_dt = actual_start_dt + timedelta(seconds=duration_horizons)

  # Helper to get ET (or approx ET)
  J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0)
  def get_et(dt):
    if use_spice:
      try:
        return utc_to_et(dt)
      except:
        pass
    return (dt - J2000_EPOCH).total_seconds()

  # Calculate Ephemeris Times (ET) for integration
  time_et_o_actual = get_et(actual_start_dt)
  time_et_f_actual = get_et(actual_end_dt)
  
  # Calculate ETs for display
  time_et_o_desired = get_et(target_start_dt)
  time_et_f_desired = get_et(target_end_dt)

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
  delta_time = (target_end_dt - target_start_dt).total_seconds()
  grid_points = 0
  step_size = 0.0
  if result_horizons and result_horizons.get('success'):
    grid_points = len(result_horizons['plot_time_s'])
    if grid_points > 1:
      step_size = result_horizons['plot_time_s'][1] - result_horizons['plot_time_s'][0]

  # Set up high-fidelity dynamics model
  print("  Configuration")
  print( "    Timespan")
  print(f"      Desired")
  print(f"        Start    : {target_start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC ({time_et_o_desired:.6f} ET)")
  print(f"        End      : {target_end_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC ({time_et_f_desired:.6f} ET)")
  print(f"        Duration : {delta_time:.1f} s")
  
  if result_horizons and result_horizons.get('success'):
      duration_actual = (actual_end_dt - actual_start_dt).total_seconds()
      print(f"      Actual")
      print(f"        Start    : {actual_start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC ({time_et_o_actual:.6f} ET)")
      print(f"        End      : {actual_end_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC ({time_et_f_actual:.6f} ET)")
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
    print("          Bodies    : Sun, Moon")
    if use_spice:
      print("          Ephemeris : SPICE (High Accuracy)")
      print(f"          Note      : SPICE kernels loaded for third-body ephemerides.")
    else:
      print("          Ephemeris : Analytical (Approximate)")
      
  print("      Atmospheric Drag")
  print( "        Model      : Exponential Atmosphere")
  print(f"        Parameters : Cd={cd}, Area_Drag={area_drag} m², Mass={mass} kg")

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
    enable_drag             = True,
    cd                      = cd,
    area_drag               = area_drag,
    enable_srp              = include_srp,
    cr                      = cr,
    area_srp                = area_srp,
    enable_third_body       = include_third_body,
    third_body_use_spice    = use_spice,
    third_body_bodies       = ['SUN', 'MOON'],
    spice_kernel_folderpath = str(spice_kernels_folderpath),
  )
  
  # Propagate with high-fidelity model: use Horizons time grid
  if result_horizons and result_horizons['success']:
    # Convert Horizons plot_time_s (seconds from ACTUAL start) to ET
    horizons_et_times = result_horizons['plot_time_s'] + time_et_o_actual
    
    # Update integration start/end times to match grid exactly (avoids floating point errors)
    time_et_o_actual = horizons_et_times[0]
    time_et_f_actual = horizons_et_times[-1]
  
    # Print numerical integration settings
    print("    Numerical Integration")
    print(f"      Method     : DOP853")
    print(f"      Tolerances : rtol=1e-12, atol=1e-12")

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
    print(f"  Propagation failed: {result_high_fidelity['message']}")
  
  return result_high_fidelity


def run_propagations(
  initial_state            : np.ndarray,
  integ_time_o             : float,
  integ_time_f             : float,
  target_start_dt          : datetime,
  target_end_dt            : datetime,
  mass                     : float,
  cd                       : float,
  area_drag                : float,
  cr                       : float,
  area_srp                 : float,
  use_spice                : bool,
  include_third_body       : bool,
  include_zonal_harmonics  : bool,
  zonal_harmonics_list     : list,
  include_srp              : bool,
  spice_kernels_folderpath : Path,
  result_horizons          : dict,
  tle_line1                : str,
  tle_line2                : str,
) -> tuple[dict, Optional[dict]]:
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
    cr : float
      Reflectivity coefficient.
    area_srp : float
      SRP area.
    use_spice : bool
      Whether to use SPICE.
    include_third_body : bool
      Whether to enable third-body gravity.
    include_zonal_harmonics : bool
      Whether to enable zonal harmonics.
    zonal_harmonics_list : list
      List of zonal harmonics to include.
    include_srp : bool
      Whether to enable SRP.
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
      Tuple containing (result_high_fidelity, result_sgp4_at_horizons).
  """
  # Propagate: run high-fidelity propagation at Horizons time points for comparison
  result_high_fidelity = run_high_fidelity_propagation(
    initial_state            = initial_state,
    integ_time_o             = integ_time_o,
    integ_time_f             = integ_time_f,
    target_start_dt          = target_start_dt,
    target_end_dt            = target_end_dt,
    mass                     = mass,
    cd                       = cd,
    area_drag                = area_drag,
    cr                       = cr,
    area_srp                 = area_srp,
    use_spice                = use_spice,
    include_third_body       = include_third_body,
    include_zonal_harmonics  = include_zonal_harmonics,
    zonal_harmonics_list     = zonal_harmonics_list,
    include_srp              = include_srp,
    spice_kernels_folderpath = spice_kernels_folderpath,
    result_horizons          = result_horizons,
  )
  
  # Propagate: run SGP4 at Horizons time points for comparison
  print("\nSGP4 Model")
  result_sgp4_at_horizons = propagate_sgp4_at_horizons_grid(
    result_horizons = result_horizons,
    integ_time_o    = integ_time_o,
    tle_line1       = tle_line1,
    tle_line2       = tle_line2,
    target_start_dt = target_start_dt,
    target_end_dt   = target_end_dt,
  )
  
  return result_high_fidelity, result_sgp4_at_horizons


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
  input_object_type       : str,
  norad_id                : str,
  timespan                : list,
  use_spice               : bool = False,
  include_third_body      : bool = False,
  include_zonal_harmonics : bool = False,
  zonal_harmonics_list    : list = None,
  include_srp             : bool = False,
) -> dict:
  """
  Main function to run the high-fidelity orbit propagation.
  
  This function propagates an orbit using a high-fidelity dynamics model. The
  initial state is derived from a TLE, then propagated with detailed force
  models. The result is compared with SGP4 and JPL Horizons ephemeris.
  
  Input:
  ------
    input_object_type : str
      Type of input object.
    norad_id : str
      NORAD Catalog ID of the satellite.
    timespan : list
      Start and end time for propagation in ISO format.
    use_spice : bool
      Flag to enable/disable SPICE usage.
    include_third_body : bool
      Flag to enable/disable third-body gravity.
    include_zonal_harmonics : bool
      Flag to enable/disable zonal harmonics.
    zonal_harmonics_list : list
      List of zonal harmonics to include.
    include_srp : bool
      Flag to enable/disable Solar Radiation Pressure.
  
  Output:
  -------
    None
  """
  # Process inputs and setup
  inputs_dict = parse_and_validate_inputs(input_object_type, norad_id, timespan, use_spice, include_third_body, include_zonal_harmonics, zonal_harmonics_list, include_srp)
  config      = get_config(inputs_dict)

  # Set up paths and files
  output_folderpath, spice_kernels_folderpath, horizons_filepath, lsk_filepath = get_simulation_paths(
    norad_id        = norad_id,
    obj_name        = config.obj_props['name'],
    target_start_dt = config.target_start_dt,
    target_end_dt   = config.target_end_dt,
  )

  print("\nLoad Files")
  print(f"  Project Folderpath : {Path.cwd()}")

  # Load spice files if SPICE is enabled
  load_spice_files(config.use_spice, spice_kernels_folderpath, lsk_filepath)

  # Get Horizons ephemeris
  result_horizons = get_horizons_ephemeris(
    horizons_filepath = horizons_filepath,
    target_start_dt   = config.target_start_dt,
    target_end_dt     = config.target_end_dt,
  )

  # Determine initial state (from Horizons if available, else TLE)
  initial_state = get_initial_state(
    tle_line1            = config.tle_line1,
    tle_line2            = config.tle_line2,
    integ_time_o         = config.integ_time_o,
    result_horizons      = result_horizons,
    use_horizons_initial = True,
    to_j2000             = True,
  )

  # Run propagations: high-fidelity and SGP4 at Horizons times
  result_high_fidelity, result_sgp4_at_horizons = run_propagations(
    initial_state            = initial_state,
    integ_time_o             = config.integ_time_o,
    integ_time_f             = config.integ_time_f,
    target_start_dt          = config.target_start_dt,
    target_end_dt            = config.target_end_dt,
    mass                     = config.mass,
    cd                       = config.cd,
    area_drag                = config.area_drag,
    cr                       = config.cr,
    area_srp                 = config.area_srp,
    use_spice                = config.use_spice,
    include_third_body       = config.include_third_body,
    include_zonal_harmonics  = config.include_zonal_harmonics,
    zonal_harmonics_list     = config.zonal_harmonics_list,
    include_srp              = config.include_srp,
    spice_kernels_folderpath = spice_kernels_folderpath,
    result_horizons          = result_horizons, # type: ignore
    tle_line1                = config.tle_line1,
    tle_line2                = config.tle_line2,
  )
  
  # Display results and create plots
  print_results_summary(result_horizons, result_high_fidelity, result_sgp4_at_horizons)
  
  # Create plots
  generate_plots(
    result_horizons         = result_horizons,
    result_high_fidelity    = result_high_fidelity,
    result_sgp4_at_horizons = result_sgp4_at_horizons,
    target_start_dt         = config.target_start_dt,
    output_folderpath       = output_folderpath,
  )
  
  # Unload all SPICE kernels if they were loaded
  unload_spice_files(config.use_spice)
  
  return result_high_fidelity


def print_input_table(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
  """
  Print a table of input arguments, their values, and status.
  """
  # Identify explicit actions from sys.argv order
  explicit_actions = []
  seen_dests = set()
  
  # Map option strings to actions for quick lookup
  opt_map = {}
  for action in parser._actions:
    for opt in action.option_strings:
      opt_map[opt] = action

  # Scan sys.argv to find order of explicit arguments
  for arg in sys.argv[1:]:
    if arg.startswith('-'):
      # Handle --opt=val syntax
      opt_name = arg.split('=')[0]
      if opt_name in opt_map:
        action = opt_map[opt_name]
        if action.dest not in seen_dests and action.dest != 'help':
          explicit_actions.append(action)
          seen_dests.add(action.dest)

  # Add remaining actions (defaults or required ones not yet seen)
  all_actions = [a for a in parser._actions if a.dest != 'help']
  remaining_actions = [a for a in all_actions if a.dest not in seen_dests]
  
  # Combine lists
  display_actions = explicit_actions + remaining_actions

  # Prepare data for printing
  rows = []
  headers = ["Argument", "Value", "Default", "Explicit"]
  col_widths = [len(h) for h in headers]
  data_widths = [0] * len(headers)

  for action in display_actions:
    val = getattr(args, action.dest, None)
    
    # Determine if explicitly set
    is_explicit = (action.dest in seen_dests) or action.required
    
    # Fallback check for explicit if not found in simple scan (e.g. abbreviations)
    if not is_explicit:
       for opt in action.option_strings:
        for arg in sys.argv:
          if arg == opt or arg.startswith(opt + "="):
            is_explicit = True
            break
        if is_explicit:
          break
        
    # Format value
    if isinstance(val, list):
      val_str = " ".join(map(str, val))
    else:
      val_str = str(val)
      
    if len(val_str) > 42:
      val_str = val_str[:39] + "..."

    # Format default value
    if isinstance(action.default, list):
      default_str = " ".join(map(str, action.default))
    else:
      default_str = str(action.default)
      
    row = [action.dest, val_str, default_str, str(is_explicit)]
    rows.append(row)
    
    # Update column widths
    for i, item in enumerate(row):
      col_widths[i] = max(col_widths[i], len(item))
      data_widths[i] = max(data_widths[i], len(item))

  # Print table
  print("\nInput Configuration")
  
  # Print headers
  header_fmt = f"  {{:<{col_widths[0]}}}    {{:<{col_widths[1]}}}    {{:<{col_widths[2]}}}    {{:<{col_widths[3]}}}"
  print(header_fmt.format(*headers))
  
  # Print separators
  separators = ["-" * w for w in col_widths]
  print(header_fmt.format(*separators))
  
  # Print rows
  for row in rows:
    print(header_fmt.format(*row))


def parse_command_line_arguments() -> argparse.Namespace:
  """
  Parse command-line arguments for the orbit propagation script.
  
  Output:
  -------
    argparse.Namespace
      An object containing the parsed arguments.
  """
  parser = argparse.ArgumentParser(description="Run high-fidelity orbit propagation.")
  
  # If no arguments provided, print help and exit
  if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)
  
  # Object definition arguments
  parser.add_argument('--input-object-type', type=str, choices=['norad-id', 'norad_id', 'norad id'], required=True, help="Type of input object identifier.")
  parser.add_argument('--norad-id', type=str, help="NORAD Catalog ID of the satellite (e.g., '25544' for ISS).")
  
  # Time arguments
  parser.add_argument('--timespan', nargs=2, metavar=('TIME_START', 'TIME_END'), required=True, help="Start and end time for propagation in ISO format (e.g., '2025-10-01T00:00:00 2025-10-02T00:00:00').")
  
  # Optional arguments
  parser.add_argument('--include-spice'          , dest='use_spice'              , action='store_true', help="Enable SPICE functionality (disabled by default).")
  parser.add_argument('--include-third-body'     , dest='include_third_body'     , action='store_true', help="Enable third-body gravity (disabled by default).")
  parser.add_argument('--include-zonal-harmonics', dest='include_zonal_harmonics', action='store_true', help="Enable zonal harmonics (disabled by default).")
  parser.add_argument('--zonal-harmonics'        , dest='zonal_harmonics_list'   , nargs='+', choices=['J2', 'J3', 'J4'], default=['J2'], help="List of zonal harmonics to include (default: J2).")
  parser.add_argument('--include-srp'            , dest='include_srp'            , action='store_true', help="Enable Solar Radiation Pressure (disabled by default).")

  # Set default values for optional flags
  parser.set_defaults(use_spice=False, include_third_body=False, include_zonal_harmonics=False, include_srp=False)
  
  args = parser.parse_args()
  
  # Print input summary
  print_input_table(args, parser)
  
  return args


if __name__ == "__main__":
  args = parse_command_line_arguments()
  main(
    args.input_object_type,
    args.norad_id,
    args.timespan,
    args.use_spice,
    args.include_third_body,
    args.include_zonal_harmonics,
    args.zonal_harmonics_list,
    args.include_srp,
  )