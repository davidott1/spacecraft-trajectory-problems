import yaml
import spiceypy as spice
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from src.model.time_converter import utc_to_et
from src.model.dynamics       import OrbitConverter
from src.model.constants      import PHYSICALCONSTANTS
from src.input.cli            import parse_time


def load_supported_objects() -> dict:
  """
  Load supported objects from YAML configuration file.
  
  Output:
  -------
    dict
      Dictionary of supported objects loaded from the YAML file.
  """
  # Adjust path traversal: input -> src -> project_root
  project_root = Path(__file__).parent.parent.parent
  config_path  = project_root / 'data' / 'supported_objects.yaml'
  
  if not config_path.exists():
    raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
  with open(config_path, 'r') as f:
    return yaml.safe_load(f)


def load_files(
  use_spice                : bool,
  spice_kernels_folderpath : Path,
  lsk_filepath             : Path,
) -> None:
  """
  Load necessary files for the simulation, including SPICE kernels if enabled.
  
  Input:
  ------
    use_spice : bool
      Flag to enable/disable SPICE usage.
    spice_kernels_folderpath : Path
      Path to the SPICE kernels folder.
    lsk_filepath : Path
      Path to the leap seconds kernel file.
  """
  print("\nLoad Files")
  print(f"  Project Folderpath : {Path.cwd()}")

  # Load spice files if SPICE is enabled
  load_spice_files(use_spice, spice_kernels_folderpath, lsk_filepath)


def unload_files(
  use_spice: bool,
) -> None:
  """
  Unload files that were loaded for the simulation, including SPICE kernels if enabled.
  
  Input:
  ------
    use_spice : bool
      Flag to enable/disable SPICE usage.
  """
  # Unload SPICE files if SPICE was enabled
  unload_spice_files(use_spice)


def load_spice_files(
  use_spice                : bool,
  spice_kernels_folderpath : Path,
  lsk_filepath             : Path,
) -> None:
  """
  Load required data files, e.g., SPICE kernels.
  
  Input:
  ------
    use_spice : bool
      Flag to enable/disable SPICE usage.
    spice_kernels_folderpath : Path
      Path to the SPICE kernels folder.
    lsk_filepath : Path
      Path to the leap seconds kernel file.
  """
  if use_spice:
    if not spice_kernels_folderpath.exists():
      raise FileNotFoundError(f"SPICE kernels folder not found: {spice_kernels_folderpath}")
    if not lsk_filepath.exists():
      raise FileNotFoundError(f"SPICE leap seconds kernel not found: {lsk_filepath}")

    try:
      rel_path = spice_kernels_folderpath.relative_to(Path.cwd())
      display_path = f"<project_folderpath>/{rel_path}"
    except ValueError:
      display_path = spice_kernels_folderpath

    print(f"  Spice Kernels")
    print(f"    Folderpath : {display_path}")
    
    # Load leap seconds kernel first (minimal kernel set for time conversion)
    spice.furnsh(str(lsk_filepath))


def unload_spice_files(
  use_spice : bool,
) -> None:
  """
  Unload all SPICE kernels if they were loaded.
  
  Input:
  ------
    use_spice : bool
      Flag to enable/disable SPICE usage.
  """
  if use_spice:
    spice.kclear()


def load_horizons_ephemeris(
  filepath : str,
  time_start_dt : Optional[datetime] = None,
  time_end_dt   : Optional[datetime] = None,
) -> dict:
  """
  Load ephemeris from a custom JPL Horizons .csv file produced by the download script module: src.download.ephems_and_tles
  
  Strictly expects:
  - Row 1  : Header (variable names)
             targetname,datetime,tdb_utc_offset,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,lighttime,range,range_rate
  - Row 2  : Units
  - Row 3+ : Data
  
  If the structure is not followed, raises ValueError.
  
  Input:
  ------
  filepath : str
    Path to the CSV file.
  time_start_dt : datetime, optional
    Start datetime to filter data.
  time_end_dt : datetime, optional
    End datetime to filter data.
    
  Output:
  -------
  dict
    Dictionary containing:
    - success : bool
    - message : str
    - epoch   : datetime (of the first data point)
    - time    : np.ndarray (seconds from epoch)
    - state   : np.ndarray (6xN state vectors in m and m/s)

  Code Structure:
  ---------------
  1. Read first two rows to validate structure and get units.
  2. Read actual data, skipping the units row.
  3. Parse time and filter by requested timespan.
  4. Extract position and velocity, applying unit conversions.
  5. Return results or error message.
  """
  try:
    # 1. Read first two rows to validate structure and get units
    # Read header (row 0) and units (row 1)
    # We read 2 rows: header is row 0 (file line 1), data row 0 is file line 2 (units)
    df_preview = pd.read_csv(filepath, nrows=1)
    
    if len(df_preview) < 1:
      raise ValueError(f"CSV file {filepath} is empty or too short.")

    # Check units in the first row of data (which corresponds to line 2 of file)
    units_row = df_preview.iloc[0]
    
    # Validate Position Units
    #   - ensure m on output
    if 'pos_x' not in df_preview.columns:
      raise ValueError(f"Column 'pos_x' not found in {filepath}")
       
    pos_unit = str(units_row['pos_x']).strip().lower()
    if pos_unit == 'km':
      pos_multiplier = 1000.0
    elif pos_unit == 'm':
      pos_multiplier = 1.0
    else:
      raise ValueError(f"Invalid position unit in second row: '{pos_unit}'. Expected 'km' or 'm'.")

    # Validate Velocity Units
    #   - ensure m/s on output
    if 'vel_x' not in df_preview.columns:
      raise ValueError(f"Column 'vel_x' not found in {filepath}")

    vel_unit = str(units_row['vel_x']).strip().lower()
    if vel_unit == 'km/s':
      vel_multiplier = 1000.0
    elif vel_unit == 'm/s':
      vel_multiplier = 1.0
    else:
      raise ValueError(f"Invalid velocity unit in second row: '{vel_unit}'. Expected 'km/s' or 'm/s'.")

    # 2. Read actual data, skipping the units row
    # header=0 means line 1 is header. skiprows=[1] means skip line 2.
    df_raw = pd.read_csv(filepath, header=0, skiprows=[1])
    
    # 3. Parse time and filter by requested timespan

    # Validate datetime column
    if 'datetime' not in df_raw.columns:
      raise ValueError(f"Column 'datetime' not found in {filepath}")

    # Convert to datetime objects
    time_dt = [parse_time(str(t)) for t in df_raw['datetime']]
    
    # Timespan Filter (if requested)
    timespan_filter_index = np.full(len(time_dt), True)
    if time_start_dt:
      timespan_filter_index &= np.array([t >= time_start_dt for t in time_dt])
    if time_end_dt:
      timespan_filter_index &= np.array([t <=   time_end_dt for t in time_dt])
    
    if not np.any(timespan_filter_index):
      return {
        'success' : False,
        'message' : f"No data points found within requested time range {time_start_dt} to {time_end_dt}",
        'time'    : [],
        'state'   : [],
      }

    # Apply timespan filter
    df_filtered    = df_raw[timespan_filter_index].reset_index(drop=True) # apply .reset_index b/c xxx
    time_filtered = np.array(time_dt)[timespan_filter_index]
    
    # Define epoch as the first time point in the filtered series
    time_o     = time_filtered[0]
    time_f     = time_filtered[-1]
    delta_time = np.array([(t - time_o).total_seconds() for t in time_filtered])
    
    # 4. Extract position and velocity, applying unit conversions
    pos   = df_filtered[['pos_x', 'pos_y', 'pos_z']].to_numpy().T * pos_multiplier
    vel   = df_filtered[['vel_x', 'vel_y', 'vel_z']].to_numpy().T * vel_multiplier
    state = np.vstack((pos, vel))
    
    # 5. Return results
    return {
      'success'    : True,
      'message'    : 'Horizons ephemeris loaded successfully',
      'time_o'     : time_o,
      'time_f'     : time_f,
      'delta_time' : delta_time,
      'state'      : state,
    }

  except Exception as e:
    return {
      'success'    : False,
      'message'    : str(e),
      'time_o'     : [],
      'time_f'     : [],
      'delta_time' : [],
      'state'      : [],
    }


def get_horizons_ephemeris(
  jpl_horizons_filepath : Path,
  desired_time_o_dt     : datetime,
  target_end_dt         : datetime,
) -> Optional[dict]:
  """
  Load and process JPL Horizons ephemeris.
  
  Input:
  ------
    jpl_horizons_filepath : Path
      Path to the Horizons ephemeris file.
    desired_time_o_dt : datetime
      Start time for data request.
    target_end_dt : datetime
      End time for data request.
      
  Output:
  -------
    result_jpl_horizons : dict | None
      Processed Horizons result dictionary, or None if loading failed.
  """
  # Load Horizons ephemeris
  try:
    rel_path = jpl_horizons_filepath.relative_to(Path.cwd())
    display_path = f"<project_folderpath>/{rel_path}"
  except ValueError:
    display_path = jpl_horizons_filepath

  print("  JPL Horizons Ephemeris")
  print(f"    Filepath : {display_path}")
  print(f"    Timespan")
  print(f"      Desired")
  
  try:
    start_et = f"{utc_to_et(desired_time_o_dt):.6f} ET"
    end_et   = f"{utc_to_et(target_end_dt):.6f} ET"
  except:
    start_et = "N/A ET"
    end_et   = "N/A ET"

  duration_s = (target_end_dt - desired_time_o_dt).total_seconds()

  print(f"        Start    : {desired_time_o_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC ({start_et})")
  print(f"        End      : {target_end_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC ({end_et})")
  print(f"        Duration : {duration_s:.1f} s")
  
  # Load Horizons data
  result_jpl_horizons = load_horizons_ephemeris(
    filepath      = str(jpl_horizons_filepath),
    time_start_dt = desired_time_o_dt,
    time_end_dt   = target_end_dt,
  )

  # Process Horizons data
  result_jpl_horizons = process_horizons_result(result_jpl_horizons)
  
  return result_jpl_horizons


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
    print(f"        Initial  : {actual_start.strftime('%Y-%m-%d %H:%M:%S')} UTC ({start_et})")
    print(f"        Final    : {  actual_end.strftime('%Y-%m-%d %H:%M:%S')} UTC ({end_et})")
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
