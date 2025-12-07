import yaml
import spiceypy as spice
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import sys
import subprocess

from src.model.time_converter import utc_to_et
from src.model.dynamics       import OrbitConverter
from src.model.constants      import PHYSICALCONSTANTS
from src.input.cli            import parse_time
from src.utility.tle_helper   import get_tle_satellite_and_tle_epoch


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
  jpl_horizons_folderpath : Path,
  desired_time_o_dt       : datetime,
  desired_time_f_dt       : datetime,
  norad_id                : str,
  object_name             : str,
  step                    : str = "1m",
) -> Optional[dict]:
  """
  Load and process JPL Horizons ephemeris.
  
  Searches for compatible files in the folder that contain the desired timespan.
  
  Input:
  ------
    jpl_horizons_folderpath : Path
      Path to the folder containing Horizons ephemeris files.
    desired_time_o_dt : datetime
      Start time for data request.
    desired_time_f_dt : datetime
      End time for data request.
    norad_id : str
      NORAD catalog ID for the object.
    object_name : str
      Name of the object (for display purposes).
    step : str
      Time step for ephemeris download (default "1m").
      
  Output:
  -------
    result_jpl_horizons : dict | None
      Processed Horizons result dictionary, or None if loading failed.
      Contains 'success' key indicating if data was loaded successfully.
  """
  # Display JPL Horizons folderpath
  try:
    rel_folderpath = jpl_horizons_folderpath.relative_to(Path.cwd())
    display_path = f"<project_folderpath>/{rel_folderpath}"
  except ValueError:
    display_path = jpl_horizons_folderpath

  print("  JPL Horizons Ephemeris")
  print(f"    Folderpath : {display_path}")
  
  # Search for compatible file
  compatible_file = find_compatible_horizons_file(
    jpl_horizons_folderpath = jpl_horizons_folderpath,
    norad_id                = norad_id,
    time_start_dt           = desired_time_o_dt,
    time_end_dt             = desired_time_f_dt,
  )
  
  if compatible_file is None:
    # Prompt user to download
    print(f"               :   ... No compatible files found. Download from JPL Horizons? (y/n)", end=" ", flush=True)
    user_response = input().strip().lower()
    
    if user_response == 'y':
      print(f"               :   ... Downloading {object_name} ({norad_id}) ...", end=" ", flush=True)
      
      try:
        # Construct command to run the download module
        cmd = [
          sys.executable, "-m", "src.download.ephems_and_tles",
          norad_id,
          desired_time_o_dt.strftime('%Y-%m-%dT%H:%M:%S'),
          desired_time_f_dt.strftime('%Y-%m-%dT%H:%M:%S'),
          step
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Done")
        
        # Search again for the downloaded file
        compatible_file = find_compatible_horizons_file(
          jpl_horizons_folderpath = jpl_horizons_folderpath,
          norad_id                = norad_id,
          time_start_dt           = desired_time_o_dt,
          time_end_dt             = desired_time_f_dt,
        )
        
      except subprocess.CalledProcessError as e:
        print(f"Error : {e}")
      except Exception as e:
        print(f"Error : {e}")

    if compatible_file is None:
      return {
        'success' : False,
        'message' : f"No compatible Horizons ephemeris files found in: {jpl_horizons_folderpath}",
      }
  
  # Display the file being loaded
  try:
    rel_path = compatible_file.relative_to(Path.cwd())
    display_path = f"<project_folderpath>/{rel_path}"
  except ValueError:
    display_path = compatible_file
  print(f"    Filepath   : {display_path}")

  # Print timespan info
  print(f"    Timespan")
  print(f"      Desired")
  try:
    start_et = utc_to_et(desired_time_o_dt)
    end_et   = utc_to_et(desired_time_f_dt)
    start_time_str = f"{desired_time_o_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC / {start_et:.6f} ET"
    end_time_str   = f"{desired_time_f_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC / {end_et:.6f} ET"
  except:
    start_time_str = f"{desired_time_o_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC / N/A ET"
    end_time_str   = f"{desired_time_f_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC / N/A ET"
  duration_s = (desired_time_f_dt - desired_time_o_dt).total_seconds()
  print(f"        Initial  : {start_time_str}")
  print(f"        Final    : {end_time_str}")
  print(f"        Duration : {duration_s:.1f} s")
  
  # Load Horizons data
  result_jpl_horizons = load_horizons_ephemeris(
    filepath      = str(compatible_file),
    time_start_dt = desired_time_o_dt,
    time_end_dt   = desired_time_f_dt,
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
      start_et = utc_to_et(actual_start)
      end_et   = utc_to_et(actual_end)
      start_time_str = f"{actual_start.strftime('%Y-%m-%d %H:%M:%S')} UTC / {start_et:.6f} ET"
      end_time_str   = f"{actual_end.strftime('%Y-%m-%d %H:%M:%S')} UTC / {end_et:.6f} ET"
    except:
      start_time_str = f"{actual_start.strftime('%Y-%m-%d %H:%M:%S')} UTC / N/A ET"
      end_time_str   = f"{actual_end.strftime('%Y-%m-%d %H:%M:%S')} UTC / N/A ET"

    duration_s = result_horizons['delta_time'][-1] - result_horizons['delta_time'][0]

    print(f"      Actual")
    print(f"        Initial  : {start_time_str}")
    print(f"        Final    : {end_time_str}")
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


def find_compatible_horizons_file(
  jpl_horizons_folderpath : Path,
  norad_id                : str,
  time_start_dt           : datetime,
  time_end_dt             : datetime,
) -> Optional[Path]:
  """
  Search for existing Horizons ephemeris files that contain the desired timespan.
  
  Parses the timespan directly from filenames for speed.
  Expected filename format: horizons_ephem_<norad_id>_<name>_<start>_<end>_<step>.csv
  where <start> and <end> are in format: YYYYMMDDTHHMMSSz
  
  Input:
  ------
    jpl_horizons_folderpath : Path
      Path to the folder containing JPL Horizons ephemeris files.
    norad_id : str
      NORAD catalog ID for the object.
    time_start_dt : datetime
      Desired start time.
    time_end_dt : datetime
      Desired end time.
      
  Output:
  -------
    Path | None
      Path to a compatible file if found, None otherwise.
  """
  # Ensure the folder exists before searching
  if not jpl_horizons_folderpath.exists():
    return None
  
  # Use Path.glob() to find all ephemeris files for this object
  glob_pattern = f'horizons_ephem_{norad_id}_*.csv'
  matching_files = list(jpl_horizons_folderpath.glob(glob_pattern))

  if not matching_files:
    return None
  
  # Time tolerance for matching
  time_tolerance = timedelta(minutes=5)
  
  for filepath in matching_files:
    # Parse timespan from filename
    # Format: horizons_ephem_<norad_id>_<name>_<start>_<end>_<step>.csv
    filename = filepath.stem  # Remove .csv extension
    parts = filename.split('_')
    
    # Need at least: horizons, ephem, norad_id, name, start, end, step = 7 parts
    if len(parts) < 7:
      continue
    
    try:
      # Start time is third-to-last, end time is second-to-last (before step)
      # Format: YYYYMMDDTHHMMSSz
      start_str = parts[-3]  # e.g., "20251001T000000Z"
      end_str   = parts[-2]  # e.g., "20251002T000000Z"
      
      # Parse the datetime strings
      file_start_dt = datetime.strptime(start_str, '%Y%m%dT%H%M%SZ')
      file_end_dt = datetime.strptime(end_str, '%Y%m%dT%H%M%SZ')
      
      # Check if this file contains the desired timespan (with tolerance)
      start_ok = file_start_dt <= (time_start_dt + time_tolerance)
      end_ok   = file_end_dt   >= (time_end_dt   - time_tolerance)
      
      if start_ok and end_ok:
        return filepath
        
    except (ValueError, IndexError):
      # Skip files with unparseable names
      continue
  
  return None


def find_compatible_tle_file(
  norad_id          : str,
  tles_folderpath   : Path,
  desired_time_o_dt : datetime,
  desired_time_f_dt : datetime,
) -> Optional[Path]:
  """
  Search for existing TLE files for the given NORAD ID that cover the desired timespan.
  
  Parses the timespan directly from filenames for speed.
  Expected filename format: celestrak_tle_<norad_id>_<name>_<start>_<end>.txt
  where <start> and <end> are in format: YYYYMMDDTHHMMSSz
  
  Input:
  ------
    norad_id : str
      NORAD catalog ID.
    tles_folderpath : Path
      Path to the folder containing TLE files.
    desired_time_o_dt : datetime
      Desired initial time.
    desired_time_f_dt : datetime
      Desired final time.
      
  Output:
  -------
    Path | None
      Path to a compatible file if found, None otherwise.
  """
  # Ensure the folder exists before searching
  if not tles_folderpath.exists():
    return None
  
  # Search for TLE files matching the NORAD ID (new format)
  glob_pattern = f'celestrak_tle_{norad_id}_*.txt'
  matching_files = list(tles_folderpath.glob(glob_pattern))
  
  if not matching_files:
    return None
  
  # Time tolerance for matching (TLEs can be valid for longer periods)
  time_tolerance = timedelta(days=7)
  
  for filepath in matching_files:
    # Parse timespan from filename
    # Format: celestrak_tle_<norad_id>_<name>_<start>_<end>.txt
    filename = filepath.stem  # Remove .txt extension
    parts = filename.split('_')
    
    # Need at least: celestrak, tle, norad_id, name, start, end = 6 parts
    if len(parts) < 6:
      continue
    
    try:
      # Start time is second-to-last, end time is last
      # Format: YYYYMMDDTHHMMSSz
      start_str = parts[-2]  # e.g., "20251001T000000Z"
      end_str = parts[-1]    # e.g., "20251002T000000Z"
      
      # Parse the datetime strings
      file_start_dt = datetime.strptime(start_str, '%Y%m%dT%H%M%SZ')
      file_end_dt = datetime.strptime(end_str, '%Y%m%dT%H%M%SZ')
      
      # Check if this file covers the desired timespan (with tolerance)
      start_ok = file_start_dt <= (desired_time_o_dt + time_tolerance)
      end_ok = file_end_dt >= (desired_time_f_dt - time_tolerance)
      
      if start_ok and end_ok:
        return filepath
        
    except (ValueError, IndexError):
      # Skip files with unparseable names
      continue
  
  return None


def get_celestrak_tle(
  norad_id          : str,
  object_name       : str,
  tles_folderpath   : Path,
  desired_time_o_dt : datetime,
  desired_time_f_dt : datetime,
) -> Optional[dict]:
  """
  Load TLE from local file or download from Celestrak if not available.
  
  Input:
  ------
    norad_id : str
      NORAD catalog ID for the object.
    object_name : str
      Name of the object (for display purposes).
    tles_folderpath : Path
      Path to the folder containing TLE files.
    desired_time_o_dt : datetime
      Desired initial time for TLE coverage.
    desired_time_f_dt : datetime
      Desired final time for TLE coverage.
      
  Output:
  -------
    dict | None
      Dictionary containing TLE data if successful:
      - success      : bool
      - message      : str
      - tle_line_1   : str (closest TLE to desired_time_o_dt)
      - tle_line_2   : str (closest TLE to desired_time_o_dt)
      - tle_epoch_dt : datetime (epoch of closest TLE)
      - all_tles     : list[dict] (all TLEs in the file)
      Returns None if loading failed.
  """
  # Construct expected TLE filepath
  time_o_str   = desired_time_o_dt.strftime('%Y%m%dT%H%M%SZ')
  time_f_str   = desired_time_f_dt.strftime('%Y%m%dT%H%M%SZ')
  tle_filename = f"celestrak_tle_{norad_id}_{object_name.lower()}_{time_o_str}_{time_f_str}.txt"
  tle_filepath = tles_folderpath / tle_filename
  
  # Display TLE filepath
  try:
    rel_path     = tle_filepath.relative_to(Path.cwd())
    display_path = f"<project_folderpath>/{rel_path}"
  except ValueError:
    display_path = tle_filepath

  print("  Celestrak TLE")
  print(f"    Filepath  : {display_path}")
  
  # Check if TLE file exists
  if not tle_filepath.exists():
    print(f"             :   ... TLE file not found, searching for compatible files ...", end=" ", flush=True)
    
    # Search for compatible TLE files
    compatible_file = find_compatible_tle_file(
      norad_id          = norad_id,
      tles_folderpath   = tles_folderpath,
      desired_time_o_dt = desired_time_o_dt,
      desired_time_f_dt = desired_time_f_dt,
    )
    
    if compatible_file is None:
      print(f"No compatible TLE files found")
      
      # Prompt user to download
      print(f"             :   ... Download from Celestrak? (y/n)", end=" ", flush=True)
      user_response = input().strip().lower()
      
      if user_response == 'y':
        print(f"             :   ... Downloading {object_name} ({norad_id}) ...", end=" ", flush=True)
        
        try:
          # Download TLE from Celestrak
          tle_data = download_tle_from_celestrak(norad_id)
          
          if tle_data:
            # Ensure TLEs folder exists
            tles_folderpath.mkdir(parents=True, exist_ok=True)
            
            # Save TLE to file (3-line format per TLE)
            with open(tle_filepath, 'w') as f:
              f.write(f"{tle_data['tle_line_0']}\n")
              f.write(f"{tle_data['tle_line_1']}\n")
              f.write(f"{tle_data['tle_line_2']}\n")
            
            print("Done")
            compatible_file = tle_filepath
          else:
            print("Failed - No TLE data returned")
            
        except Exception as e:
          print(f"Error : {e}")

      if compatible_file is None:
        return {
          'success' : False,
          'message' : f"TLE file not found and download failed: {tle_filepath}",
        }
    else:
      print(f"Found")
    
    tle_filepath = compatible_file
    try:
      rel_filepath     = compatible_file.relative_to(Path.cwd())
      display_filepath = f"<project_folderpath>/{rel_filepath}"
    except ValueError:
      display_filepath = compatible_file
    print(f"             : {display_filepath}")

  # Load TLE from file
  result = load_tle_file(tle_filepath, desired_time_o_dt)
  
  if result and result.get('success'):
    print(f"    TLE Count : {len(result.get('all_tles', []))} TLE(s)")
    print(f"    TLE Epoch : {result['tle_epoch_dt'].strftime('%Y-%m-%d %H:%M:%S')} UTC (closest to desired initial time)")
  
  return result


def load_tle_file(
  filepath          : Path,
  reference_time_dt : Optional[datetime] = None,
) -> Optional[dict]:
  """
  Load TLE data from a file. Supports multiple TLEs in a single file.
  
  Expected file format (can have multiple TLE sets):
  - Line 0 (optional): Object name
  - Line 1: TLE line 1
  - Line 2: TLE line 2
  (repeat for additional TLEs)
  
  Input:
  ------
    filepath : Path
      Path to the TLE file.
    reference_time_dt : datetime, optional
      Reference time to select the closest TLE. If None, uses the first TLE.
      
  Output:
  -------
    dict | None
      Dictionary containing TLE data if successful:
      - success      : bool
      - message      : str
      - tle_line_1   : str (closest TLE to reference_time_dt)
      - tle_line_2   : str (closest TLE to reference_time_dt)
      - tle_epoch_dt : datetime (epoch of closest TLE)
      - all_tles     : list[dict] (all TLEs in the file)
  """
  try:
    with open(filepath, 'r') as f:
      lines = [line.strip() for line in f.readlines() if line.strip()]
    
    if len(lines) < 2:
      return {
        'success' : False,
        'message' : f"TLE file has insufficient lines: {filepath}",
      }
    
    # Parse all TLEs from the file
    all_tles = []
    i = 0
    while i < len(lines):
      # Check if this is a 3-line format (name + TLE) or 2-line format
      if i + 2 < len(lines) and lines[i + 1].startswith('1 ') and lines[i + 2].startswith('2 '):
        # 3-line format with name
        tle_line_0 = lines[i]
        tle_line_1 = lines[i + 1]
        tle_line_2 = lines[i + 2]
        i += 3
      elif lines[i].startswith('1 ') and i + 1 < len(lines) and lines[i + 1].startswith('2 '):
        # 2-line format without name
        tle_line_0 = ""
        tle_line_1 = lines[i]
        tle_line_2 = lines[i + 1]
        i += 2
      else:
        # Skip unrecognized line
        i += 1
        continue
      
      # Parse TLE epoch
      try:
        tle_epoch_dt, _ = get_tle_satellite_and_tle_epoch(tle_line_1, tle_line_2)
        all_tles.append({
          'tle_line_0'   : tle_line_0,
          'tle_line_1'   : tle_line_1,
          'tle_line_2'   : tle_line_2,
          'tle_epoch_dt' : tle_epoch_dt,
        })
      except Exception:
        # Skip TLEs that can't be parsed
        continue
    
    if not all_tles:
      return {
        'success' : False,
        'message' : f"No valid TLEs found in file: {filepath}",
      }
    
    # Select the closest TLE to reference time, or first TLE if no reference
    if reference_time_dt is not None:
      # Find TLE with epoch closest to reference time
      closest_tle = min(
        all_tles,
        key=lambda tle: abs((tle['tle_epoch_dt'] - reference_time_dt).total_seconds())
      )
    else:
      # Assume first TLE if no reference time provided
      closest_tle = all_tles[0]
    
    return {
      'success'      : True,
      'message'      : 'TLE loaded successfully',
      'tle_line_0'   : closest_tle['tle_line_0'],
      'tle_line_1'   : closest_tle['tle_line_1'],
      'tle_line_2'   : closest_tle['tle_line_2'],
      'tle_epoch_dt' : closest_tle['tle_epoch_dt'],
      'all_tles'     : all_tles,
    }
    
  except Exception as e:
    return {
      'success' : False,
      'message' : str(e),
    }


def download_tle_from_celestrak(
  norad_id : str,
) -> Optional[dict]:
  """
  Download TLE from Celestrak API.
  
  Input:
  ------
    norad_id : str
      NORAD catalog ID.
      
  Output:
  -------
    dict | None
      Dictionary containing TLE lines if successful, None otherwise.
  """
  import urllib.request
  import urllib.error
  
  # Celestrak API URL for single satellite TLE
  url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=TLE"
  
  try:
    with urllib.request.urlopen(url, timeout=30) as response:
      content = response.read().decode('utf-8')
    
    lines = [line.strip() for line in content.strip().split('\n') if line.strip()]
    
    if len(lines) < 2:
      return None
    
    # Parse the response
    if len(lines) >= 3 and lines[1].startswith('1 ') and lines[2].startswith('2 '):
      # 3-line format
      return {
        'tle_line_0' : lines[0],
        'tle_line_1' : lines[1],
        'tle_line_2' : lines[2],
      }
    elif lines[0].startswith('1 ') and lines[1].startswith('2 '):
      # 2-line format
      return {
        'tle_line_0' : "",
        'tle_line_1' : lines[0],
        'tle_line_2' : lines[1],
      }
    
    return None
    
  except urllib.error.URLError as e:
    print(f"URL Error: {e}")
    return None
  except Exception as e:
    print(f"Error: {e}")
    return None
