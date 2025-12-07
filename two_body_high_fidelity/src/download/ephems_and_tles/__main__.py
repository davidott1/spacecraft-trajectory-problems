"""
Download ephemeris and TLE data for any satellite by NORAD ID

Usage:
  python -m src.download.ephems_and_tles <norad_id> <start_time> <end_time> [step]
  
Example:
  python -m src.download.ephems_and_tles 25544 "2025-10-01" "2025-10-08"
  python -m src.download.ephems_and_tles 25544 "2025-10-01T00:00:00Z" "2025-10-02T00:00:00Z" 1m
  python -m src.download.ephems_and_tles 25544 "2025-10-01 00:00" "2025-10-08 00:00"
"""

import requests
import sys
import numpy as np
import os
from astroquery.jplhorizons import Horizons
from pathlib                import Path
from datetime               import datetime, timedelta
from astropy.time           import Time, TimeDelta
from astropy.table          import Table
from astropy                import units as u
from typing                 import List, Dict, Tuple, Optional, Union

# Project root directory:
#   /local_absolute_path/two_body_high_fidelity/src/download/ephems_and_tles/__main__.py
#     ->
#   /local_absolute_path/two_body_high_fidelity/
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  #

from src.model.constants import CONVERTER
from src.input.cli       import parse_time

EPHEM_OUTPUT_DIR = PROJECT_ROOT / 'data' / 'ephems'
TLE_OUTPUT_DIR   = PROJECT_ROOT / 'data' / 'tles'


def get_satellite_name(
  norad_id : int,
) -> str:
  """
  Get satellite name from NORAD ID or return generic name.
  
  Input:
  ------
  norad_id : int
    NORAD catalog number.
  
  Output:
  -------
  str:
    Satellite name.
  """
  known_satellites = {
    25544: 'iss',
    25994: 'terra',
    27424: 'aqua',
    26407: 'gps_iir_5',
    38833: 'gps_iif_2',
    39166: 'gps_iif_3',
    41866: 'goes16',
    43226: 'goes17',
    51850: 'goes18',
  }
  return known_satellites.get(norad_id, f'sat{norad_id}')

def download_tle_for_satellite(
  norad_id    : int,
  output_file : Optional[Union[str, Path]] = None,
) -> Tuple[str, str, float]:
  """
  Download TLE for a satellite from Celestrak.
  
  Input:
  ------
  norad_id : int
    NORAD catalog number.
  output_file : Optional[Union[str, Path]]
    Output file path.
  
  Output:
  -------
  Tuple[str, str, float]:
    (line1, line2, epoch_jd) TLE lines and epoch.
  """
  # Try multiple sources
  sources = [
    f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=TLE",
    f"https://celestrak.org/satcat/tle.php?CATNR={norad_id}",
  ]
  
  for url in sources:
    try:
      response = requests.get(url, timeout=10)
      if response.status_code == 200:
        lines = response.text.strip().split('\n')
        if len(lines) >= 2:
          # Find the two TLE lines (starting with '1 ' and '2 ')
          tle_lines = [l for l in lines if l.startswith('1 ') or l.startswith('2 ')]
          if len(tle_lines) >= 2:
            line1 = tle_lines[0]
            line2 = tle_lines[1]
            
            # Extract TLE epoch
            tle_epoch_str = line1[18:32]
            tle_year = int(tle_epoch_str[0:2])
            tle_year = 2000 + tle_year if tle_year < 57 else 1900 + tle_year
            tle_day_of_year = float(tle_epoch_str[2:])
            
            tle_epoch_time = Time(f"{tle_year}-01-01", format='iso') + TimeDelta((tle_day_of_year - 1) * u.day) # type: ignore
            epoch_jd = tle_epoch_time.jd
            
            print(f"Downloaded TLE for NORAD {norad_id}")
            print(f"  Epoch: {tle_epoch_time.iso} (JD {epoch_jd})")
            print(f"  Line 1: {line1}")
            print(f"  Line 2: {line2}")
            
            if output_file:
              with open(output_file, 'w') as f:
                f.write(f"{line1}\n{line2}\n")
              print(f"  Saved TLE to: {output_file}")
            
            return line1, line2, epoch_jd
    except Exception as e:
      print(f"Failed to download from {url}: {e}")
      continue
  
  raise RuntimeError(f"Could not download TLE for NORAD {norad_id}")

def download_historical_tles(
  norad_id    : int,
  start_time  : datetime,
  end_time    : datetime,
  output_file : Optional[Union[str, Path]] = None,
) -> List[Dict]:
  """
  Download historical TLEs from Space-Track.org for a given time range.
  
  NOTE: Requires Space-Track.org credentials. Set environment variables:
        SPACETRACK_USER and SPACETRACK_PASSWORD
  
  Input:
  ------
  norad_id : int
    NORAD catalog number.
  start_time : datetime
    Start of time range.
  end_time : datetime  
    End of time range.
  output_file : Optional[Union[str, Path]]
    Output file path.
  
  Output:
  -------
  List[Dict]:
    List of TLE data with keys 'line1', 'line2', 'epoch_jd'.
  """
  
  
  username = os.getenv('SPACETRACK_USER')
  password = os.getenv('SPACETRACK_PASSWORD')
  
  if not username or not password:
    print("ERROR: Space-Track credentials not found!")
    print("Set SPACETRACK_USER and SPACETRACK_PASSWORD environment variables to download historical TLEs.")
    print("You can register for a free account at: https://www.space-track.org/auth/createAccount")
    raise RuntimeError("Space-Track credentials are required for historical TLE download.")
  
  # Space-Track API
  base_url  = "https://www.space-track.org"
  login_url = f"{base_url}/ajaxauth/login"
  query_url = f"{base_url}/basicspacedata/query/class/gp_history/NORAD_CAT_ID/{norad_id}/orderby/EPOCH asc/EPOCH/{start_time.strftime('%Y-%m-%d')}--{end_time.strftime('%Y-%m-%d')}/format/3le"
  session   = requests.Session()
  
  try:
    # Login
    print(f"Logging into Space-Track.org ...")
    response = session.post(login_url, data={'identity': username, 'password': password})
    if response.status_code != 200:
      raise RuntimeError(f"Login failed: {response.status_code}")
    
    # Query TLEs
    print(f"Querying TLEs for NORAD {norad_id} from {start_time} to {end_time} ...")
    response = session.get(query_url)
    if response.status_code != 200:
      raise RuntimeError(f"Query failed: {response.status_code}")
    
    # Parse TLEs
    lines = response.text.strip().split('\n')
    tle_data = []
    
    for i in range(0, len(lines)-1, 2):
      if lines[i].startswith('1 ') and lines[i+1].startswith('2 '):
        # Extract TLE lines
        line1 = lines[i].strip()
        line2 = lines[i+1].strip()
        
        # Extract epoch
        tle_epoch_str   = line1[18:32]
        tle_year        = int(tle_epoch_str[0:2])
        tle_year        = 2000 + tle_year if tle_year < 57 else 1900 + tle_year
        tle_day_of_year = float(tle_epoch_str[2:])
        tle_epoch_time  = Time(f"{tle_year}-01-01", format='iso') + TimeDelta((tle_day_of_year - 1) * u.day) # type: ignore
        
        # Store TLE data
        tle_data.append({
          'line1'     : line1,
          'line2'     : line2,
          'epoch_jd'  : tle_epoch_time.jd,
          'epoch_iso' : tle_epoch_time.iso,
        })
    
    print(f"Retrieved {len(tle_data)} TLEs")
    
    if output_file:
      with open(output_file, 'w') as f:
        for tle in tle_data:
          f.write(f"{tle['line1']}\n{tle['line2']}\n")
      print(f"Saved TLEs to: {output_file}")
    
    return tle_data
    
  except Exception as e:
    print(f"Error downloading from Space-Track: {e}")
    raise RuntimeError(f"Failed to download TLEs from Space-Track for NORAD {norad_id}") from e
  finally:
    session.close()

def download_horizons_ephemeris(
  norad_id    : int,
  start_time  : datetime,
  end_time    : datetime,
  step        : str = '1h',
  output_file : Optional[Union[str, Path]] = None,
) -> Table:
  """
  Download satellite ephemeris from HORIZONS.
  
  Input:
  ------
  norad_id : int
    NORAD catalog number.
  start_time : datetime
    Start time for ephemeris.
  end_time : datetime
    End time for ephemeris.
  step : str
    Time step (default '1h').
  output_file : Optional[Union[str, Path]]
    Output file path.
  
  Output:
  -------
  Table:
    Astropy Table with ephemeris data.
  """
  print(f"Downloading HORIZONS ephemeris for NORAD {norad_id}")
  print(f"Time range: {start_time} to {end_time}")
  print(f"Time step: {step}")
  
  # Query HORIZONS using NORAD catalog number (negative ID format)
  sat_id = -(100000 + norad_id)
  obj = Horizons(
    id       = f"{sat_id}",
    location = '@399',  # Earth center (geocentric)
    epochs   = {'start' : start_time.strftime('%Y-%m-%d %H:%M'),
                'stop'  : end_time.strftime('%Y-%m-%d %H:%M'),
                'step'  : step}
  )
  
  # Get vectors in ICRF/J2000 equatorial frame
  vectors = obj.vectors(refplane='earth', cache=False)
  
  print(f"Retrieved {len(vectors)} data points")
  print(f"Reference frame: ICRF/J2000 Earth equatorial (refplane='earth')")
  
  # Add UTC time columns manually (TDB to UTC conversion)
  tdb_times = Time(vectors['datetime_jd'], format='jd', scale='tdb')
  utc_times = tdb_times.utc
  
  # Calculate TDB-UTC offset
  tdb_utc_offset_sec = (tdb_times.jd - utc_times.jd) * 86400.0  # Convert days to seconds
  
  # Add and convert columns
  vectors['datetime'] = [t.iso for t in utc_times]
  vectors['tdb_utc_offset'] = tdb_utc_offset_sec
  
  # Convert position from AU to meters
  vectors['pos_x'] = vectors['x'] * CONVERTER.M_PER_AU
  vectors['pos_y'] = vectors['y'] * CONVERTER.M_PER_AU
  vectors['pos_z'] = vectors['z'] * CONVERTER.M_PER_AU
  
  # Convert velocity from AU/day to m/s
  vectors['vel_x'] = vectors['vx'] * CONVERTER.M_PER_SEC__PER__AU_PER_DAY
  vectors['vel_y'] = vectors['vy'] * CONVERTER.M_PER_SEC__PER__AU_PER_DAY
  vectors['vel_z'] = vectors['vz'] * CONVERTER.M_PER_SEC__PER__AU_PER_DAY
  
  # Convert lighttime from days to seconds
  vectors['lighttime'] = vectors['lighttime'] * CONVERTER.SEC_PER_DAY
  
  # Convert range from AU to meters
  vectors['range'] = vectors['range'] * CONVERTER.M_PER_AU
  
  # Convert range_rate from AU/day to m/s
  vectors['range_rate'] = vectors['range_rate'] * CONVERTER.M_PER_SEC__PER__AU_PER_DAY
  
  # Reorder columns with new names (UTC only)
  column_order = [
    'targetname',
    'datetime',           # UTC ISO string
    'tdb_utc_offset',     # Time scale offset (seconds)
    'pos_x',              # Position X (meters)
    'pos_y',              # Position Y (meters)
    'pos_z',              # Position Z (meters)
    'vel_x',              # Velocity X (m/s)
    'vel_y',              # Velocity Y (m/s)
    'vel_z',              # Velocity Z (m/s)
    'lighttime',          # Light time (seconds)
    'range',              # Range (meters)
    'range_rate',         # Range rate (m/s)
  ]
  
  # Reorder the table
  vectors = vectors[column_order]
  
  print(f"Columns (reordered with SI units): {vectors.colnames}")
  print(f"Time scale: UTC (Coordinated Universal Time)")
  print(f"TDB-UTC offset: ~{np.mean(tdb_utc_offset_sec):.3f} ± {np.std(tdb_utc_offset_sec):.3f} sec")
  print(f"Units: Position [m], Velocity [m/s], Time [s], Range [m], Range-rate [m/s]")
  
  if output_file:
    # Write to CSV with custom header including units row
    import csv
    
    # First write using astropy to get the data
    vectors.write(output_file, format='csv', overwrite=True)
    
    # Read the file back
    with open(output_file, 'r') as f:
      lines = f.readlines()
    
    # Insert units row after header
    header = lines[0].strip()
    units_row = ',iso_utc,s,m,m,m,m/s,m/s,m/s,s,m,m/s\n'
    
    # Write back with units row
    with open(output_file, 'w') as f:
      f.write(header + '\n')
      f.write(units_row)
      f.writelines(lines[1:])
    
    print(f"Saved ephemeris to: {output_file}")
    file_size = Path(output_file).stat().st_size / (1024*1024)
    print(f"File size: {file_size:.2f} MB")
  
  return vectors

def download_ephems_and_tles(
  norad_id   : int,
  start_time : datetime,
  end_time   : datetime,
  step       : str = '1h',
) -> Dict:
  """
  Download both HORIZONS ephemeris and TLE history for a satellite.
  
  Input:
  ------
  norad_id : int
    NORAD catalog number.
  start_time : datetime
    Start time.
  end_time : datetime
    End time.
  step : str
    Time step for ephemeris (default: '1h').
  
  Output:
  -------
  Dict:
    Dictionary with file paths and data.
  """
  # Create output directories
  EPHEM_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
  TLE_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
  
  sat_name            = get_satellite_name(norad_id)
  start_timestamp_str = start_time.strftime('%Y%m%dT%H%M%SZ')
  end_timestamp_str   = end_time.strftime('%Y%m%dT%H%M%SZ')
  
  # Define output file paths
  horizons_file    = EPHEM_OUTPUT_DIR / f'horizons_ephem_{norad_id}_{sat_name}_{start_timestamp_str}_{end_timestamp_str}_{step}.csv'
  tle_history_file = TLE_OUTPUT_DIR / f'celestrak_tles_{norad_id}_{sat_name}_{start_timestamp_str}_{end_timestamp_str}.txt'
  
  horizons_status = "skipped"
  tle_status      = "skipped"
  
  print()
  print("="*80)
  print(f"PROCESSING STARTED")
  print("="*80)
  print(f"Satellite NORAD ID          : {norad_id}")
  print(f"Satellite name              : {sat_name.upper()}")
  print(f"Time range                  : {start_time} to {end_time}")
  print(f"Ephemeris output folderpath : {EPHEM_OUTPUT_DIR}")
  print(f"TLEs output folderpath      : {TLE_OUTPUT_DIR}")
  
  # Download HORIZONS ephemeris
  print("\n" + "="*80)
  print("PROCESSING HORIZONS EPHEMERIS")
  print("="*80)
  if horizons_file.exists():
    print(f"HORIZONS file already exists, skipping download:\n  {horizons_file}")
    horizons_data = None # Data not loaded, just confirming file exists
  else:
    horizons_data = download_horizons_ephemeris(
      norad_id    = norad_id,
      start_time  = start_time,
      end_time    = end_time,
      step        = step,
      output_file = horizons_file
    )
    horizons_status = "downloaded"
  
  # Download TLE history
  print("\n" + "="*80)
  print("PROCESSING TLE HISTORY")
  print("="*80)
  if tle_history_file.exists():
    print(f"TLE history file already exists, skipping download:\n  {tle_history_file}")
    tle_history = None # Data not loaded, just confirming file exists
  else:
    tle_history = download_historical_tles(
      norad_id    = norad_id,
      start_time  = start_time,
      end_time    = end_time,
      output_file = tle_history_file
    )
    tle_status = "downloaded"
  
  print("\n" + "="*80)
  print("PROCESSING COMPLETE")
  print("="*80)
  print(f"HORIZONS file    : {horizons_status}")
  print(f"TLE history file : {tle_status}")
  print()
  
  return {
    'horizons_file'    : horizons_file,
    'tle_history_file' : tle_history_file,
    'horizons_data'    : horizons_data,
    'tle_history'      : tle_history,
    'sat_name'         : sat_name,
    'norad_id'         : norad_id,
  }

def parse_command_line_args() -> Tuple[int, datetime, datetime, str]:
  """
  Parse and validate command line arguments.
  
  Output:
  -------
  Tuple[int, datetime, datetime, str]:
    (norad_id, start_time, end_time, step).
  
  Raises:
  -------
  SystemExit:
    If arguments are invalid.
  """
  if len(sys.argv) < 4:
    print("Usage: python -m src.download.ephems_and_tles <norad_id> <start_time> <end_time> [step]")
    print("\nExample:")
    print('  python -m src.download.ephems_and_tles 25544 "2025-10-01" "2025-10-08"')
    print('  python -m src.download.ephems_and_tles 25544 "2025-10-01T00:00:00Z" "2025-10-02T00:00:00Z" 1m')
    print('  python -m src.download.ephems_and_tles 25544 "2025-10-01 00:00" "2025-10-08 00:00"')
    print("\nTime format (UTC assumed):")
    print("  YYYY-MM-DD")
    print("  YYYY-MM-DD HH:MM")
    print("  YYYY-MM-DD HH:MM:SS")
    print("  YYYY-MM-DDTHH:MM")
    print("  YYYY-MM-DDTHH:MM:SS")
    print("  YYYY-MM-DDTHH:MM:SSZ")
    print("\nCommon satellites:")
    print("  25544 - ISS")
    print("  41866 - GOES-16")
    print("  43226 - GOES-17")
    print("  51850 - GOES-18")
    sys.exit(1)
  
  # Parse norad id
  norad_id  = int(sys.argv[1])

  # Parse times
  start_str = sys.argv[2]
  end_str   = sys.argv[3]
  try:
    step = sys.argv[4]
  except IndexError:
    step = '1h'
  
  try:
    start_time = parse_time(start_str)
    end_time   = parse_time(end_str)
  except ValueError as e:
    print(f"Error parsing dates: {e}")
    print('Supported formats (UTC assumed):')
    print('  "YYYY-MM-DD"')
    print('  "YYYY-MM-DD HH:MM"')
    print('  "YYYY-MM-DD HH:MM:SS"')
    print('  "YYYY-MM-DDTHH:MM"')
    print('  "YYYY-MM-DDTHH:MM:SS"')
    print('  "YYYY-MM-DDTHH:MM:SSZ"')
    sys.exit(1)
  
  # Validate time range
  if end_time <= start_time:
    print("Error: End time must be after start time")
    sys.exit(1)

  # Validate time step
  if not step.endswith(('m', 'h', 'd')):
    print("Error: Step must end with 'm' (minutes), 'h' (hours), or 'd' (days)")
    sys.exit(1)
  
  step_val_str = step[:-1]
  if not step_val_str.isdigit() or int(step_val_str) <= 0:
    print("Error: Step must be a positive integer followed by 'm', 'h', or 'd'")
    sys.exit(1)
  
  return norad_id, start_time, end_time, step

if __name__ == "__main__":
  norad_id, start_time, end_time, step = parse_command_line_args()
  download_ephems_and_tles(norad_id, start_time, end_time, step)
