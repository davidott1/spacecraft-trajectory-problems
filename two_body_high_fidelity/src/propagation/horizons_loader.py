"""
Horizons Ephemeris Loader
=========================

Load and process JPL Horizons ephemeris data for comparison with propagators.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
from src.utility.parse import parse_time


def load_horizons_ephemeris(
  filepath : str,
  time_start_dt : Optional[datetime] = None,
  time_end_dt   : Optional[datetime] = None,
) -> dict:
  """
  Load ephemeris from a JPL Horizons CSV file.
  
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
      'delta_time' : delta_time,
      'state'      : state,
    }

  except Exception as e:
    return {
      'success'    : False,
      'message'    : str(e),
      'time_o'     : [],
      'delta_time' : [],
      'state'      : [],
    }
