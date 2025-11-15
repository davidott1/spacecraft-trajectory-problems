"""
Horizons Ephemeris Loader
=========================

Load and process JPL Horizons ephemeris data for comparison with propagators.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def load_horizons_ephemeris(
  filepath  : str,
  start_dt  : datetime = None, # type: ignore
  end_dt    : datetime = None, # type: ignore
) -> dict:
  """
  Load Horizons ephemeris CSV file and extract data between start and end times.
  
  Input:
  ------
  filepath : str
      Path to Horizons CSV file
  start_dt : datetime, optional
      Start time for data extraction. If None, uses first time in file.
  end_dt : datetime, optional
      End time for data extraction. If None, uses last time in file.
  
  Output:
  -------
  dict : Dictionary containing:
      - success     : bool
      - message     : str
      - time        : np.ndarray - Time array starting from 0 [s]
      - state       : np.ndarray - State history [6 x N] [m, m/s]
      - epoch       : datetime - Reference epoch (start_dt or first time in file)
      - plot_time_s : np.ndarray - Same as time (for plotting compatibility)
  
  Notes:
  ------
  Expects CSV format with columns:
  - datetime : ISO UTC timestamp
  - pos_x, pos_y, pos_z : Position (auto-detects m or km from units row)
  - vel_x, vel_y, vel_z : Velocity (auto-detects m/s or km/s from units row)
  
  The loader automatically detects units from the second row if present.
  Output is always in meters and m/s regardless of input units.
  """
  try:
    # Read CSV, check for units row
    df = pd.read_csv(filepath)
    
    # Debug: Print first few rows and column names
    print(f"  CSV columns: {list(df.columns)}")
    
    # Detect units from second row (index 1 after header)
    pos_unit_multiplier = 1.0  # Default to meters
    vel_unit_multiplier = 1.0  # Default to m/s
    
    # Check if row 1 (second row) contains units
    if len(df) > 1:
      first_val = str(df.iloc[0]['pos_x']).strip().lower()
      second_val = str(df.iloc[1]['pos_x']).strip().lower()
      
      # If first row contains unit strings, use it for detection and skip it
      if first_val in ['m', 'km', 'km/s', 'm/s']:
        print(f"  Detected units in first data row: pos={first_val}")
        if 'km' in first_val:
          pos_unit_multiplier = 1000.0  # km to m
          vel_unit_multiplier = 1000.0  # km/s to m/s
        df = pd.read_csv(filepath, skiprows=[1])  # Re-read, skipping units row
      # If second row contains unit strings, the first row is data
      elif second_val in ['m', 'km', 'km/s', 'm/s']:
        print(f"  Detected units in second row: pos={second_val}")
        if 'km' in second_val:
          pos_unit_multiplier = 1000.0  # km to m
          vel_unit_multiplier = 1000.0  # km/s to m/s
        df = pd.read_csv(filepath, skiprows=[1])  # Re-read, skipping units row
      else:
        # No units row detected, try to infer from data magnitude
        sample_pos = abs(pd.to_numeric(df.iloc[0]['pos_x'], errors='coerce'))
        if sample_pos < 10000:  # Likely in thousands of km
          print(f"  No units row found. Data magnitude suggests km (sample: {sample_pos:.1f})")
          pos_unit_multiplier = 1000.0
          vel_unit_multiplier = 1000.0
        else:
          print(f"  No units row found. Data magnitude suggests m (sample: {sample_pos:.1f})")
    
    # Extract position and velocity, converting to floats
    pos_x = pd.to_numeric(df['pos_x'], errors='coerce').values * pos_unit_multiplier
    pos_y = pd.to_numeric(df['pos_y'], errors='coerce').values * pos_unit_multiplier
    pos_z = pd.to_numeric(df['pos_z'], errors='coerce').values * pos_unit_multiplier
    vel_x = pd.to_numeric(df['vel_x'], errors='coerce').values * vel_unit_multiplier
    vel_y = pd.to_numeric(df['vel_y'], errors='coerce').values * vel_unit_multiplier
    vel_z = pd.to_numeric(df['vel_z'], errors='coerce').values * vel_unit_multiplier
    
    # Remove any NaN rows
    valid_mask = ~(np.isnan(pos_x) | np.isnan(pos_y) | np.isnan(pos_z) | 
                   np.isnan(vel_x) | np.isnan(vel_y) | np.isnan(vel_z))
    
    pos_x = pos_x[valid_mask]
    pos_y = pos_y[valid_mask]
    pos_z = pos_z[valid_mask]
    vel_x = vel_x[valid_mask]
    vel_y = vel_y[valid_mask]
    vel_z = vel_z[valid_mask]
    
    # Parse datetime strings to create time array
    datetime_series = df['datetime'][valid_mask].reset_index(drop=True)
    
    # Parse datetimes
    datetimes = pd.to_datetime(datetime_series)
    
    # Determine start and end indices based on input datetimes
    if start_dt is not None:
      # Find closest index to start_dt
      start_idx = np.argmin(np.abs(datetimes - pd.Timestamp(start_dt)))
      epoch = start_dt
    else:
      start_idx = 0
      epoch = datetimes.iloc[0].to_pydatetime()
    
    if end_dt is not None:
      # Find closest index to end_dt (inclusive)
      end_idx = np.argmin(np.abs(datetimes - pd.Timestamp(end_dt)))
      # Make sure end_idx is after start_idx and includes the endpoint
      end_idx = max(end_idx + 1, start_idx + 1)
    else:
      end_idx = len(datetimes)
    
    # Slice the data
    datetimes = datetimes.iloc[start_idx:end_idx]
    pos_x = pos_x[start_idx:end_idx]
    pos_y = pos_y[start_idx:end_idx]
    pos_z = pos_z[start_idx:end_idx]
    vel_x = vel_x[start_idx:end_idx]
    vel_y = vel_y[start_idx:end_idx]
    vel_z = vel_z[start_idx:end_idx]
    
    # Create time array relative to epoch (starts at 0)
    time_deltas = (datetimes - datetimes.iloc[0]).dt.total_seconds()
    time = time_deltas.values
    
    # Create state array
    num_points = len(pos_x)
    state = np.zeros((6, num_points))
    state[0, :] = pos_x
    state[1, :] = pos_y
    state[2, :] = pos_z
    state[3, :] = vel_x
    state[4, :] = vel_y
    state[5, :] = vel_z
    
    # Print diagnostics
    print(f"  Loaded {num_points} data points")
    print(f"  Data time range: {datetimes.iloc[0]} to {datetimes.iloc[-1]}")
    print(f"  First position: [{state[0,0]/1e3:.3f}, {state[1,0]/1e3:.3f}, {state[2,0]/1e3:.3f}] km")
    print(f"  First velocity: [{state[3,0]/1e3:.3f}, {state[4,0]/1e3:.3f}, {state[5,0]/1e3:.3f}] km/s")
    print(f"  Time range: {time[0]:.1f} to {time[-1]:.1f} seconds from epoch")
    if len(time) > 1:
      print(f"  Time step: {time[1] - time[0]:.1f} seconds")
    
    return {
      'success'     : True,
      'message'     : 'Horizons ephemeris loaded successfully',
      'time'        : time,
      'plot_time_s' : time,  # Add for consistency with other propagators
      'state'       : state,
      'final_state' : state[:, -1],
      'coe'         : {},
      'epoch'       : epoch,
    }
    
  except Exception as e:
    import traceback
    return {
      'success'     : False,
      'message'     : f'Failed to load Horizons ephemeris: {str(e)}\n{traceback.format_exc()}',
      'time'        : np.array([]),
      'plot_time_s' : np.array([]),
      'state'       : np.zeros((6, 0)),
      'final_state' : np.zeros(6),
      'coe'         : {},
      'epoch'       : None,
    }
