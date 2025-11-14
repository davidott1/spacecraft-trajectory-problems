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
  filepath: str,
  time_o: float = 0.0,
  time_f: float = None,
) -> dict:
  """
  Load Horizons ephemeris CSV file and convert to standard format.
  
  Input:
  ------
  filepath : str
      Path to Horizons CSV file
  time_o : float
      Reference time (epoch) in seconds [s]
  time_f : float, optional
      Final time in seconds [s]. If provided, only load data up to this time.
  
  Output:
  -------
  dict : Dictionary containing:
      - success : bool
      - message : str
      - time : np.ndarray - Time array relative to time_o [s]
      - state : np.ndarray - State history [6 x N] [m, m/s]
      - epoch : datetime - Reference epoch
  
  Notes:
  ------
  Expects CSV format with columns:
  - datetime : ISO UTC timestamp
  - pos_x, pos_y, pos_z : Position in meters (J2000 frame)
  - vel_x, vel_y, vel_z : Velocity in m/s (J2000 frame)
  
  IMPORTANT: Ensure that when downloading from Horizons:
  - Reference frame is set to J2000 (not Earth Mean Equator)
  - Center is set to Earth Geocenter (500@399)
  - Units are meters and m/s
  - Time step matches your propagation needs
  """
  try:
    # Read CSV file
    df = pd.read_csv(filepath)
    
    # Debug: Print first few rows and column names
    print(f"  CSV columns: {list(df.columns)}")
    print(f"  First data row sample: {df.iloc[0]['pos_x'] if len(df) > 0 else 'N/A'}")
    
    # Skip the units row (second row where pos_x = 'm')
    # Check if first row contains units
    if str(df.iloc[0]['pos_x']).strip().lower() == 'm':
      print("  Detected units row, skipping...")
      df = df.iloc[1:].reset_index(drop=True)
    
    # Extract position and velocity (already in meters and m/s based on CSV)
    pos_x = pd.to_numeric(df['pos_x'], errors='coerce').values
    pos_y = pd.to_numeric(df['pos_y'], errors='coerce').values
    pos_z = pd.to_numeric(df['pos_z'], errors='coerce').values
    vel_x = pd.to_numeric(df['vel_x'], errors='coerce').values
    vel_y = pd.to_numeric(df['vel_y'], errors='coerce').values
    vel_z = pd.to_numeric(df['vel_z'], errors='coerce').values
    
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
    
    # Parse first datetime as epoch
    epoch_str = datetime_series.iloc[0]
    epoch = datetime.fromisoformat(epoch_str.replace(' ', 'T'))
    
    # Create time array from datetime differences
    # This is more accurate than assuming fixed intervals
    datetimes = pd.to_datetime(datetime_series)
    time_deltas = (datetimes - datetimes.iloc[0]).dt.total_seconds()
    time = time_deltas.values
    
    # Adjust time relative to time_o
    time = time + time_o
    
    # Limit to time_f if provided
    if time_f is not None:
      time_mask = time <= time_f
      time = time[time_mask]
      pos_x = pos_x[time_mask]
      pos_y = pos_y[time_mask]
      pos_z = pos_z[time_mask]
      vel_x = vel_x[time_mask]
      vel_y = vel_y[time_mask]
      vel_z = vel_z[time_mask]
    
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
    print(f"  First position: [{state[0,0]/1e3:.3f}, {state[1,0]/1e3:.3f}, {state[2,0]/1e3:.3f}] km")
    print(f"  First velocity: [{state[3,0]/1e3:.3f}, {state[4,0]/1e3:.3f}, {state[5,0]/1e3:.3f}] km/s")
    print(f"  Time step: {time[1] - time[0] if len(time) > 1 else 0:.1f} seconds")
    
    return {
      'success': True,
      'message': 'Horizons ephemeris loaded successfully',
      'time': time,
      'state': state,
      'final_state': state[:, -1],
      'coe': {},  # COE will be computed if needed
      'epoch': epoch,
    }
    
  except Exception as e:
    import traceback
    return {
      'success': False,
      'message': f'Failed to load Horizons ephemeris: {str(e)}\n{traceback.format_exc()}',
      'time': np.array([]),
      'state': np.zeros((6, 0)),
      'final_state': np.zeros(6),
      'coe': {},
      'epoch': None,
    }
