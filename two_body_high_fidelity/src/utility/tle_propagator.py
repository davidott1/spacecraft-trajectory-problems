import numpy as np
import math
from sgp4.api                    import Satrec, jday
from datetime                    import datetime, timedelta
from src.model.dynamics          import OrbitConverter
from src.model.constants         import PHYSICALCONSTANTS
from src.model.frame_conversions import FrameConversions
from typing                      import Optional


def modify_tle_bstar(
  tle_line1   : str,
  bstar_value : float = 0.0,
) -> str:
  """
  Modify the B* drag term in TLE line 1.
  
  Input:
  ------
    tle_line1 : str
      First line of TLE.
    bstar_value : float
      New B* value (default 0.0).
  
  Output:
  -------
    str
      Modified TLE line 1 with new B* value.
  """
  # B* is in columns 54-61 (0-indexed: 53-60) in format: ±.nnnnn±n
  # Example: " 12345-3" means 0.12345 × 10^-3
  
  if bstar_value == 0.0:
    bstar_str = " 00000-0"
  else:
    # Convert to scientific notation and format
    if bstar_value != 0:
      exponent = math.floor(math.log10(abs(bstar_value)))
      mantissa = bstar_value / (10 ** exponent)
      sign = '-' if bstar_value < 0 else ' '
      exp_sign = '-' if exponent < 0 else '+'
      bstar_str = f"{sign}{abs(mantissa):.5f}[1:]{exp_sign}{abs(exponent)}"
    else:
      bstar_str = " 00000-0"
  
  # Replace B* in TLE line 1 (columns 53-60)
  modified_line1 = tle_line1[:53] + bstar_str + tle_line1[61:]
  
  # Recalculate checksum (last character)
  checksum = 0
  for char in modified_line1[:-1]:
    if char.isdigit():
      checksum += int(char)
    elif char == '-':
      checksum += 1
  modified_line1 = modified_line1[:-1] + str(checksum % 10)
  
  return modified_line1


def propagate_tle(
  tle_line1       : str,
  tle_line2       : str,
  time_o          : Optional[float] = None,
  time_f          : Optional[float] = None,
  num_time_points : int  = 100,
  time_eval       : Optional[np.ndarray] = None,
  to_j2000        : bool = False,
  disable_drag    : bool = False,
) -> dict:
  """
  Propagate orbit from TLE using SGP4.
  
  Input:
  ------
    tle_line1 : str
      First line of TLE.
    tle_line2 : str
      Second line of TLE.
    time_o : float, optional
      Initial time in seconds from TLE epoch. Required if time_eval is not provided.
    time_f : float, optional
      Final time in seconds from TLE epoch. Required if time_eval is not provided.
    num_time_points : int
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
    tle_line1 = modify_tle_bstar(tle_line1, 0.0)
  
  try:
    # Create satellite object from TLE
    satellite = Satrec.twoline2rv(tle_line1, tle_line2)
    
    # Extract epoch from TLE
    year = satellite.epochyr
    if year < 57:
      year += 2000
    else:
      year += 1900
    
    epoch_days = satellite.epochdays
    epoch_datetime = datetime(year, 1, 1) + timedelta(days=epoch_days - 1)
    
    # Generate time array
    if time_eval is not None:
      time            = time_eval
      num_time_points = len(time_eval)
    else:
      time = np.linspace(time_o, time_f, num_time_points) # type: ignore
    
    # Initialize arrays
    state_array = np.zeros((6, num_time_points))
    coe_time_series = {
      'sma'  : np.zeros(num_time_points),
      'ecc'  : np.zeros(num_time_points),
      'inc'  : np.zeros(num_time_points),
      'raan' : np.zeros(num_time_points),
      'argp' : np.zeros(num_time_points),
      'ma'   : np.zeros(num_time_points),
      'ta'   : np.zeros(num_time_points),
      'ea'   : np.zeros(num_time_points),
    }

    # Propagate at each time step
    for i, t in enumerate(time):
      # Convert time to datetime
      dt = epoch_datetime + timedelta(seconds=float(t))
      
      # Get Julian date
      jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond/1e6)
      
      # Propagate
      error_code, teme_pos_vec, teme_vel_vec = satellite.sgp4(jd, fr)
      if error_code != 0:
        return {
          'success' : False,
          'message' : f'SGP4 error code: {error_code}',
          'time'    : time,
          'state'   : state_array,
          'coe'     : coe_time_series,
        }
      
      # Transform TEME to J2000/GCRS
      if to_j2000:
        j2000_pos_vec, j2000_vel_vec = FrameConversions.teme_to_j2000(teme_pos_vec, teme_vel_vec, jd + fr)
        pos_vec = np.array(j2000_pos_vec) * 1000.0  # km -> m
        vel_vec = np.array(j2000_vel_vec) * 1000.0  # km/s -> m/s
      else:
        pos_vec = np.array(teme_pos_vec) * 1000.0  # km -> m
        vel_vec = np.array(teme_vel_vec) * 1000.0  # km/s -> m/s
      
      # Store state
      state_array[0:3, i] = pos_vec
      state_array[3:6, i] = vel_vec
      
      # Compute osculating elements
      coe = OrbitConverter.pv_to_coe(
        pos_vec,
        vel_vec,
        gp=PHYSICALCONSTANTS.EARTH.GP,
      )
      for key in coe_time_series.keys():
        if coe[key] is not None:
          coe_time_series[key][i] = coe[key]
    
    return {
      'success'     : True,
      'message'     : 'SGP4 propagation successful',
      'time'        : time,
      'state'       : state_array,
      'final_state' : state_array[:, -1],
      'coe'         : coe_time_series,
    }
  except Exception as e:
    return {
      'success' : False,
      'message' : str(e),
      'time'    : [],
      'state'   : [],
      'coe'     : [],
    }


def get_tle_initial_state(
  tle_line1    : str,
  tle_line2    : str,
  disable_drag : bool = False,
  to_j2000     : bool = True,
) -> np.ndarray:
  """
  Extract initial position and velocity from TLE at epoch.
  
  Input:
  ------
    tle_line1 : str
      First line of TLE.
    tle_line2 : str
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
    tle_line1 = modify_tle_bstar(tle_line1, 0.0)
  
  satellite = Satrec.twoline2rv(tle_line1, tle_line2)
  
  # Get epoch
  year = satellite.epochyr
  if year < 57:
    year += 2000
  else:
    year += 1900
  
  epoch_days     = satellite.epochdays
  epoch_datetime = datetime(year, 1, 1) + timedelta(days=epoch_days - 1)
  
  jd, fr = jday(epoch_datetime.year, epoch_datetime.month, epoch_datetime.day,
                epoch_datetime.hour, epoch_datetime.minute, 
                epoch_datetime.second + epoch_datetime.microsecond/1e6)
  
  error_code, teme_pos_vec, teme_vel_vec = satellite.sgp4(jd, fr)
  
  if error_code != 0:
    raise ValueError(f'SGP4 error code: {error_code}')
  
  # Transform TEME to J2000/GCRS if requested
  if to_j2000:
    j2000_pos_vec, j2000_vel_vec = FrameConversions.teme_to_j2000(teme_pos_vec, teme_vel_vec, jd + fr, units_pos='m', units_vel='m/s')
    pos_vec = np.array(j2000_pos_vec)
    vel_vec = np.array(j2000_vel_vec)
  else:
    # Convert from km to m and km/s to m/s
    pos_vec = np.array(teme_pos_vec) * 1000.0
    vel_vec = np.array(teme_vel_vec) * 1000.0

  return np.concatenate([pos_vec, vel_vec])
