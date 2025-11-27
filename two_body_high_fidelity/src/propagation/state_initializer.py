import numpy as np

from datetime import datetime, timedelta
from sgp4.api import Satrec
from typing import Optional

from src.propagation.propagator  import propagate_tle
from src.model.time_converter    import utc_to_et


def get_initial_state(
  tle_line_1           : str,
  tle_line_2           : str,
  desired_time_o_dt    : datetime,
  result_horizons      : Optional[dict],
  initial_state_source : str = 'jpl_horizons',
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
    desired_time_o_dt : datetime
      The start time of the integration.
    result_horizons : dict | None
      The dictionary containing Horizons ephemeris data.
    initial_state_source : str
      Source for the initial state ('jpl_horizons' or 'tle').
    to_j2000 : bool
      Flag to indicate if the output state should be in the J2000 frame.
  
  Output:
  -------
    np.ndarray
      A 6x1 state vector [m, m, m, m/s, m/s, m/s].
  """
  print("\nInitial State")
  
  # Determine if we should use Horizons
  use_horizons = 'jpl_horizons' in initial_state_source.lower()

  # 1. Use Horizons if available and requested
  if use_horizons and result_horizons and result_horizons.get('success'):
    horizons_initial_state = result_horizons['state'][:, 0]
    
    epoch_dt = result_horizons['time_o']
    try:
      epoch_et = utc_to_et(epoch_dt)
      et_str   = f" ({epoch_et:.6f} ET)"
    except:
      et_str = ""

    print(f"  JPL-Horizons-Derived")
    print(f"    Epoch    : {epoch_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC{et_str}")
    print(f"    Frame    : J2000")
    print(f"    Position : {horizons_initial_state[0]:>19.12e}  {horizons_initial_state[1]:>19.12e}  {horizons_initial_state[2]:>19.12e} m")
    print(f"    Velocity : {horizons_initial_state[3]:>19.12e}  {horizons_initial_state[4]:>19.12e}  {horizons_initial_state[5]:>19.12e} m/s")
    return horizons_initial_state

  # 2. Fallback to TLE
  print(f"  TLE-Derived")
  print(f"    TLE Line 1 : {tle_line_1}")
  print(f"    TLE Line 2 : {tle_line_2}")

  # Calculate integ_time_o (seconds from TLE epoch)
  satellite = Satrec.twoline2rv(tle_line_1, tle_line_2)
  year = satellite.epochyr
  if year < 57: year += 2000
  else: year += 1900
  epoch_days = satellite.epochdays
  tle_epoch_dt = datetime(year, 1, 1) + timedelta(days=epoch_days - 1)
  integ_time_o = (desired_time_o_dt - tle_epoch_dt).total_seconds()

  result_tle_initial = propagate_tle(
    tle_line_1 = tle_line_1,
    tle_line_2 = tle_line_2,
    time_o     = integ_time_o,
    time_f     = integ_time_o,
    num_points = 1,
    to_j2000   = to_j2000,
  )
  if not result_tle_initial['success']:
    raise RuntimeError(f"Failed to get initial state from TLE: {result_tle_initial['message']}")

  tle_initial_state = result_tle_initial['state'][:, 0]
  print(f"    Epoch      : {integ_time_o:>19.12e} s (from TLE epoch)")
  print(f"    Position   : [{tle_initial_state[0]:>19.12e}, {tle_initial_state[1]:>19.12e}, {tle_initial_state[2]:>19.12e}] m")
  print(f"    Velocity   : [{tle_initial_state[3]:>19.12e}, {tle_initial_state[4]:>19.12e}, {tle_initial_state[5]:>19.12e}] m/s")

  return tle_initial_state
