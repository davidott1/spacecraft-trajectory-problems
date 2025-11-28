from typing   import Optional
from datetime import datetime, timedelta


def determine_actual_times(
  result_jpl_horizons_ephemeris : Optional[dict],
  desired_time_o_dt             : datetime,
  desired_time_f_dt             : datetime,
) -> tuple[datetime, datetime]:
  """
  Determine the actual start and end times for propagation.
  
  If Horizons data is available, aligns the time grid with Horizons.
  Otherwise, uses the desired start and end times.
  
  Input:
  ------
    result_jpl_horizons_ephemeris : dict | None
      Result from Horizons loader.
    desired_time_o_dt : datetime
      Desired initial datetime.
    desired_time_f_dt : datetime
      Desired final datetime.
      
  Output:
  -------
    tuple[datetime, datetime]
      Tuple containing (actual_time_o_dt, actual_time_f_dt).
  """
  if result_jpl_horizons_ephemeris and result_jpl_horizons_ephemeris.get('success'):
    actual_time_o_dt = result_jpl_horizons_ephemeris['time_o']
    actual_time_f_dt = result_jpl_horizons_ephemeris['time_f']
  else:
    actual_time_o_dt = desired_time_o_dt
    actual_time_f_dt = desired_time_f_dt
    
  return actual_time_o_dt, actual_time_f_dt
