import numpy as np

from datetime            import datetime, timedelta
from sgp4.api            import Satrec
from typing              import Optional

from src.propagation.propagator import propagate_tle
from src.model.time_converter   import utc_to_et
from src.model.dynamics         import OrbitConverter
from src.model.constants        import SOLARSYSTEMCONSTANTS, CONVERTER
from src.utility.time_helper    import format_time_offset


def get_initial_state(
  tle_line_1                    : Optional[str],
  tle_line_2                    : Optional[str],
  time_o_dt                     : datetime,
  result_jpl_horizons_ephemeris : Optional[dict],
  initial_state_source          : str = 'jpl_horizons',
  to_j2000                      : bool = True,
) -> np.ndarray:
  """
  Get initial state vector from specified source.
  
  Input:
  ------
    tle_line_1 : str | None
      TLE line 1.
    tle_line_2 : str | None
      TLE line 2.
    time_o_dt : datetime
      Initial time for propagation.
    result_jpl_horizons_ephemeris : dict | None
      JPL Horizons ephemeris result.
    initial_state_source : str
      Source for initial state ('jpl_horizons' or 'tle').
    to_j2000 : bool
      Convert to J2000 frame.
      
  Output:
  -------
    initial_state : np.ndarray
      Initial state vector [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z] in m and m/s.
  """
  print("\nInitial State")
  
  # Determine if we should use Horizons
  use_jpl_horizons = 'jpl_horizons' in initial_state_source.lower()

  # Use Horizons if available and requested
  if use_jpl_horizons and result_jpl_horizons_ephemeris and result_jpl_horizons_ephemeris.get('success'):
    # Use the first available state from the ephemeris (not interpolated)
    horizons_initial_state = result_jpl_horizons_ephemeris['state'][:, 0]
    
    # Get the actual epoch from the ephemeris
    epoch_dt = result_jpl_horizons_ephemeris['time_o']
    try:
      epoch_et = utc_to_et(epoch_dt)
      et_str   = f" / {epoch_et:.6f} ET"
    except Exception:
      et_str = ""

    # Convert position, velocity to classical orbital elements for display
    coe = OrbitConverter.pv_to_coe(
      pos_vec = horizons_initial_state[0:3],
      vel_vec = horizons_initial_state[3:6],
      gp      = SOLARSYSTEMCONSTANTS.EARTH.GP,
    )

    # Display Horizons-derived initial state
    print(f"  Source : JPL Horizons")
    print(f"  Epoch  : {epoch_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC{et_str}")
    print(f"  Frame  : J2000")
    print(f"  Cartesian State" )
    print(f"    Position :  {horizons_initial_state[0]:>19.12e}  {horizons_initial_state[1]:>19.12e}  {horizons_initial_state[2]:>19.12e} m")
    print(f"    Velocity :  {horizons_initial_state[3]:>19.12e}  {horizons_initial_state[4]:>19.12e}  {horizons_initial_state[5]:>19.12e} m/s")
    print(f"  Classical Orbital Elements")
    print(f"    SMA  :  {coe['sma' ]:19.12e} m")
    print(f"    ECC  :  {coe['ecc' ]:19.12e} -")
    print(f"    INC  :  {coe['inc' ]*CONVERTER.DEG_PER_RAD:19.12e} deg")
    print(f"    RAAN :  {coe['raan']*CONVERTER.DEG_PER_RAD:19.12e} deg")
    print(f"    AOP  :  {coe['aop' ]*CONVERTER.DEG_PER_RAD:19.12e} deg")
    print(f"    TA   :  {coe['ta'  ]*CONVERTER.DEG_PER_RAD:19.12e} deg")
    print(f"    EA   :  {coe['ea'  ]*CONVERTER.DEG_PER_RAD:19.12e} deg")
    print(f"    MA   :  {coe['ma'  ]*CONVERTER.DEG_PER_RAD:19.12e} deg")
    return horizons_initial_state

  # Fallback to TLE

  # Calculate integ_time_o (seconds from TLE epoch)
  satellite = Satrec.twoline2rv(tle_line_1, tle_line_2)
  year = satellite.epochyr
  if year < 57: 
    year += 2000
  else: 
    year += 1900
  epoch_days   = satellite.epochdays
  tle_epoch_dt = datetime(year, 1, 1) + timedelta(days=epoch_days - 1)
  integ_time_o = (time_o_dt - tle_epoch_dt).total_seconds()

  # Propagate TLE to get initial state
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

  # Extract initial state from TLE propagation result
  tle_initial_state = result_tle_initial['state'][:, 0]

  # Get ET for display
  try:
    epoch_et = utc_to_et(time_o_dt)
    et_str   = f" / {epoch_et:.6f} ET"
  except Exception:
    et_str = ""

  # Convert position, velocity to classical orbital elements for display
  coe = OrbitConverter.pv_to_coe(
    pos_vec = tle_initial_state[0:3],
    vel_vec = tle_initial_state[3:6],
    gp      = SOLARSYSTEMCONSTANTS.EARTH.GP,
  )

  # Display TLE-derived initial state
  print(f"  Source : Celestrak TLE")
  print(f"  Epoch  : {time_o_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC{et_str}")
  print(f"  Frame  : J2000")
  print(f"  Cartesian State")
  print(f"    Position :  {tle_initial_state[0]:>19.12e}  {tle_initial_state[1]:>19.12e}  {tle_initial_state[2]:>19.12e} m")
  print(f"    Velocity :  {tle_initial_state[3]:>19.12e}  {tle_initial_state[4]:>19.12e}  {tle_initial_state[5]:>19.12e} m/s")
  print(f"  Classical Orbital Elements")
  print(f"    SMA  :  {coe['sma' ]:19.12e} m")
  print(f"    ECC  :  {coe['ecc' ]:19.12e} -")
  print(f"    INC  :  {coe['inc' ]*CONVERTER.DEG_PER_RAD:19.12e} deg")
  print(f"    RAAN :  {coe['raan']*CONVERTER.DEG_PER_RAD:19.12e} deg")
  print(f"    AOP  :  {coe['aop' ]*CONVERTER.DEG_PER_RAD:19.12e} deg")
  print(f"    TA   :  {coe['ta'  ]*CONVERTER.DEG_PER_RAD:19.12e} deg")
  print(f"    EA   :  {coe['ea'  ]*CONVERTER.DEG_PER_RAD:19.12e} deg")
  print(f"    MA   :  {coe['ma'  ]*CONVERTER.DEG_PER_RAD:19.12e} deg")

  return tle_initial_state
