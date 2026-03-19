import numpy as np

from datetime            import datetime, timedelta
from sgp4.api            import Satrec
from typing              import Optional

from src.propagation.numerical_propagator import propagate_tle
from src.model.time_converter   import utc_to_et
from src.model.orbit_converter  import OrbitConverter
from src.model.constants        import SOLARSYSTEMCONSTANTS, CONVERTER
from src.schemas.propagation    import PropagationResult
from src.schemas.state           import CartesianState


def get_initial_state(
  tle_line_1                    : Optional[str],
  tle_line_2                    : Optional[str],
  time_o_dt                     : datetime,
  result_jpl_horizons_ephemeris : Optional[PropagationResult],
  initial_state_source          : str = 'jpl_horizons',
  custom_state_vector           : Optional[np.ndarray] = None,
  initial_state_filename        : Optional[str] = None,
) -> tuple[np.ndarray, datetime]:
  """
  Get initial state vector and epoch from specified source.

  Input:
  ------
    tle_line_1 : str | None
      TLE line 1.
    tle_line_2 : str | None
      TLE line 2.
    time_o_dt : datetime
      Desired initial time for propagation.
    result_jpl_horizons_ephemeris : PropagationResult | None
      JPL Horizons ephemeris result.
    initial_state_source : str
      Source for initial state ('jpl_horizons', 'tle', or 'custom_state_vector').
    custom_state_vector : np.ndarray | None
      Custom state vector if source is 'custom_state_vector'.
    initial_state_filename : str | None
      Filename of custom state vector file.

  Output:
  -------
    initial_state : np.ndarray
      Initial state vector [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z] in m and m/s.
    initial_epoch : datetime
      Actual epoch of the initial state.
  """
  title = "Initial State"
  print("\n" + "-" * len(title))
  print(title)
  print("-" * len(title))
  print()

  # Progress subsection
  print("  Progress")

  # Custom State Vector
  if initial_state_source == 'custom_state_vector':
    print("    Extract initial state from custom state vector file")
    print("    Convert to classical orbital elements for display")
    print()
    if custom_state_vector is None:
      raise ValueError("Custom state vector source selected but no vector provided.")

    # Handle CartesianState object or numpy array
    if isinstance(custom_state_vector, CartesianState):
      state_vec = custom_state_vector.state_vector
      frame_name = custom_state_vector.frame
    else:
      state_vec = custom_state_vector
      frame_name = "J2000 (Assumed)"

    # Get ET for display
    try:
      epoch_et = utc_to_et(time_o_dt)
      et_str   = f" / {epoch_et:.6f} ET"
    except Exception:
      et_str = ""

    # Convert position, velocity to classical orbital elements for display
    coe = OrbitConverter.pv_to_coe(
      pos_vec = state_vec[0:3],
      vel_vec = state_vec[3:6],
      gp      = SOLARSYSTEMCONSTANTS.EARTH.GP,
    )

    # Summary subsection
    print("  Summary")
    print(f"    Source : Custom State Vector File")
    print(f"    File   : {initial_state_filename}")
    print(f"    Epoch  : {time_o_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC{et_str}")
    print(f"    Frame  : {frame_name}")
    print(f"    Cartesian State")
    print(f"      Position :  {state_vec[0]:>19.12e}  {state_vec[1]:>19.12e}  {state_vec[2]:>19.12e} m")
    print(f"      Velocity :  {state_vec[3]:>19.12e}  {state_vec[4]:>19.12e}  {state_vec[5]:>19.12e} m/s")
    print(f"    Classical Orbital Elements")
    print(f"      SMA  :  {coe.sma:19.12e} m")
    print(f"      ECC  :  {coe.ecc:19.12e} -")
    print(f"      INC  :  {coe.inc  * CONVERTER.DEG_PER_RAD:19.12e} deg")
    print(f"      RAAN :  {coe.raan * CONVERTER.DEG_PER_RAD:19.12e} deg")
    print(f"      AOP  :  {coe.aop  * CONVERTER.DEG_PER_RAD:19.12e} deg")
    if coe.ta is not None:
      print(f"      TA   :  {coe.ta   * CONVERTER.DEG_PER_RAD:19.12e} deg")
    if coe.ea is not None:
      print(f"      EA   :  {coe.ea   * CONVERTER.DEG_PER_RAD:19.12e} deg")
    if coe.ma is not None:
      print(f"      MA   :  {coe.ma   * CONVERTER.DEG_PER_RAD:19.12e} deg")
    if coe.ha is not None:
      print(f"      HA   :  {coe.ha   * CONVERTER.DEG_PER_RAD:19.12e} deg")
    if coe.pa is not None:
      print(f"      PA   :  {coe.pa:19.12e} -")

    return state_vec, time_o_dt

  # Determine if we should use Horizons
  use_jpl_horizons = 'jpl_horizons' in initial_state_source.lower()

  # Use Horizons if available and requested
  if use_jpl_horizons and result_jpl_horizons_ephemeris and result_jpl_horizons_ephemeris.success:
    print("    Extract initial state from Horizons ephemeris")
    print("    Convert to classical orbital elements for display")
    print()

    # Get the Horizons time grid
    if not result_jpl_horizons_ephemeris.time:
      raise ValueError("Horizons ephemeris missing time grid data")

    # Use the first state from the filtered Horizons data
    horizons_initial_state = result_jpl_horizons_ephemeris.state[:, 0]
    epoch_dt = result_jpl_horizons_ephemeris.time.initial.utc

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
    # Summary subsection
    print("  Summary")
    print(f"    Source : JPL Horizons")
    print(f"    Epoch  : {epoch_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC{et_str}")
    print(f"    Frame  : J2000")
    print(f"    Cartesian State" )
    print(f"      Position :  {horizons_initial_state[0]:>19.12e}  {horizons_initial_state[1]:>19.12e}  {horizons_initial_state[2]:>19.12e} m")
    print(f"      Velocity :  {horizons_initial_state[3]:>19.12e}  {horizons_initial_state[4]:>19.12e}  {horizons_initial_state[5]:>19.12e} m/s")
    print(f"    Classical Orbital Elements")
    print(f"      SMA  :  {coe.sma                         :19.12e} m")
    print(f"      ECC  :  {coe.ecc                         :19.12e} -")
    print(f"      INC  :  {coe.inc  * CONVERTER.DEG_PER_RAD:19.12e} deg")
    print(f"      RAAN :  {coe.raan * CONVERTER.DEG_PER_RAD:19.12e} deg")
    print(f"      AOP  :  {coe.aop  * CONVERTER.DEG_PER_RAD:19.12e} deg")
    if coe.ta is not None:
      print(f"      TA   :  {coe.ta   * CONVERTER.DEG_PER_RAD:19.12e} deg")
    if coe.ea is not None:
      print(f"      EA   :  {coe.ea   * CONVERTER.DEG_PER_RAD:19.12e} deg")
    if coe.ma is not None:
      print(f"      MA   :  {coe.ma   * CONVERTER.DEG_PER_RAD:19.12e} deg")
    if coe.ha is not None:
      print(f"      HA   :  {coe.ha   * CONVERTER.DEG_PER_RAD:19.12e} deg")
    if coe.pa is not None:
      print(f"      PA   :  {coe.pa:19.12e} -")
    return horizons_initial_state, epoch_dt

  # Fallback to TLE
  print("    Propagate TLE to desired epoch")
  print("    Convert to classical orbital elements for display")
  print()

  # Validate TLE lines are available
  if tle_line_1 is None or tle_line_2 is None:
    raise ValueError("TLE source selected but TLE lines are not available.")

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
    to_j2000   = True,
  )
  if not result_tle_initial.success:
    raise RuntimeError(f"Failed to get initial state from TLE: {result_tle_initial.message}")

  # Extract initial state from TLE propagation result
  tle_initial_state = result_tle_initial.state[:, 0]

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
  # Summary subsection
  print("  Summary")
  print(f"    Source : Celestrak TLE")
  print(f"    Epoch  : {time_o_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC{et_str}")
  print(f"    Frame  : J2000")
  print(f"    Cartesian State")
  print(f"      Position :  {tle_initial_state[0]:>19.12e}  {tle_initial_state[1]:>19.12e}  {tle_initial_state[2]:>19.12e} m")
  print(f"      Velocity :  {tle_initial_state[3]:>19.12e}  {tle_initial_state[4]:>19.12e}  {tle_initial_state[5]:>19.12e} m/s")
  print(f"    Classical Orbital Elements")
  print(f"      SMA  :  {coe.sma:19.12e} m")
  print(f"      ECC  :  {coe.ecc:19.12e} -")
  print(f"      INC  :  {coe.inc *CONVERTER.DEG_PER_RAD:19.12e} deg")
  print(f"      RAAN :  {coe.raan*CONVERTER.DEG_PER_RAD:19.12e} deg")
  print(f"      AOP  :  {coe.aop *CONVERTER.DEG_PER_RAD:19.12e} deg")
  if coe.ta is not None:
    print(f"      TA   :  {coe.ta  *CONVERTER.DEG_PER_RAD:19.12e} deg")
  if coe.ea is not None:
    print(f"      EA   :  {coe.ea  *CONVERTER.DEG_PER_RAD:19.12e} deg")
  if coe.ma is not None:
    print(f"      MA   :  {coe.ma  *CONVERTER.DEG_PER_RAD:19.12e} deg")
  if coe.ha is not None:
    print(f"      HA   :  {coe.ha  *CONVERTER.DEG_PER_RAD:19.12e} deg")
  if coe.pa is not None:
    print(f"      PA   :  {coe.pa:19.12e} -")

  return tle_initial_state, time_o_dt


def get_initial_state_from_maneuver_plan(
  decision_state : 'DecisionState',
) -> tuple[np.ndarray, datetime]:
  """
  Get initial state vector and epoch from a DecisionState (maneuver plan).

  Input:
  ------
    decision_state : DecisionState
      Decision state loaded from a maneuver plan YAML file.

  Output:
  -------
    initial_state : np.ndarray
      Initial state vector [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z] in m and m/s.
    initial_epoch : datetime
      Epoch of the initial state.
  """
  from src.schemas.optimization import DecisionState

  title = "Initial State"
  print("\n" + "-" * len(title))
  print(title)
  print("-" * len(title))
  print()

  print("  Progress")
  print("    Extract initial state from maneuver plan")
  print("    Convert to classical orbital elements for display")
  print()

  state_vec = np.concatenate((decision_state.position, decision_state.velocity))
  epoch_dt  = decision_state.epoch

  # Get ET for display
  try:
    epoch_et = utc_to_et(epoch_dt)
    et_str   = f" / {epoch_et:.6f} ET"
  except Exception:
    et_str = ""

  # Convert to COE for display
  coe = OrbitConverter.pv_to_coe(
    pos_vec = decision_state.position,
    vel_vec = decision_state.velocity,
    gp      = SOLARSYSTEMCONSTANTS.EARTH.GP,
  )

  n_maneuvers = len(decision_state.maneuvers) if decision_state.maneuvers else 0
  n_vars      = decision_state.n_variables

  print("  Summary")
  print(f"    Source     : Maneuver Plan")
  print(f"    Epoch      : {epoch_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC{et_str}")
  print(f"    Frame      : J2000")
  print(f"    Maneuvers  : {n_maneuvers}")
  print(f"    Variables  : {n_vars}")
  print(f"    Cartesian State")
  print(f"      Position :  {state_vec[0]:>19.12e}  {state_vec[1]:>19.12e}  {state_vec[2]:>19.12e} m")
  print(f"      Velocity :  {state_vec[3]:>19.12e}  {state_vec[4]:>19.12e}  {state_vec[5]:>19.12e} m/s")
  print(f"    Classical Orbital Elements")
  print(f"      SMA  :  {coe.sma:19.12e} m")
  print(f"      ECC  :  {coe.ecc:19.12e} -")
  print(f"      INC  :  {coe.inc  * CONVERTER.DEG_PER_RAD:19.12e} deg")
  print(f"      RAAN :  {coe.raan * CONVERTER.DEG_PER_RAD:19.12e} deg")
  print(f"      AOP  :  {coe.aop  * CONVERTER.DEG_PER_RAD:19.12e} deg")
  if coe.ta is not None:
    print(f"      TA   :  {coe.ta   * CONVERTER.DEG_PER_RAD:19.12e} deg")
  if coe.ea is not None:
    print(f"      EA   :  {coe.ea   * CONVERTER.DEG_PER_RAD:19.12e} deg")
  if coe.ma is not None:
    print(f"      MA   :  {coe.ma   * CONVERTER.DEG_PER_RAD:19.12e} deg")
  if coe.ha is not None:
    print(f"      HA   :  {coe.ha   * CONVERTER.DEG_PER_RAD:19.12e} deg")
  if coe.pa is not None:
    print(f"      PA   :  {coe.pa:19.12e} -")

  if n_maneuvers > 0:
    print(f"    Maneuver Details")
    for i, m in enumerate(decision_state.maneuvers):
      var_t  = decision_state.variable_maneuver_time[i] if i < len(decision_state.variable_maneuver_time) else False
      var_dv = decision_state.variable_maneuver_delta_v[i] if i < len(decision_state.variable_maneuver_delta_v) else np.array([False, False, False])
      dv_mag = np.linalg.norm(m.delta_vel_vec)
      print(f"      Burn {i+1}: {m.time_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC  |ΔV| = {dv_mag:.3f} m/s  frame = {m.frame}  var_t = {var_t}  var_dv = {var_dv.tolist()}")

  return state_vec, epoch_dt
