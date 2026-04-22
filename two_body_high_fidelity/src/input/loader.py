import warnings
import yaml
import sys
import subprocess
import spiceypy as spice
import numpy    as np
import pandas   as pd
from pathlib  import Path
from datetime import datetime, timedelta
from typing   import Optional
from src.model.time_converter  import utc_to_et
from src.model.orbit_converter import OrbitConverter
from src.model.constants       import PRINTFORMATTER, SOLARSYSTEMCONSTANTS
from src.input.cli             import parse_time
from src.utility.tle_helper    import get_tle_satellite_and_tle_epoch
from src.schemas.time          import TimeStructure
from src.schemas.propagation   import PropagationResult
from src.schemas.state         import ClassicalOrbitalElements, ModifiedEquinoctialElements, TLEData, TrackerStation
from src.model.constants       import CONVERTER

def load_supported_objects() -> dict:
  """
  Load supported objects from YAML configuration file.
  
  Input:
  ------
    None
  
  Output:
  -------
    supported_objects : dict
      Dictionary of supported objects loaded from the YAML file.
  """
  # Adjust path traversal: input -> src -> project_root
  project_root = Path(__file__).parent.parent.parent
  config_path  = project_root / 'data' / 'supported_objects.yaml'
  
  if not config_path.exists():
    raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
  with open(config_path, 'r') as f:
    return yaml.safe_load(f)


def _parse_single_tracker(data: dict) -> 'TrackerStation':
  """
  Parse a single tracker from dictionary data.

  Input:
  ------
    data : dict
      Dictionary containing tracker data from YAML.

  Output:
  -------
    tracker : TrackerStation
      Parsed tracker station.

  Raises:
  -------
    ValueError
      If required fields are missing.
  """
  # Import nested dataclasses
  from src.schemas.state import TrackerStation, TrackerPosition, TrackerPerformance, TrackerConstraints, TrackerUncertainty, AzimuthLimits, ElevationLimits, RangeLimits

  # Validate required fields
  if 'name' not in data:
    raise ValueError(f"Tracker data missing required field: name")
  if 'position' not in data:
    raise ValueError(f"Tracker data missing required section: position")

  position = data['position']
  if 'latitude__deg' not in position or 'longitude__deg' not in position or 'altitude__m' not in position:
    raise ValueError(f"Tracker 'position' section missing required fields")

  # Convert degrees to radians
  latitude__rad = position['latitude__deg'] * CONVERTER.RAD_PER_DEG
  longitude__rad = position['longitude__deg'] * CONVERTER.RAD_PER_DEG

  # Create position object
  tracker_position = TrackerPosition(
    latitude  = latitude__rad,
    longitude = longitude__rad,
    altitude  = position['altitude__m'],
  )

  # Parse performance section (optional)
  tracker_performance = None
  if 'performance' in data:
    perf = data['performance']

    # Parse constraints subsection
    tracker_constraints = None
    if 'constraints' in perf:
      constraints = perf['constraints']

      # Default values for performance limits
      default_azimuth_min__deg   = -180.0
      default_azimuth_max__deg   = 180.0
      default_elevation_min__deg = 0.0
      default_elevation_max__deg = 90.0
      default_range_min__m       = 0.0
      default_range_max__m       = 5.0e6

      azimuth_limits   = None
      elevation_limits = None
      range_limits     = None

      # Parse azimuth limits
      if 'azimuth_min_max__deg' in constraints:
        az_limits = constraints['azimuth_min_max__deg']
        # Handle both string "min, max" and list/tuple [min, max]
        if isinstance(az_limits, str):
          az_values = [float(x.strip()) for x in az_limits.split(',')]
          if len(az_values) == 2:
            azimuth_limits = AzimuthLimits(
              min = az_values[0] * CONVERTER.RAD_PER_DEG,
              max = az_values[1] * CONVERTER.RAD_PER_DEG,
            )
        elif isinstance(az_limits, (list, tuple)) and len(az_limits) == 2:
          azimuth_limits = AzimuthLimits(
            min = az_limits[0] * CONVERTER.RAD_PER_DEG,
            max = az_limits[1] * CONVERTER.RAD_PER_DEG,
          )

      # Parse elevation limits
      if 'elevation_min_max__deg' in constraints:
        el_limits = constraints['elevation_min_max__deg']
        # Handle both string "min, max" and list/tuple [min, max]
        if isinstance(el_limits, str):
          el_values = [float(x.strip()) for x in el_limits.split(',')]
          if len(el_values) == 2:
            elevation_limits = ElevationLimits(
              min = el_values[0] * CONVERTER.RAD_PER_DEG,
              max = el_values[1] * CONVERTER.RAD_PER_DEG,
            )
        elif isinstance(el_limits, (list, tuple)) and len(el_limits) == 2:
          elevation_limits = ElevationLimits(
            min = el_limits[0] * CONVERTER.RAD_PER_DEG,
            max = el_limits[1] * CONVERTER.RAD_PER_DEG,
          )

      # Parse range limits
      if 'range_min_max__m' in constraints:
        rg_limits = constraints['range_min_max__m']
        # Handle both string "min, max" and list/tuple [min, max]
        if isinstance(rg_limits, str):
          rg_values = [float(x.strip()) for x in rg_limits.split(',')]
          if len(rg_values) == 2:
            range_limits = RangeLimits(
              min = rg_values[0],
              max = rg_values[1],
            )
        elif isinstance(rg_limits, (list, tuple)) and len(rg_limits) == 2:
          range_limits = RangeLimits(
            min = rg_limits[0],
            max = rg_limits[1],
          )

      # Apply defaults for any unspecified limits
      if azimuth_limits is None:
        azimuth_limits = AzimuthLimits(
          min = default_azimuth_min__deg * CONVERTER.RAD_PER_DEG,
          max = default_azimuth_max__deg * CONVERTER.RAD_PER_DEG,
        )

      if elevation_limits is None:
        elevation_limits = ElevationLimits(
          min = default_elevation_min__deg * CONVERTER.RAD_PER_DEG,
          max = default_elevation_max__deg * CONVERTER.RAD_PER_DEG,
        )

      if range_limits is None:
        range_limits = RangeLimits(
          min = default_range_min__m,
          max = default_range_max__m,
        )

      # Create constraints object
      tracker_constraints = TrackerConstraints(
        azimuth   = azimuth_limits,
        elevation = elevation_limits,
        range     = range_limits,
      )

    # Parse uncertainty subsection
    tracker_uncertainty = None
    if 'uncertainty' in perf:
      uncertainty = perf['uncertainty']

      # Extract uncertainty values (defaults to 0.0 if not specified)
      # Convert to float in case YAML parsed scientific notation as string
      range_unc                    = float(uncertainty.get('range__m', 0.0))
      range_rate_unc               = float(uncertainty.get('range_rate__m_per_s', 0.0))
      azimuth_unc__deg              = float(uncertainty.get('azimuth__deg', 0.0))
      azimuth_rate_unc__deg_per_s   = float(uncertainty.get('azimuth_rate__deg_per_s', 0.0))
      elevation_unc__deg            = float(uncertainty.get('elevation__deg', 0.0))
      elevation_rate_unc__deg_per_s = float(uncertainty.get('elevation_rate__deg_per_s', 0.0))

      # Convert angular uncertainties from degrees to radians
      tracker_uncertainty = TrackerUncertainty(
        range          = range_unc,
        range_rate     = range_rate_unc,
        azimuth        = azimuth_unc__deg * CONVERTER.RAD_PER_DEG,
        azimuth_rate   = azimuth_rate_unc__deg_per_s * CONVERTER.RAD_PER_DEG,
        elevation      = elevation_unc__deg * CONVERTER.RAD_PER_DEG,
        elevation_rate = elevation_rate_unc__deg_per_s * CONVERTER.RAD_PER_DEG,
      )

    # Create performance object
    tracker_performance = TrackerPerformance(
      constraints = tracker_constraints,
      uncertainty = tracker_uncertainty,
    )

  return TrackerStation(
    name        = data['name'],
    position    = tracker_position,
    performance = tracker_performance,
  )


def load_tracker_station(
  tracker_filepath : Path,
) -> list['TrackerStation']:
  """
  Load tracking station data from YAML configuration file.

  Supports both single tracker (dict) and multiple trackers (list of dicts).

  Input:
  ------
    tracker_filepath : Path
      Path to the tracker YAML file.

  Output:
  -------
    trackers : list[TrackerStation]
      List of tracker station dataclasses with name, position, and performance limits.

  Raises:
  -------
    FileNotFoundError
      If the tracker file is not found.
    ValueError
      If required fields are missing from the YAML file.
  """
  if not tracker_filepath.exists():
    raise FileNotFoundError(f"Tracker file not found: {tracker_filepath}")

  with open(tracker_filepath, 'r') as f:
    data = yaml.safe_load(f)

  # Handle both list of trackers and single tracker
  if isinstance(data, list):
    # Multiple trackers
    trackers = []
    for i, tracker_data in enumerate(data):
      try:
        tracker = _parse_single_tracker(tracker_data)
        trackers.append(tracker)
      except ValueError as e:
        raise ValueError(f"Error parsing tracker {i}: {e}")
    return trackers
  else:
    # Single tracker - wrap in list
    tracker = _parse_single_tracker(data)
    return [tracker]


def load_maneuvers(
  maneuver_filepath : Path,
) -> list:  # list[ImpulsiveManeuver]
  """
  Load impulsive maneuver data from YAML configuration file.

  Input:
  ------
    maneuver_filepath : Path
      Path to the maneuver YAML file.

  Output:
  -------
    maneuvers : list[ImpulsiveManeuver]
      List of impulsive maneuver dataclasses.

  Raises:
  -------
    FileNotFoundError
      If the maneuver file is not found.
    ValueError
      If required fields are missing or invalid.

  YAML Format:
  ------------
    maneuvers:
      - time_iso_utc: "2025-10-01T06:00:00"
        delta_vel__m_per_s: [0.0, 50.0, 0.0]
        frame: "RIC"
      - time_iso_utc: "2025-10-01T12:00:00"
        delta_vel__m_per_s: [0.0, 30.0, 0.0]
        frame: "RIC"
  """
  from src.schemas.spacecraft import ImpulsiveManeuver

  if not maneuver_filepath.exists():
    raise FileNotFoundError(f"Maneuver file not found: {maneuver_filepath}")

  with open(maneuver_filepath, 'r') as f:
    data = yaml.safe_load(f)

  # Expect a 'maneuvers' key with list of maneuvers
  if 'maneuvers' not in data:
    raise ValueError("Maneuver file must contain 'maneuvers' key")

  maneuver_list_data = data['maneuvers']

  if not isinstance(maneuver_list_data, list):
    raise ValueError("'maneuvers' must be a list")

  maneuvers = []
  for i, maneuver_data in enumerate(maneuver_list_data):
    try:
      # Parse required fields
      if 'time_iso_utc' not in maneuver_data:
        raise ValueError("Missing required field 'time_iso_utc'")
      if 'delta_vel__m_per_s' not in maneuver_data:
        raise ValueError("Missing required field 'delta_vel__m_per_s'")

      # Parse time
      time_dt = parse_time(maneuver_data['time_iso_utc'])

      # Parse Delta-V
      delta_vel_vec = np.array(maneuver_data['delta_vel__m_per_s'], dtype=float)
      if len(delta_vel_vec) != 3:
        raise ValueError(f"delta_vel__m_per_s must have 3 components, got {len(delta_vel_vec)}")

      # Parse frame (optional, default J2000)
      frame = maneuver_data.get('frame', 'J2000')

      # Create ImpulsiveManeuver object
      maneuver = ImpulsiveManeuver(
        time_dt       = time_dt,
        delta_vel_vec = delta_vel_vec,
        frame         = frame,
      )
      maneuvers.append(maneuver)

    except Exception as e:
      raise ValueError(f"Error parsing maneuver {i}: {e}")

  # Check time ordering — warn and sort if out of order
  for i in range(1, len(maneuvers)):
    if maneuvers[i].time_dt <= maneuvers[i - 1].time_dt:
      warnings.warn(
        f"Maneuvers in {maneuver_filepath.name} are not in chronological order. Sorting by time.",
        UserWarning,
      )
      maneuvers.sort(key=lambda m: m.time_dt)
      break

  return maneuvers


def load_maneuver_plan(
  maneuver_plan_filepath : Path,
) -> 'DecisionState':
  """
  Load a maneuver plan YAML file into a DecisionState.

  A maneuver plan contains an initial state (position, velocity, epoch)
  and a sequence of impulsive maneuvers, each with optional variable flags
  for trajectory optimization.

  Input:
  ------
    maneuver_plan_filepath : Path
      Path to the maneuver plan YAML file.

  Output:
  -------
    decision_state : DecisionState
      Decision state with initial state, maneuvers, and variable flags.

  Raises:
  -------
    FileNotFoundError
      If the maneuver plan file is not found.
    ValueError
      If required fields are missing or invalid.

  YAML Format:
  ------------
    initial_state:
      epoch_iso_utc: "2025-10-01T00:00:00"
      pos_vec__m:        [x, y, z]
      vel_vec__m_per_s:  [vx, vy, vz]
      frame: "J2000"
      variable_epoch:    false
      variable_position: [false, false, false]
      variable_velocity: [false, false, false]

    maneuvers:
      - time_iso_utc: "2025-10-01T06:00:00"
        delta_vel__m_per_s: [0.0, 50.0, 0.0]
        frame: "RIC"
        variable_time: true
        variable_delta_v: [false, true, false]
  """
  from src.schemas.spacecraft    import ImpulsiveManeuver, ManeuversConfig
  from src.schemas.optimization  import DecisionState

  if not maneuver_plan_filepath.exists():
    raise FileNotFoundError(f"Maneuver plan file not found: {maneuver_plan_filepath}")

  with open(maneuver_plan_filepath, 'r') as f:
    data = yaml.safe_load(f)

  if 'initial_state' not in data:
    raise ValueError("Maneuver plan file must contain 'initial_state' key")

  init = data['initial_state']

  # Parse initial state
  # Support both 'epoch' and 'epoch_iso_utc' field names
  epoch_raw = init.get('epoch', init.get('epoch_iso_utc'))
  if epoch_raw is None:
    raise ValueError("initial_state must contain 'epoch' or 'epoch_iso_utc'")
  if 'pos_vec__m' not in init:
    raise ValueError("initial_state must contain 'pos_vec__m'")
  if 'vel_vec__m_per_s' not in init:
    raise ValueError("initial_state must contain 'vel_vec__m_per_s'")

  epoch    = epoch_raw if isinstance(epoch_raw, datetime) else parse_time(epoch_raw)
  if epoch.tzinfo is not None:
    epoch = epoch.replace(tzinfo=None)

  def _parse_vec(raw):
    if isinstance(raw, str):
      return np.array([float(x.strip()) for x in raw.replace('[', '').replace(']', '').split(',')])
    return np.array(raw, dtype=float)

  def _parse_bool_vec(raw):
    if isinstance(raw, str):
      return np.array([x.strip().lower() == 'true' for x in raw.replace('[', '').replace(']', '').split(',')], dtype=bool)
    return np.array(raw, dtype=bool)

  position = _parse_vec(init['pos_vec__m'])
  velocity = _parse_vec(init['vel_vec__m_per_s'])

  if len(position) != 3:
    raise ValueError(f"pos_vec__m must have 3 components, got {len(position)}")
  if len(velocity) != 3:
    raise ValueError(f"vel_vec__m_per_s must have 3 components, got {len(velocity)}")

  # Parse initial state variable flags (default: all fixed)
  variable_epoch    = bool(init.get('variable_epoch', False))
  variable_position = _parse_bool_vec(init.get('variable_position', [False, False, False]))
  variable_velocity = _parse_bool_vec(init.get('variable_velocity', [False, False, False]))

  # Parse maneuvers (optional)
  maneuvers_config          = ManeuversConfig()
  variable_maneuver_time    = []
  variable_maneuver_delta_v = []

  if 'maneuvers' in data and data['maneuvers'] is not None:
    for i, m_data in enumerate(data['maneuvers']):
      if 'time_iso_utc' not in m_data:
        raise ValueError(f"Maneuver {i} missing 'time_iso_utc'")
      # Support both old and new field names
      if 'delta_vel_vec__m_per_s' in m_data:
        delta_vel_raw = m_data['delta_vel_vec__m_per_s']
      elif 'delta_vel__m_per_s' in m_data:
        delta_vel_raw = m_data['delta_vel__m_per_s']
      else:
        raise ValueError(f"Maneuver {i} missing 'delta_vel_vec__m_per_s'")

      time_raw      = m_data['time_iso_utc']
      time_dt       = time_raw if isinstance(time_raw, datetime) else parse_time(time_raw)
      if time_dt.tzinfo is not None:
        time_dt = time_dt.replace(tzinfo=None)
      delta_vel_vec = _parse_vec(delta_vel_raw)
      frame         = m_data.get('frame', 'J2000')
      name          = m_data.get('name', None)
      description   = m_data.get('description', None)

      if len(delta_vel_vec) != 3:
        raise ValueError(f"Maneuver {i}: delta_vel__m_per_s must have 3 components, got {len(delta_vel_vec)}")

      maneuvers_config.append(ImpulsiveManeuver(
        time_dt       = time_dt,
        delta_vel_vec = delta_vel_vec,
        frame         = frame,
        name          = name,
        description   = description,
      ))

      # Variable flags for this maneuver
      # Defaults: time is fixed, delta-v components are all variable
      variable_maneuver_time.append(bool(m_data.get('variable_time', False)))
      variable_maneuver_delta_v.append(
        _parse_bool_vec(m_data.get('variable_delta_vel_vec', m_data.get('variable_delta_v', [True, True, True])))
      )

  return DecisionState(
    position                = position,
    velocity                = velocity,
    epoch                   = epoch,
    maneuvers               = maneuvers_config,
    variable_position       = variable_position,
    variable_velocity       = variable_velocity,
    variable_epoch          = variable_epoch,
    variable_maneuver_time  = variable_maneuver_time,
    variable_maneuver_delta_v = variable_maneuver_delta_v,
  )


def save_maneuver_plan(
  decision_state  : 'DecisionState',
  output_filepath : Path,
) -> None:
  """
  Write a solved maneuver plan to YAML.

  Uses the same format as the input maneuver plan so it can be loaded
  with --initial-maneuver-plan or --resume-from.

  Input:
  ------
    decision_state : DecisionState
      Decision state to write.
    output_filepath : Path
      Path to write the YAML file.
  """
  from src.schemas.optimization import DecisionState

  data = {
    'initial_state': {
      'epoch_iso_utc':     decision_state.epoch.strftime('%Y-%m-%dT%H:%M:%S.%f') if decision_state.epoch else None,
      'pos_vec__m':        decision_state.position.tolist(),
      'vel_vec__m_per_s':  decision_state.velocity.tolist(),
      'frame':             'J2000',
      'variable_epoch':    decision_state.variable_epoch,
      'variable_position': decision_state.variable_position.tolist(),
      'variable_velocity': decision_state.variable_velocity.tolist(),
    }
  }

  if decision_state.maneuvers and len(decision_state.maneuvers) > 0:
    maneuvers_list = []
    for i, m in enumerate(decision_state.maneuvers):
      m_dict = {
        'time_iso_utc':       m.time_dt.strftime('%Y-%m-%dT%H:%M:%S.%f'),
        'delta_vel_vec__m_per_s': m.delta_vel_vec.tolist(),
        'frame':              m.frame,
        'name':               m.name,
        'description':        m.description,
      }
      if i < len(decision_state.variable_maneuver_time):
        m_dict['variable_time'] = decision_state.variable_maneuver_time[i]
      if i < len(decision_state.variable_maneuver_delta_v):
        m_dict['variable_delta_vel_vec'] = decision_state.variable_maneuver_delta_v[i].tolist()
      maneuvers_list.append(m_dict)
    data['maneuvers'] = maneuvers_list

  output_filepath.parent.mkdir(parents=True, exist_ok=True)
  with open(output_filepath, 'w') as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_gravity_field_model(
  gravity_model_folderpath : Path,
  gravity_model_filename   : str,
  gravity_model_degree     : int,
  gravity_model_order      : int,
):
  """
  Load gravity field model from coefficient file.
  
  Input:
  ------
    gravity_model_folderpath : Path
      Path to the folder containing gravity coefficient files.
    gravity_model_filename : str
      Filename of the gravity coefficient file (e.g., 'EGM2008.gfc').
    gravity_model_degree : int
      Maximum degree for spherical harmonics expansion.
    gravity_model_order : int
      Maximum order for spherical harmonics expansion.
      
  Output:
  -------
    result : dict | None
      Dictionary containing:
      - model : SphericalHarmonicsGravity - Loaded gravity model object
      - gp : float - Gravitational parameter from model
      - radius : float - Reference radius from model
      Returns None if loading failed.
  """
  # Import
  from src.model.gravity_field import load_gravity_field
  
  # Display gravity field info
  print("    Gravity Field Model")

  # Build full path to gravity file
  gravity_model_filepath = gravity_model_folderpath / gravity_model_filename

  # Format filepath relative to project root
  try:
    relative_path = gravity_model_filepath.relative_to(Path.cwd())
    formatted_path = f"<project_folderpath>/{relative_path}"
  except ValueError:
    formatted_path = str(gravity_model_filepath)

  if not gravity_model_filepath.exists():
    print(f"      Filepath   : {formatted_path} (NOT FOUND)")
    print(f"      Status     : Failed - file not found")
    return None

  print(f"      Filepath   : {formatted_path}")
  print(f"      Degree     : {gravity_model_degree}")
  print(f"      Order      : {gravity_model_order}")

  try:
    spherical_harmonics_model = load_gravity_field(
      filepath = gravity_model_filepath,
      degree   = gravity_model_degree,
      order    = gravity_model_order,
    )

    # Extract gp and radius from the loaded model
    gp     = spherical_harmonics_model.gp
    radius = spherical_harmonics_model.radius

    print(f"      GP         : {gp:{PRINTFORMATTER.SCIENTIFIC_NOTATION}} m³/s²")
    print(f"      Radius     : {radius:{PRINTFORMATTER.SCIENTIFIC_NOTATION}} m")

    return spherical_harmonics_model
  except Exception as e:
    print(f"\n      [ERROR] Failed to load gravity field model: {e}")
    raise


def load_gravity_model_from_coefficients(
  coefficient_names        : list,
  gravity_model_folderpath : Optional[Path] = None,
  gravity_model_filename   : Optional[str]  = None,
):
  """
  Load gravity model using only specific named coefficients.
  
  Input:
  ------
    coefficient_names : list
      List of coefficient names (e.g., ['J2', 'J3', 'C22', 'S22']).
    gravity_model_folderpath : Path, optional
      Path to folder containing gravity model files.
    gravity_model_filename : str
      Filename of gravity model to use for coefficient values.
      
  Output:
  -------
    model : SphericalHarmonicsGravity | None
      Gravity model with only the specified coefficients, or None if failed.
  """
  from src.model.gravity_field import create_gravity_model_from_coefficients

  print("    Gravity Field Model (Explicit Coefficients)")
  print(f"      Coefficients : {', '.join(coefficient_names)}")

  # Build gravity file path if folder is provided
  gravity_file_path = None
  if gravity_model_folderpath is not None and gravity_model_filename is not None:
    gravity_file_path = gravity_model_folderpath / gravity_model_filename
    if gravity_file_path.exists():
      print(f"      Source File  : {gravity_model_filename}")
    else:
      print(f"      Source File  : {gravity_model_filename} (NOT FOUND - using defaults)")
      gravity_file_path = None
  else:
    print(f"      Source File  : None (using hardcoded defaults)")

  try:
    model = create_gravity_model_from_coefficients(
      coefficient_names = coefficient_names,
      gravity_file_path = gravity_file_path,
    )
    print(f"      GP           : {model.gp:{PRINTFORMATTER.SCIENTIFIC_NOTATION}} m³/s²")
    print(f"      Radius       : {model.radius:{PRINTFORMATTER.SCIENTIFIC_NOTATION}} m")
    print(f"      Max Degree   : {model.degree}")
    print(f"      Max Order    : {model.order}")
    return model
  except Exception as e:
    print(f"      Status       : Failed - {e}")
    return None


def load_files(
  spice_kernels_folderpath  : Path,
  lsk_filepath              : Path,
  gravity_model_folderpath  : Optional[Path] = None,
  gravity_model_filename    : Optional[str]  = None,
  gravity_model_degree      : Optional[int]  = None,
  gravity_model_order       : Optional[int]  = None,
  gravity_coefficient_names : Optional[list] = None,
  tracker_filepath          : Optional[Path] = None,
  maneuver_filename         : Optional[str]  = None,
):
  """
  Load necessary files for the simulation, including SPICE kernels, gravity model, and tracker.

  Input:
  ------
    spice_kernels_folderpath : Path
      Path to the SPICE kernels folder.
    lsk_filepath : Path
      Path to the leap seconds kernel file.
    gravity_model_folderpath : Path, optional
      Path to gravity models folder.
    gravity_model_filename : str, optional
      Gravity model filename.
    gravity_model_degree : int, optional
      Maximum degree.
    gravity_model_order : int, optional
      Maximum order.
    gravity_coefficient_names : list, optional
      List of explicit coefficient names (e.g., ['J2', 'J3', 'C22']).
      If provided, creates a model with only these coefficients.
    tracker_filepath : Path, optional
      Path to tracker station YAML file.
    maneuver_filename : str, optional
      Filename of maneuver YAML file (in input/initial_states_and_maneuvers/ folder).

  Output:
  -------
    spherical_harmonics_model : SphericalHarmonicsGravity | None
      Loaded gravity model or None if not requested/failed.
    trackers : list[TrackerStation] | None
      List of loaded tracker stations or None if not requested/failed.
    maneuvers : list[ImpulsiveManeuver] | None
      List of loaded maneuvers or None if not requested/failed.

  Raises:
  -------
    ValueError
      If gravity model was requested but failed to load.
  """
  title = "Load Files"
  print("\n" + "-" * len(title))
  print(title)
  print("-" * len(title))
  print()

  # Progress subsection
  print("  Progress")
  print("    Load SPICE kernel files")
  print("    Load gravity field model")
  if tracker_filepath is not None:
    print("    Load tracker station configuration")
    print("    Normalize tracker azimuth constraints")
  if maneuver_filename is not None:
    print("    Load maneuver configuration")
  print()

  # Summary subsection
  print("  Summary")
  print(f"    Project Folderpath : {Path.cwd()}")

  # Load SPICE files
  load_spice_files(spice_kernels_folderpath, lsk_filepath)

  # Load gravity model
  spherical_harmonics_model = None

  # Option 1: Explicit coefficient names (e.g., ['J2', 'J3', 'C22', 'S22'])
  if gravity_coefficient_names is not None and len(gravity_coefficient_names) > 0:
    spherical_harmonics_model = load_gravity_model_from_coefficients(
      coefficient_names        = gravity_coefficient_names,
      gravity_model_folderpath = gravity_model_folderpath,
      gravity_model_filename   = gravity_model_filename,
    )
    if spherical_harmonics_model is None:
      raise ValueError(
        "--gravity-harmonics-coefficients was specified but model creation failed."
      )

  # Option 2: Full spherical harmonics file with degree/order
  elif (gravity_model_filename is not None and
        gravity_model_degree is not None and
        gravity_model_degree > 0 and
        gravity_model_order is not None and
        gravity_model_folderpath is not None):
    spherical_harmonics_model = load_gravity_field_model(
      gravity_model_folderpath = gravity_model_folderpath,
      gravity_model_filename   = gravity_model_filename,
      gravity_model_degree     = gravity_model_degree,
      gravity_model_order      = gravity_model_order,
    )
    if spherical_harmonics_model is None:
      raise ValueError(
        "--gravity-harmonics-degree-order was specified but gravity model failed to load. "
        "Please check the gravity model file exists and is valid."
      )

  # Load tracker stations
  trackers = None
  if tracker_filepath is not None:
    trackers = load_tracker_station(tracker_filepath)

    # Format filepath relative to project root
    try:
      relative_path = tracker_filepath.relative_to(Path.cwd())
      formatted_path = f"<project_folderpath>/{relative_path}"
    except ValueError:
      formatted_path = str(tracker_filepath)
    print(f"    Tracker Stations")
    print(f"      Filepath   : {formatted_path}")
    for i, tracker in enumerate(trackers, 1):
      print(f"      Tracker {i}  :")
      print(f"        Name     : {tracker.name}")
      print(f"        Position :")
      print(f"          Lat    : {tracker.position.latitude * CONVERTER.DEG_PER_RAD:.1f}°")
      print(f"          Lon    : {tracker.position.longitude * CONVERTER.DEG_PER_RAD:.1f}°")
      print(f"          Alt    : {tracker.position.altitude:.1f} m")

    # Validate and normalize values from loaded files
    trackers = [validate_tracker_input(t) for t in trackers]
    trackers = [normalize_tracker_azimuth(t) for t in trackers]

  # Load maneuvers
  maneuvers = []
  if maneuver_filename is not None:
    from pathlib import Path as PathLib
    project_root = PathLib(__file__).parent.parent.parent
    maneuver_filepath = project_root / 'input' / 'initial_states_and_maneuvers' / maneuver_filename

    if maneuver_filepath.exists():
      maneuvers = load_maneuvers(maneuver_filepath)

      # Format filepath relative to project root
      try:
        relative_path = maneuver_filepath.relative_to(PathLib.cwd())
        formatted_path = f"<project_folderpath>/{relative_path}"
      except ValueError:
        formatted_path = str(maneuver_filepath)

      print(f"    Maneuvers")
      print(f"      Filepath : {formatted_path}")
      print(f"      Count    : {len(maneuvers)}")
      for i, mnvr in enumerate(maneuvers, 1):
        print(f"      Maneuver {i}")
        print(f"        Time   : {mnvr.time_dt.isoformat()}")
        print(f"        ΔV     : {mnvr.mag():.3f} m/s")
        print(f"        Frame  : {mnvr.frame}")
    else:
      print(f"    Maneuvers")
      print(f"      Status   : File not found - {maneuver_filename}")

  # Return loaded model, trackers, and maneuvers (or None if not requested)
  return spherical_harmonics_model, trackers, maneuvers


def validate_tracker_input(tracker):
  """
  Validate tracker input data.

  Checks that tracker performance limits are valid:
  - Azimuth: min < max (clockwise convention)
  - Elevation: min < max
  - Range: min < max

  Input:
  ------
    tracker : TrackerStation | None
      Tracker station object to validate. If None, returns None.

  Output:
  -------
    tracker : TrackerStation | None
      Same tracker object (unmodified) if valid.

  Raises:
  -------
    ValueError
      If any performance limits are invalid.
  """
  if tracker is None:
    return None

  if tracker.performance and tracker.performance.constraints:
    # Validate azimuth limits
    if tracker.performance.constraints.azimuth:
      az_min__deg = tracker.performance.constraints.azimuth.min * CONVERTER.DEG_PER_RAD
      az_max__deg = tracker.performance.constraints.azimuth.max * CONVERTER.DEG_PER_RAD

      if az_min__deg > az_max__deg:
        raise ValueError(
          f"Invalid azimuth range for tracker '{tracker.name}': "
          f"min ({az_min__deg:.1f}°) > max ({az_max__deg:.1f}°). "
          f"Azimuth must be specified with min < max."
        )

    # Validate elevation limits
    if tracker.performance.constraints.elevation:
      el_min__deg = tracker.performance.constraints.elevation.min * CONVERTER.DEG_PER_RAD
      el_max__deg = tracker.performance.constraints.elevation.max * CONVERTER.DEG_PER_RAD

      if el_min__deg > el_max__deg:
        raise ValueError(
          f"Invalid elevation range for tracker '{tracker.name}': "
          f"min ({el_min__deg:.1f}°) > max ({el_max__deg:.1f}°). "
          f"Elevation must be specified with min < max."
        )

    # Validate range limits
    if tracker.performance.constraints.range:
      if tracker.performance.constraints.range.min > tracker.performance.constraints.range.max:
        raise ValueError(
          f"Invalid range limits for tracker '{tracker.name}': "
          f"min ({tracker.performance.constraints.range.min:.1f} m) > max ({tracker.performance.constraints.range.max:.1f} m). "
          f"Range must be specified with min < max."
        )

  return tracker


def normalize_tracker_azimuth(tracker):
  """
  Normalize tracker azimuth constraints to -180° to +180° range.

  Modifies the tracker object in-place by converting azimuth values outside
  the -180° to +180° range to their equivalent values within that range.

  Input:
  ------
    tracker : TrackerStation | None
      Tracker station object to normalize. If None, returns None.

  Output:
  -------
    tracker : TrackerStation | None
      Same tracker object with normalized azimuth values (modified in-place).
  """
  if tracker is None:
    return None

  if tracker.performance and tracker.performance.constraints and tracker.performance.constraints.azimuth:
    from src.schemas.state import AzimuthLimits

    # Convert radians to degrees for normalization
    az_min__deg = tracker.performance.constraints.azimuth.min * CONVERTER.DEG_PER_RAD
    az_max__deg = tracker.performance.constraints.azimuth.max * CONVERTER.DEG_PER_RAD

    # Check if this represents full circle coverage before normalizing
    az_range__deg = az_max__deg - az_min__deg
    is_full_circle = abs(az_range__deg - 360.0) < 1e-6

    if is_full_circle:
      # Full circle: keep as -180 to 180 (canonical full range)
      az_min__deg = -180.0
      az_max__deg = 180.0
    else:
      # Normalize to -180° to +180° range
      # Use (value + 180) % 360 - 180 to map to [-180, 180]
      az_min__deg = ((az_min__deg + 180.0) % 360.0) - 180.0
      az_max__deg = ((az_max__deg + 180.0) % 360.0) - 180.0

    # Update the tracker with normalized values (convert back to radians)
    tracker.performance.constraints.azimuth = AzimuthLimits(
      min = az_min__deg * CONVERTER.RAD_PER_DEG,
      max = az_max__deg * CONVERTER.RAD_PER_DEG,
    )

  return tracker


def unload_files() -> None:
  """
  Unload files that were loaded for the simulation, including SPICE kernels.
  
  Input:
  ------
    None
      
  Output:
  -------
    None
  """
  unload_spice_files()


def load_spice_files(
  spice_kernels_folderpath : Path,
  lsk_filepath             : Path,
) -> None:
  """
  Load required SPICE kernels.
  
  Input:
  ------
    spice_kernels_folderpath : Path
      Path to the SPICE kernels folder.
    lsk_filepath : Path
      Path to the leap seconds kernel file.
      
  Output:
  -------
    None
    
  Raises:
  -------
    FileNotFoundError
      If SPICE kernels folder or required kernel files are not found.
  """
  if not spice_kernels_folderpath.exists():
    raise FileNotFoundError(f"SPICE kernels folder not found: {spice_kernels_folderpath}")
  if not lsk_filepath.exists():
    raise FileNotFoundError(f"SPICE leap seconds kernel not found: {lsk_filepath}")

  try:
    rel_path = spice_kernels_folderpath.relative_to(Path.cwd())
    display_path = f"<project_folderpath>/{rel_path}"
  except ValueError:
    display_path = spice_kernels_folderpath

  print(f"    Spice Kernels")
  print(f"      Folderpath : {display_path}")

  # Load leap seconds kernel first (minimal kernel set for time conversion)
  spice.furnsh(str(lsk_filepath))

  # Load planetary ephemeris (SPK)
  # Sort to ensure deterministic behavior. SPICE uses the last loaded kernel for precedence.
  spk_files = sorted(list(spice_kernels_folderpath.glob('de*.bsp')))
  if spk_files:
    for spk_file in spk_files:
      spice.furnsh(str(spk_file))
      print(f"      Loaded SPK : {spk_file.name}")
  else:
    raise FileNotFoundError(f"No SPK files (de*.bsp) found in {spice_kernels_folderpath}")

  # Load planetary constants (PCK)
  pck_files = sorted(list(spice_kernels_folderpath.glob('pck*.tpc')))
  if pck_files:
    for pck_file in pck_files:
      spice.furnsh(str(pck_file))
      print(f"      Loaded PCK : {pck_file.name}")
  else:
    raise FileNotFoundError(f"No PCK files (pck*.tpc) found in {spice_kernels_folderpath}")

  # Load binary PCK (High-precision Earth orientation)
  # Load AFTER text PCK to ensure precedence
  bpc_files = sorted(list(spice_kernels_folderpath.glob('earth_*.bpc')))
  if bpc_files:
    for bpc_file in bpc_files:
      spice.furnsh(str(bpc_file))
      print(f"      Loaded Binary PCK : {bpc_file.name}")


def unload_spice_files() -> None:
  """
  Unload all SPICE kernels.
  
  Input:
  ------
    None
      
  Output:
  -------
    None
  """
  spice.kclear()


def load_horizons_ephemeris(
  filepath : str,
  time_start_dt : Optional[datetime] = None,
  time_end_dt   : Optional[datetime] = None,
) -> dict:
  """
  Load ephemeris from a custom JPL Horizons .csv file produced by the download script module.
  
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
    result : dict
      Dictionary containing:
      - success : bool
      - message : str
      - time_o : datetime (of the first data point)
      - time_f : datetime (of the last data point)
      - delta_time : np.ndarray (seconds from epoch)
      - state : np.ndarray (6xN state vectors in m and m/s)
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
  norad_id                : Optional[str] = None,
  object_name             : str = "object",
  auto_download           : bool = False,
  step                    : str = "1m",
) -> Optional[PropagationResult]:
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
    norad_id : str | None
      NORAD ID for the object.
    object_name : str
      Object name for filename.
    auto_download : bool
      If True, automatically download ephemeris if no compatible file is found.
      
  Output:
  -------
    result : PropagationResult | None
      PropagationResult object or None if failed.
  """
  # Display JPL Horizons folderpath
  try:
    rel_folderpath = jpl_horizons_folderpath.relative_to(Path.cwd())
    display_path = f"<project_folderpath>/{rel_folderpath}"
  except ValueError:
    display_path = jpl_horizons_folderpath

  print("    JPL Horizons Ephemeris")
  print(f"      Folderpath : {display_path}")
  
  # Search for compatible file
  compatible_file = find_compatible_horizons_file(
    jpl_horizons_folderpath = jpl_horizons_folderpath,
    norad_id                = norad_id,
    time_start_dt           = desired_time_o_dt,
    time_end_dt             = desired_time_f_dt,
  )
  
  if compatible_file is None:
    # Prompt user to download
    if auto_download:
      user_response = 'y'
    else:
      print(f"               :   ... No compatible files found. Download from JPL Horizons? (y/n)", end=" ", flush=True)
      user_response = input().strip().lower()
    
    if user_response == 'y':
      print(f"                 :   ... Downloading {object_name} ({norad_id}) ...", end=" ", flush=True)
      
      try:
        # Construct command to run the standalone ephemeris download module
        cmd = [
          sys.executable, "-m", "src.download.ephems",
          norad_id,
          desired_time_o_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
          desired_time_f_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
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
        print(f"Error")
        print(f"               :   ... stderr: {e.stderr}")
        print(f"               :   ... stdout: {e.stdout}")
      except Exception as e:
        print(f"Error : {e}")
  
  # Check if we have a compatible file after potential download
  if compatible_file is None:
    print(f"      Filepath   : None (no compatible file available)")
    return PropagationResult(success=False, message='No JPL Horizons ephemeris file available')
  # Display the file being loaded
  try:
    rel_path = compatible_file.relative_to(Path.cwd())
    display_path = f"<project_folderpath>/{rel_path}"
  except ValueError:
    display_path = compatible_file
  print(f"      Filepath   : {display_path}")

  # Print timespan info
  print(f"      Timespan")
  print(f"        Desired")
  try:
    start_et = utc_to_et(desired_time_o_dt)
    end_et   = utc_to_et(desired_time_f_dt)
    start_time_str = f"{desired_time_o_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC / {start_et:.6f} ET"
    end_time_str   = f"{desired_time_f_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC / {end_et:.6f} ET"
  except Exception:
    start_time_str = f"{desired_time_o_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC / N/A ET"
    end_time_str   = f"{desired_time_f_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC / N/A ET"
  duration__s = (desired_time_f_dt - desired_time_o_dt).total_seconds()
  print(f"          Initial  : {start_time_str}")
  print(f"          Final    : {end_time_str}")
  print(f"          Duration : {duration__s:.1f} s")
  
  # Load Horizons data
  result_horizons = load_horizons_ephemeris(
    filepath      = str(compatible_file),
    time_start_dt = desired_time_o_dt,
    time_end_dt   = desired_time_f_dt,
  )

  # Process Horizons data
  if result_horizons.get('success'):
    result_horizons = process_horizons_result(result_horizons)

  if result_horizons and result_horizons.get('success'):
    # Construct TimeStructure
    time = TimeStructure(
        initial                = result_horizons['time_o'],
        grid_relative_initial  = result_horizons['delta_time']
    )

    # Construct COE object
    coe_obj = ClassicalOrbitalElements(
        sma  = result_horizons['coe']['sma'],
        ecc  = result_horizons['coe']['ecc'],
        inc  = result_horizons['coe']['inc'],
        raan = result_horizons['coe']['raan'],
        aop  = result_horizons['coe']['aop'],
        ta   = result_horizons['coe']['ta'],
        ea   = result_horizons['coe'].get('ea'),
        ma   = result_horizons['coe'].get('ma')
    )

    # Construct MEE object
    mee_obj = ModifiedEquinoctialElements(
        p = result_horizons['mee']['p'],
        f = result_horizons['mee']['f'],
        g = result_horizons['mee']['g'],
        h = result_horizons['mee']['h'],
        k = result_horizons['mee']['k'],
        L = result_horizons['mee']['L']
    )

    return PropagationResult(
        success   = True,
        message   = "JPL Horizons ephemeris loaded successfully",
        time = time,
        state     = result_horizons['state'],
        coe       = coe_obj,
        mee       = mee_obj,
    )
  else:
    msg = result_horizons.get('message') if result_horizons else "Failed to process Horizons data"
    return PropagationResult(success=False, message=msg)


def process_horizons_result(
  result_horizons : dict,
) -> dict:
  """
  Process Horizons result to add COE time series.
  
  Input:
  ------
    result_horizons : dict
      Raw Horizons result dictionary.
      
  Output:
  -------
    result_horizons : dict
      Processed Horizons result with added 'coe', 'mee', and 'plot_delta_time' fields.
  """
  if result_horizons and result_horizons.get('success'):
    actual_start = result_horizons['time_o']
    actual_end   = actual_start + timedelta(seconds=result_horizons['delta_time'][-1])
    
    try:
      start_et = utc_to_et(actual_start)
      end_et   = utc_to_et(actual_end)
      start_time_str = f"{actual_start.strftime('%Y-%m-%d %H:%M:%S')} UTC / {start_et:.6f} ET"
      end_time_str   = f"{actual_end.strftime('%Y-%m-%d %H:%M:%S')} UTC / {end_et:.6f} ET"
    except Exception:
      start_time_str = f"{actual_start.strftime('%Y-%m-%d %H:%M:%S')} UTC / N/A ET"
      end_time_str   = f"{actual_end.strftime('%Y-%m-%d %H:%M:%S')} UTC / N/A ET"

    duration__s = result_horizons['delta_time'][-1] - result_horizons['delta_time'][0]

    print(f"        Actual")
    print(f"          Initial  : {start_time_str}")
    print(f"          Final    : {end_time_str}")
    print(f"          Duration : {duration__s:.1f} s")
    print(f"          Grid     : {len(result_horizons['delta_time'])} points")

    # Create plot_delta_time for seconds-based, zero-start plotting time
    result_horizons['plot_delta_time'] = result_horizons['delta_time'] - result_horizons['delta_time'][0]
    
    # Compute classical orbital elements for Horizons data
    num_points = result_horizons['state'].shape[1]
    result_horizons['coe'] = {
      'sma'  : np.zeros(num_points),
      'ecc'  : np.zeros(num_points),
      'inc'  : np.zeros(num_points),
      'raan' : np.zeros(num_points),
      'aop'  : np.zeros(num_points),
      'ma'   : np.zeros(num_points),
      'ta'   : np.zeros(num_points),
      'ea'   : np.zeros(num_points),
    }
    
    # Compute modified equinoctial elements for Horizons data
    result_horizons['mee'] = {
      'p' : np.zeros(num_points),
      'f' : np.zeros(num_points),
      'g' : np.zeros(num_points),
      'h' : np.zeros(num_points),
      'k' : np.zeros(num_points),
      'L' : np.zeros(num_points),
    }

    for i in range(num_points):
      pos_vec = result_horizons['state'][0:3, i]
      vel_vec = result_horizons['state'][3:6, i]
      
      coe = OrbitConverter.pv_to_coe(
        pos_vec,
        vel_vec,
        SOLARSYSTEMCONSTANTS.EARTH.GP,
      )
      # Access attributes directly from dataclass
      result_horizons['coe']['sma' ][i] = coe.sma
      result_horizons['coe']['ecc' ][i] = coe.ecc
      result_horizons['coe']['inc' ][i] = coe.inc
      result_horizons['coe']['raan'][i] = coe.raan
      result_horizons['coe']['aop' ][i] = coe.aop
      result_horizons['coe']['ta'  ][i] = coe.ta if coe.ta is not None else 0.0
      result_horizons['coe']['ea'  ][i] = coe.ea if coe.ea is not None else 0.0
      result_horizons['coe']['ma'  ][i] = coe.ma if coe.ma is not None else 0.0
      
      # pv_to_mee returns ModifiedEquinoctialElements dataclass
      mee = OrbitConverter.pv_to_mee(
        pos_vec,
        vel_vec,
        SOLARSYSTEMCONSTANTS.EARTH.GP,
      )
      # Access attributes directly from dataclass
      result_horizons['mee']['p'][i] = mee.p
      result_horizons['mee']['f'][i] = mee.f
      result_horizons['mee']['g'][i] = mee.g
      result_horizons['mee']['h'][i] = mee.h
      result_horizons['mee']['k'][i] = mee.k
      result_horizons['mee']['L'][i] = mee.L

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
    filepath : Path | None
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
      # Format: YYYYMMDDTHHMMSSz (uppercase Z)
      start_str = parts[-3]  # e.g., "20251001T000000Z"
      end_str   = parts[-2]  # e.g., "20251002T000000Z"
      
      # Parse the datetime strings (handle both uppercase and lowercase Z)
      start_str_clean = start_str.rstrip('Zz')
      end_str_clean   = end_str.rstrip('Zz')
      
      file_start_dt = datetime.strptime(start_str_clean, '%Y%m%dT%H%M%S')
      file_end_dt   = datetime.strptime(end_str_clean, '%Y%m%dT%H%M%S')
      
      # Check if this file contains the desired timespan (with tolerance)
      # Ensure datetimes are timezone-naive for comparison (file times are parsed as naive UTC)
      ts_dt_naive = time_start_dt.replace(tzinfo=None) if time_start_dt.tzinfo else time_start_dt
      te_dt_naive = time_end_dt.replace(tzinfo=None) if time_end_dt.tzinfo else time_end_dt
      
      start_ok = file_start_dt <= (ts_dt_naive + time_tolerance)
      end_ok   = file_end_dt   >= (te_dt_naive - time_tolerance)
      
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
    filepath : Path | None
      Path to a compatible file if found, None otherwise.
  """
  # Ensure the folder exists before searching
  if not tles_folderpath.exists():
    return None
  
  # Search for TLE files matching the NORAD ID
  glob_pattern   = f'celestrak_tle_{norad_id}_*.txt'
  matching_files = list(tles_folderpath.glob(glob_pattern))
  
  if not matching_files:
    return None
  
  # Time tolerance for matching (TLEs can be valid for longer periods)
  time_tolerance = timedelta(days=7)
  
  for filepath in matching_files:
    # Parse timespan from filename
    # Format: celestrak_tle_<norad_id>_<name>_<start>_<end>.txt
    filename = filepath.stem  # Remove .txt extension
    parts    = filename.split('_')
    
    # Need at least: celestrak, tle, norad_id, and then name parts, start, end
    # Find the timestamp parts (they have 'T' in them)
    timestamp_parts = [p for p in parts if 'T' in p and len(p) > 10]
    
    if len(timestamp_parts) >= 2:
      try:
        # Start and end times should be the last two timestamp-looking parts
        # Format: YYYYMMDDTHHMMSSz (uppercase Z)
        start_str = timestamp_parts[-2]  # e.g., "20251001T000000Z"
        end_str   = timestamp_parts[-1]  # e.g., "20251002T000000Z"
        
        # Parse the datetime strings
        start_str_clean = start_str.rstrip('Zz')
        end_str_clean   = end_str.rstrip('Zz')
        
        file_start_dt = datetime.strptime(start_str_clean, '%Y%m%dT%H%M%S')
        file_end_dt = datetime.strptime(end_str_clean, '%Y%m%dT%H%M%S')
        
        # Check if this file covers the desired timespan (with tolerance)
        start_ok = file_start_dt <= (desired_time_o_dt + time_tolerance)
        end_ok = file_end_dt >= (desired_time_f_dt - time_tolerance)
        
        if start_ok and end_ok:
          return filepath
          
      except (ValueError, IndexError):
        # Skip files with unparseable timestamps
        continue
  
  return None


def get_celestrak_tle(
  norad_id          : str,
  object_name       : str,
  tles_folderpath   : Path,
  desired_time_o_dt : datetime,
  desired_time_f_dt : datetime,
  auto_download     : bool = False,
) -> Optional[TLEData]:
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
    auto_download : bool
      If True, automatically download TLE if no compatible file is found.
      
  Output:
  -------
    result : TLEData | None
      TLEData object if successful, None if loading failed.
  """
  # Display TLE folderpath
  try:
    rel_folderpath = tles_folderpath.relative_to(Path.cwd())
    display_folderpath = f"<project_folderpath>/{rel_folderpath}"
  except ValueError:
    display_folderpath = tles_folderpath

  print("  Celestrak TLE")
  print(f"    Folderpath : {display_folderpath}")

  # Search for compatible TLE files first
  compatible_file = find_compatible_tle_file(
    norad_id          = norad_id,
    tles_folderpath   = tles_folderpath,
    desired_time_o_dt = desired_time_o_dt,
    desired_time_f_dt = desired_time_f_dt,
  )
  
  if compatible_file is not None:
    tle_filepath = compatible_file
  else:
    # No compatible file found
    tle_filepath = None
  
  # Check if TLE file exists
  if tle_filepath is None or not tle_filepath.exists():
    # Prompt user to download
    if auto_download:
      user_response = 'y'
    else:
      print(f"               :   ... No compatible files found. Download from Celestrak? (y/n)", end=" ", flush=True)
      user_response = input().strip().lower()
    
    if user_response == 'y':
      print(f"               :   ... Downloading {object_name} ({norad_id}) ...", end=" ", flush=True)
      
      try:
        # Construct command to run the standalone TLE download module
        cmd = [
          sys.executable, "-m", "src.download.tles",
          norad_id,
          desired_time_o_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
          desired_time_f_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Done")
        
        # Search again for the downloaded file
        compatible_file = find_compatible_tle_file(
          norad_id          = norad_id,
          tles_folderpath   = tles_folderpath,
          desired_time_o_dt = desired_time_o_dt,
          desired_time_f_dt = desired_time_f_dt,
        )
        
        if compatible_file:
          tle_filepath = compatible_file
        else:
          print("Failed - Downloaded file not found")
          return None
          
      except subprocess.CalledProcessError as e:
        print(f"Error")
        print(f"               :   ... stderr: {e.stderr}")
        print(f"               :   ... stdout: {e.stdout}")
        return None
      except Exception as e:
        print(f"Error : {e}")
        return None
    else:
      print(f"    Filepath   : None (no compatible file available)")
      return None

  # Display the file being loaded
  try:
    rel_path = tle_filepath.relative_to(Path.cwd())
    display_path = f"<project_folderpath>/{rel_path}"
  except ValueError:
    display_path = tle_filepath
  print(f"    Filepath   : {display_path}")

  # Load TLE from file
  tle_data = load_tle_file(tle_filepath, desired_time_o_dt)
  
  if tle_data is not None:
    print(f"    TLE Epoch  : {tle_data.epoch_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC (closest to desired initial time)")
    # Set the NORAD ID if not already set
    if tle_data.norad_id is None:
      tle_data.norad_id = norad_id
  
  return tle_data


def load_tle_file(
  filepath          : Path,
  reference_time_dt : Optional[datetime] = None,
) -> Optional[TLEData]:
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
    result : TLEData | None
      TLEData object if successful, None if failed.
  """
  try:
    with open(filepath, 'r') as f:
      lines = [line.strip() for line in f.readlines() if line.strip()]
    
    if len(lines) < 2:
      return None
    
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
      return None
    
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
    
    return TLEData(
      line_1      = closest_tle['tle_line_1'],
      line_2      = closest_tle['tle_line_2'],
      epoch_dt    = closest_tle['tle_epoch_dt'],
      object_name = closest_tle['tle_line_0'] if closest_tle['tle_line_0'] else None,
    )
    
  except Exception as e:
    return None
