from pathlib  import Path
from datetime import datetime, timedelta
from types    import SimpleNamespace
from typing   import Optional
from sgp4.api import Satrec

from src.input.loader        import load_supported_objects
from src.input.cli           import parse_time
from src.utility.tle_helper  import get_tle_satellite_and_tle_epoch


def normalize_input(
  input_object_type    : str,
  initial_state_source : str,
) -> tuple[str, str]:
  """
  Normalize input strings for object type and initial state source.
  
  Input:
  ------
    input_object_type : str
      Type of input object (e.g., 'norad-id').
    initial_state_source : str
      Source for the initial state vector (e.g., 'jpl-horizons').
  
  Output:
  -------
    tuple[str, str]
      Normalized input object type and initial state source.
  """
  # Normalize input object type
  input_object_type = input_object_type.replace('-', '_').replace(' ', '_')

  # Normalize initial state source
  initial_state_source = initial_state_source.lower().replace('-', '_').replace(' ', '_')
  if 'horizons' in initial_state_source:
    initial_state_source = 'jpl_horizons'
  
  # Return normalized values
  return input_object_type, initial_state_source


def build_config(
  input_object_type    : str,
  norad_id             : str,
  desired_timespan     : list,
  include_spice        : bool           = False,
  include_drag         : bool           = False,
  third_bodies         : Optional[list] = None,
  zonal_harmonics      : Optional[list] = None,
  include_srp          : bool           = False,
  initial_state_source : str            = 'jpl_horizons',
) -> SimpleNamespace:
  """
  Parse, validate, and set up input parameters for orbit propagation.
  
  Input:
  ------
    input_object_type : str
      Type of input object (e.g., norad-id).
    norad_id : str
      NORAD catalog ID of the satellite.
    desired_timespan : list
      Initial and final time in ISO format (e.g., ['2025-10-01T00:00:00', '2025-10-02T00:00:00']) as list of strings.
    use_spice : bool
      Flag to enable/disable SPICE usage.
    include_drag : bool
      Flag to enable/disable Drag force modeling.
    third_bodies : list | None
      List of third bodies to include (e.g., ['SUN', 'MOON']). None if disabled.
    zonal_harmonics : list | None
      List of zonal harmonics to include (e.g., ['J2', 'J3']). None if disabled.
      Empty list [] implies default ['J2'].
    include_srp : bool
      Flag to enable/disable Solar Radiation Pressure.
    initial_state_source : str
      Source for the initial state vector ('jpl_horizons' or 'tle').
  
  Output:
  -------
    SimpleNamespace
      Configuration object containing parsed and calculated propagation parameters.
  
  Raises:
  -------
    ValueError
      If NORAD ID is not supported.
  """
  # Normalize inputs
  input_object_type, initial_state_source = normalize_input(
    input_object_type,
    initial_state_source,
  )

  # Handle zonal harmonics logic
  include_zonal_harmonics = zonal_harmonics is not None
  if zonal_harmonics is not None and len(zonal_harmonics) == 0:
    zonal_harmonics_list = ['J2']
  else:
    zonal_harmonics_list = zonal_harmonics

  # Handle third bodies logic
  include_third_body = third_bodies is not None
  third_bodies_list  = [b.upper() for b in third_bodies] if third_bodies is not None else []

  # Unpack timespan
  desired_time_o_str = desired_timespan[0]
  desired_time_f_str = desired_timespan[1]

  # Validate: NORAD ID required for norad-id input type
  if input_object_type == 'norad_id' and not norad_id:
    raise ValueError("NORAD ID is required when input-object-type is 'norad-id'")

  # Validate: SRP requires SPICE
  if include_srp:
    include_spice = True

  # Validate: norad is insupported objects
  supported_objects = load_supported_objects()
  if norad_id not in supported_objects:
    raise ValueError(f"NORAD ID {norad_id} is not supported. Supported IDs: {list(supported_objects.keys())}")

  # Get object properties
  obj_props = supported_objects[norad_id]

  # Extract TLE lines
  tle_line_1 = obj_props['tle']['line_1']
  tle_line_2 = obj_props['tle']['line_2']

  # Parse TLE epoch
  tle_epoch_dt, _ = get_tle_satellite_and_tle_epoch(tle_line_1, tle_line_2)
  
  # Target propagation start/end times from arguments
  desired_time_o_dt = parse_time(desired_time_o_str)
  desired_time_f_dt = parse_time(desired_time_f_str)

  desired_delta_time_of_s = (desired_time_f_dt - desired_time_o_dt).total_seconds()

  # Set up paths and files
  paths = setup_paths_and_files(
    norad_id          = norad_id,
    obj_name          = obj_props['name'],
    desired_time_o_dt = desired_time_o_dt,
    desired_time_f_dt = desired_time_f_dt,
  )
  
  return SimpleNamespace(
    obj_props                = obj_props,
    tle_line_1               = tle_line_1,
    tle_line_2               = tle_line_2,
    tle_epoch_dt             = tle_epoch_dt,
    desired_time_o_dt        = desired_time_o_dt,
    desired_time_f_dt        = desired_time_f_dt,
    desired_delta_time_of_s  = desired_delta_time_of_s,
    mass                     = obj_props['mass__kg'],
    cd                       = obj_props['drag']['coeff'],
    area_drag                = obj_props['drag']['area__m2'],
    cr                       = obj_props['srp']['coeff'],
    area_srp                 = obj_props['srp']['area__m2'],
    include_spice            = include_spice,
    include_drag             = include_drag,
    include_third_body       = include_third_body,
    third_bodies_list        = third_bodies_list,
    include_zonal_harmonics  = include_zonal_harmonics,
    zonal_harmonics_list     = zonal_harmonics_list if zonal_harmonics_list else [],
    include_srp              = include_srp,
    initial_state_source     = initial_state_source,
    output_folderpath        = paths['output_folderpath'],
    spice_kernels_folderpath = paths['spice_kernels_folderpath'],
    jpl_horizons_filepath    = paths['jpl_horizons_filepath'],
    lsk_filepath             = paths['lsk_filepath'],
  )


def setup_paths_and_files(
  norad_id          : str,
  obj_name          : str,
  desired_time_o_dt : datetime,
  desired_time_f_dt : datetime,
) -> dict:
  """
  Set up all required folder paths and file names for the propagation.
  
  Input:
  ------
    norad_id : str
      NORAD catalog ID of the satellite.
    obj_name : str
      Name of the object (e.g., 'ISS').
    desired_time_o_dt : datetime
      Desired initial time as a datetime object.
    desired_time_f_dt : datetime
      Desired final time as a datetime object.
  
  Output:
  -------
    dict
      A dictionary containing paths to output, data, SPICE kernels,
      Horizons ephemeris, and leap seconds files.
  """
  # Output directory for figures
  output_folderpath = Path('./output/figures')
  output_folderpath.mkdir(parents=True, exist_ok=True)
  
  # Project and data paths
  # Adjusted for location in src/input/configuration.py (depth: src/input/configuration.py -> input -> src -> root)
  project_root    = Path(__file__).parent.parent.parent
  data_folderpath = project_root / 'data'
  
  # SPICE kernels path
  spice_kernels_folderpath = data_folderpath / 'spice_kernels'
  lsk_filepath             = spice_kernels_folderpath / 'naif0012.tls'
  
  # Horizons ephemeris file (dynamically named)
  desired_time_o_str    = desired_time_o_dt.strftime('%Y%m%dT%H%M%SZ')
  desired_time_f_str    = desired_time_f_dt.strftime('%Y%m%dT%H%M%SZ')
  jpl_horizons_filename = f"horizons_ephem_{norad_id}_{obj_name.lower()}_{desired_time_o_str}_{desired_time_f_str}_1m.csv"
  jpl_horizons_filepath = data_folderpath / 'ephems' / jpl_horizons_filename
  
  # Define output folderpath
  timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
  output_folderpath = project_root / 'output' / 'figures' / timestamp_str
  
  # Ensure output directory exists
  output_folderpath.mkdir(parents=True, exist_ok=True)
  
  return {
    'output_folderpath'        : output_folderpath,
    'spice_kernels_folderpath' : spice_kernels_folderpath,
    'jpl_horizons_filepath'    : jpl_horizons_filepath,
    'lsk_filepath'             : lsk_filepath,
  }
