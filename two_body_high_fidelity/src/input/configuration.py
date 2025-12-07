from pathlib  import Path
from datetime import datetime, timedelta
from types    import SimpleNamespace
from typing   import Optional
from sgp4.api import Satrec

from src.input.loader        import load_supported_objects
from src.input.cli           import parse_time
from src.utility.tle_helper  import get_tle_satellite_and_tle_epoch


def print_input_configuration(
  input_object_type    : str,
  norad_id             : str,
  desired_timespan     : list,
  include_spice        : bool,
  include_drag         : bool,
  compare_tle          : bool,
  compare_jpl_horizons : bool,
  third_bodies_list    : list,
  zonal_harmonics_list : list,
  include_srp          : bool,
  initial_state_source : str,
) -> None:
  """
  Print the input configuration in a formatted table.

  Input:
  ------
    input_object_type : str
      Type of input object (e.g., 'norad-id').
    norad_id : str
      NORAD catalog ID of the satellite.
    desired_timespan : list
      Initial and final time in ISO format (e.g., ['2025-10-01T00:00:00', '2025-10-02T00:00:00']) as list of strings.
    include_spice : bool
      Flag to enable/disable SPICE usage.
    include_drag : bool
      Flag to enable/disable Drag force modeling.
    compare_tle : bool
      Flag to enable/disable TLE comparison.
    compare_jpl_horizons : bool
      Flag to enable/disable JPL Horizons comparison.
    third_bodies_list : list
      List of third bodies to include (e.g., ['SUN', 'MOON']). Empty list if disabled.
    zonal_harmonics_list : list
      List of zonal harmonics to include (e.g., ['J2', 'J3']). Empty list if disabled.
    include_srp : bool
      Flag to enable/disable Solar Radiation Pressure.
    initial_state_source : str
      Source for the initial state vector ('jpl_horizons' or 'tle').
  """
  # Define defaults for comparison
  defaults = {
    'input_object_type'    : None,
    'norad_id'             : None,
    'timespan'             : None,
    'initial_state_source' : 'jpl_horizons',
    'zonal_harmonics'      : [],
    'third_bodies'         : [],
    'include_drag'         : False,
    'include_srp'          : False,
    'include_spice'        : False,
    'compare_jpl_horizons' : False,
    'compare_tle'          : False,
  }
  
  # Format values for display
  timespan_str = f"{desired_timespan[0]} {desired_timespan[1]}" if desired_timespan else "None"
  zonal_str    = ' '.join(zonal_harmonics_list) if zonal_harmonics_list else "None"
  third_str    = ' '.join(third_bodies_list) if third_bodies_list else "None"
  
  # Build configuration entries: (name, value, default, is_explicit)
  entries = [
    ('input_object_type',    input_object_type,    defaults['input_object_type'],    input_object_type is not None),
    ('norad_id',             norad_id,             defaults['norad_id'],             norad_id is not None and norad_id != ''),
    ('timespan',             timespan_str,         defaults['timespan'],             desired_timespan is not None),
    ('initial_state_source', initial_state_source, defaults['initial_state_source'], initial_state_source != defaults['initial_state_source']),
    ('zonal_harmonics',      zonal_str,            defaults['zonal_harmonics'],      len(zonal_harmonics_list) > 0),
    ('third_bodies',         third_str,            defaults['third_bodies'],         len(third_bodies_list) > 0),
    ('include_drag',         include_drag,         defaults['include_drag'],         include_drag != defaults['include_drag']),
    ('include_srp',          include_srp,          defaults['include_srp'],          include_srp != defaults['include_srp']),
    ('include_spice',        include_spice,        defaults['include_spice'],        include_spice != defaults['include_spice']),
    ('compare_jpl_horizons', compare_jpl_horizons, defaults['compare_jpl_horizons'], compare_jpl_horizons != defaults['compare_jpl_horizons']),
    ('compare_tle',          compare_tle,          defaults['compare_tle'],          compare_tle != defaults['compare_tle']),
  ]
  
  # Convert entries to strings for width calculation
  headers = ['Argument', 'Value', 'Default', 'Explicit']
  rows = []
  for name, value, default, is_explicit in entries:
    rows.append([
      name,
      str(value) if value is not None else "None",
      str(default) if default is not None else "None",
      str(is_explicit),
    ])
  
  # Calculate column widths: max of header and all values, plus 4 for spacing
  min_spacing = 4
  col_widths = []
  for col_idx in range(len(headers)):
    max_len = len(headers[col_idx])
    for row in rows:
      max_len = max(max_len, len(row[col_idx]))
    col_widths.append(max_len + min_spacing)
  
  # Print table
  print("\nInput Configuration")
  header_line = "  " + "".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
  print(header_line)
  separator_line = "  " + "".join(("-" * (col_widths[i] - min_spacing)).ljust(col_widths[i]) for i in range(len(headers)))
  print(separator_line)
  
  for row in rows:
    row_line = "  " + "".join(row[col_idx].ljust(col_widths[col_idx]) for col_idx in range(len(row)))
    print(row_line)


def print_paths(config: SimpleNamespace) -> None:
  """
  Print the paths configuration.
  
  Input:
  ------
    config : SimpleNamespace
      Configuration object containing path attributes.
  """
  data_folderpath = config.output_folderpath.parent / 'data'
  
  print("\nPaths and Files Setup")
  print(f"  Output Folderpath          : {config.output_folderpath}")
  print(f"    Timestamp Folderpath     : <output_folderpath>/{config.timestamp_folderpath.relative_to(config.output_folderpath)}")
  print(f"    Figures Folderpath       : <output_folderpath>/{config.figures_folderpath.relative_to(config.output_folderpath)}")
  print(f"    Files Folderpath         : <output_folderpath>/{config.files_folderpath.relative_to(config.output_folderpath)}")
  print(f"    Log Filepath             : <output_folderpath>/{config.log_filepath.relative_to(config.output_folderpath)}")
  print(f"  Data Folderpath            : {data_folderpath}")
  print(f"    SPICE Kernels Folderpath : <data_folderpath>/{config.spice_kernels_folderpath.relative_to(data_folderpath)}")
  print(f"    LSK Filepath             : <data_folderpath>/{config.lsk_filepath.relative_to(data_folderpath)}")
  print(f"    JPL Horizons Filepath    : <data_folderpath>/{config.jpl_horizons_filepath.relative_to(data_folderpath)}")


def print_configuration(
  config : SimpleNamespace,
) -> None:
  """
  Print the complete configuration (input arguments and paths).
  
  Input:
  ------
    config : SimpleNamespace
      Configuration object containing all input and path attributes.
  """
  print_input_configuration(
    input_object_type    = config.input_object_type,
    norad_id             = config.norad_id,
    desired_timespan     = config.desired_timespan,
    include_spice        = config.include_spice,
    include_drag         = config.include_drag,
    compare_tle          = config.compare_tle,
    compare_jpl_horizons = config.compare_jpl_horizons,
    third_bodies_list    = config.third_bodies_list,
    zonal_harmonics_list = config.zonal_harmonics_list,
    include_srp          = config.include_srp,
    initial_state_source = config.initial_state_source,
  )
  
  print_paths(config)


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
  compare_tle          : bool           = False,
  compare_jpl_horizons : bool           = False,
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

  # Get object name
  object_name = obj_props.get('name', 'object_name')

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
    obj_name          = object_name,
    desired_time_o_dt = desired_time_o_dt,
    desired_time_f_dt = desired_time_f_dt,
  )
  
  return SimpleNamespace(
    # Store original input values for print_configuration
    input_object_type    = input_object_type,
    norad_id             = norad_id,
    desired_timespan     = desired_timespan,
    # Parsed and calculated values
    obj_props                = obj_props,
    object_name              = object_name,
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
    compare_tle              = compare_tle,
    compare_jpl_horizons     = compare_jpl_horizons,
    include_third_body       = include_third_body,
    third_bodies_list        = third_bodies_list,
    include_zonal_harmonics  = include_zonal_harmonics,
    zonal_harmonics_list     = zonal_harmonics_list,
    include_srp              = include_srp,
    initial_state_source     = initial_state_source,
    output_folderpath        = paths['output_folderpath'],
    timestamp_folderpath     = paths['timestamp_folderpath'],
    figures_folderpath       = paths['figures_folderpath'],
    files_folderpath         = paths['files_folderpath'],
    log_filepath             = paths['log_filepath'],
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
    print_paths : bool
      Flag to enable/disable printing of paths.
      
  Output:
  -------
    dict
      A dictionary containing paths to output, data, SPICE kernels,
      Horizons ephemeris, and leap seconds files.
  """
  # Project and data paths
  #   Adjusted for location in src/input/configuration.py (depth: src/input/configuration.py -> input -> src -> root)
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
  output_folderpath    = project_root / 'output'
  timestamp_folderpath = output_folderpath / timestamp_str
  figures_folderpath   = timestamp_folderpath / 'figures'
  files_folderpath     = timestamp_folderpath / 'files'
  log_filepath         = files_folderpath / 'output.log'
  
  # Ensure output directory exists
  figures_folderpath.mkdir(parents=True, exist_ok=True)
  files_folderpath.mkdir(parents=True, exist_ok=True)

  return {
    'output_folderpath'        : output_folderpath,
    'timestamp_folderpath'     : timestamp_folderpath,
    'figures_folderpath'       : figures_folderpath,
    'files_folderpath'         : files_folderpath,
    'log_filepath'             : log_filepath,
    'spice_kernels_folderpath' : spice_kernels_folderpath,
    'jpl_horizons_filepath'    : jpl_horizons_filepath,
    'lsk_filepath'             : lsk_filepath,
  }
