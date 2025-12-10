from pathlib  import Path
from datetime import datetime
from types    import SimpleNamespace
from typing   import Optional

from src.input.loader        import load_supported_objects
from src.utility.time_helper import parse_time


def print_input_configuration(
  initial_state_norad_id : str,
  desired_timespan       : list,
  include_drag           : bool,
  compare_tle            : bool,
  compare_jpl_horizons   : bool,
  third_bodies_list      : list,
  zonal_harmonics_list   : list,
  include_srp            : bool,
  initial_state_source   : str,
) -> None:
  """
  Print the input configuration in a formatted table.

  Input:
  ------
    initial_state_norad_id : str
      NORAD catalog ID of the satellite.
    desired_timespan : list
      Initial and final time in ISO format as list of strings.
    include_drag : bool
      Flag to enable/disable Drag force modeling.
    compare_tle : bool
      Flag to enable/disable TLE comparison.
    compare_jpl_horizons : bool
      Flag to enable/disable JPL Horizons comparison.
    third_bodies_list : list
      List of third bodies to include. Empty list if disabled.
    zonal_harmonics_list : list
      List of zonal harmonics to include. Empty list if disabled.
    include_srp : bool
      Flag to enable/disable Solar Radiation Pressure.
    initial_state_source : str
      Source for the initial state vector ('jpl_horizons' or 'tle').
      
  Output:
  -------
    None
  """
  # Define defaults for comparison
  defaults = {
    'initial_state_norad_id' : None,
    'timespan'               : None,
    'initial_state_source'   : 'jpl_horizons',
    'zonal_harmonics'        : [],
    'third_bodies'           : [],
    'include_drag'           : False,
    'include_srp'            : False,
    'compare_jpl_horizons'   : False,
    'compare_tle'            : False,
  }
  
  # Format values for display
  timespan_str = f"{desired_timespan[0]} {desired_timespan[1]}" if desired_timespan else "None"
  zonal_str    = ' '.join(zonal_harmonics_list) if zonal_harmonics_list else "None"
  third_str    = ' '.join(third_bodies_list)    if third_bodies_list    else "None"
  
  # Build configuration entries: (name, value, default, user_set)
  entries = [
    ('initial_state_source',   initial_state_source,   defaults['initial_state_source'],   initial_state_source   != defaults['initial_state_source']),
    ('initial_state_norad_id', initial_state_norad_id, defaults['initial_state_norad_id'], initial_state_norad_id is not None),
    ('timespan',               timespan_str,           defaults['timespan'],               desired_timespan       is not None),
    ('zonal_harmonics',        zonal_str,              defaults['zonal_harmonics'],        zonal_harmonics_list   is not None and len(zonal_harmonics_list) > 0),
    ('third_bodies',           third_str,              defaults['third_bodies'],           third_bodies_list      is not None and len(third_bodies_list) > 0),
    ('include_drag',           include_drag,           defaults['include_drag'],           include_drag           != defaults['include_drag']),
    ('include_srp',            include_srp,            defaults['include_srp'],            include_srp            != defaults['include_srp']),
    ('compare_jpl_horizons',   compare_jpl_horizons,   defaults['compare_jpl_horizons'],   compare_jpl_horizons   != defaults['compare_jpl_horizons']),
    ('compare_tle',            compare_tle,            defaults['compare_tle'],            compare_tle            != defaults['compare_tle']),
  ]
  
  # Convert entries to strings for width calculation
  headers = ['Argument', 'Value', 'Default', 'User Set']
  rows = []
  for name, value, default, user_set in entries:
    rows.append([
      name,
      str(value) if value is not None else "None",
      str(default) if default is not None else "None",
      str(user_set),
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


def print_paths(
  config : SimpleNamespace,
) -> None:
  """
  Print the paths configuration.
  
  Input:
  ------
    config : SimpleNamespace
      Configuration object containing path attributes.
      
  Output:
  -------
    None
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
  print(f"    JPL Horizons Folderpath  : <data_folderpath>/{config.jpl_horizons_folderpath.relative_to(data_folderpath)}")
  print(f"    TLEs Folderpath          : <data_folderpath>/{config.tles_folderpath.relative_to(data_folderpath)}")


def print_configuration(
  config : SimpleNamespace,
) -> None:
  """
  Print the complete configuration (input arguments and paths).
  
  Input:
  ------
    config : SimpleNamespace
      Configuration object containing all input and path attributes.
      
  Output:
  -------
    None
  """
  print_input_configuration(
    initial_state_norad_id = config.initial_state_norad_id,
    desired_timespan       = config.desired_timespan,
    include_drag           = config.include_drag,
    compare_tle            = config.compare_tle,
    compare_jpl_horizons   = config.compare_jpl_horizons,
    third_bodies_list      = config.third_bodies_list,
    zonal_harmonics_list   = config.zonal_harmonics_list,
    include_srp            = config.include_srp,
    initial_state_source   = config.initial_state_source,
  )
  
  print_paths(config)


def normalize_input(
  initial_state_source : str,
) -> str:
  """
  Normalize input string for initial state source.
  
  Input:
  ------
    initial_state_source : str
      Source for the initial state vector (e.g., 'jpl-horizons').
  
  Output:
  -------
    initial_state_source : str
      Normalized initial state source.
  """
  # Normalize initial state source
  initial_state_source = initial_state_source.lower().replace('-', '_').replace(' ', '_')
  if 'horizons' in initial_state_source:
    initial_state_source = 'jpl_horizons'
  
  # Return normalized values
  return initial_state_source


def build_config(
  initial_state_norad_id : str,
  timespan_dt            : list[datetime],
  include_drag           : bool           = False,
  compare_tle            : bool           = False,
  compare_jpl_horizons   : bool           = False,
  third_bodies           : Optional[list] = None,
  zonal_harmonics        : Optional[list] = None,
  include_srp            : bool           = False,
  initial_state_source   : str            = 'jpl_horizons',
) -> SimpleNamespace:
  """
  Parse, validate, and set up input parameters for orbit propagation.
  
  Input:
  ------
    initial_state_norad_id : str
      NORAD catalog ID of the satellite.
    timespan_dt : list[datetime]
      Initial and final time as list of datetime objects.
    include_drag : bool
      Flag to enable/disable Drag force modeling.
    compare_tle : bool
      Flag to enable/disable TLE comparison.
    compare_jpl_horizons : bool
      Flag to enable/disable JPL Horizons comparison.
    third_bodies : list | None
      List of third bodies to include. None if disabled.
    zonal_harmonics : list | None
      List of zonal harmonics to include. None disables all.
      Empty list [] also disables all zonal harmonics.
    include_srp : bool
      Flag to enable/disable Solar Radiation Pressure.
    initial_state_source : str
      Source for the initial state vector ('jpl_horizons' or 'tle').
  
  Output:
  -------
    config : SimpleNamespace
      Configuration object containing parsed and calculated propagation parameters.
  
  Raises:
  -------
    ValueError
      If NORAD ID is not supported.
  """
  
  # Normalize inputs
  initial_state_source = normalize_input(
    initial_state_source,
  )
  
  # Handle zonal harmonics logic
  include_zonal_harmonics = zonal_harmonics is not None and len(zonal_harmonics) > 0
  zonal_harmonics_list    = zonal_harmonics if zonal_harmonics is not None else []

  # Handle third bodies logic
  include_third_body = third_bodies is not None and len(third_bodies) > 0
  third_bodies_list  = [b.upper() for b in third_bodies] if third_bodies is not None else []

  # Unpack timespan
  time_o_dt = timespan_dt[0]
  time_f_dt = timespan_dt[1]
  delta_time_s = (time_f_dt - time_o_dt).total_seconds()
  
  # Validate: NORAD ID required
  if not initial_state_norad_id:
    raise ValueError("Initial State NORAD ID is required")

  # Validate: norad is in supported objects
  supported_objects = load_supported_objects()
  if initial_state_norad_id not in supported_objects:
    raise ValueError(f"NORAD ID {initial_state_norad_id} is not supported. Supported IDs: {list(supported_objects.keys())}")

  # Get object properties
  obj_props = supported_objects[initial_state_norad_id]

  # Get object name
  object_name = obj_props.get('name', 'object_name')

  # Set up paths and files
  paths = setup_paths_and_files(
    norad_id   = initial_state_norad_id,
    obj_name   = object_name,
    time_o_dt  = time_o_dt,
    time_f_dt  = time_f_dt,
  )
  
  return SimpleNamespace(
    # Store original input values for print_configuration
    initial_state_norad_id = initial_state_norad_id,
    desired_timespan       = timespan_dt,  # Keep for print_configuration
    # Parsed and calculated values
    obj_props                = obj_props,
    object_name              = object_name,
    time_o_dt                = time_o_dt,
    time_f_dt                = time_f_dt,
    delta_time_s             = delta_time_s,
    mass                     = obj_props['mass__kg'],
    cd                       = obj_props['drag']['coeff'],
    area_drag                = obj_props['drag']['area__m2'],
    cr                       = obj_props['srp']['coeff'],
    area_srp                 = obj_props['srp']['area__m2'],
    include_spice            = True,
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
    jpl_horizons_folderpath  = paths['jpl_horizons_folderpath'],
    tles_folderpath          = paths['tles_folderpath'],
    lsk_filepath             = paths['lsk_filepath'],
    # Values calculated later
    tle_line_0   = None,
    tle_line_1   = None,
    tle_line_2   = None,
    tle_epoch_dt = None,
  )


def setup_paths_and_files(
  norad_id  : str,
  obj_name  : str,
  time_o_dt : datetime,
  time_f_dt : datetime,
) -> dict:
  """
  Set up all required folder paths and file names for the propagation.
  
  Input:
  ------
    norad_id : str
      NORAD catalog ID of the satellite.
    obj_name : str
      Name of the object (e.g., 'ISS').
    time_o_dt : datetime
      Initial time as a datetime object.
    time_f_dt : datetime
      Final time as a datetime object.
      
  Output:
  -------
    paths : dict
      A dictionary containing paths to output, data, SPICE kernels,
      Horizons ephemeris folder, TLEs folder, and leap seconds files.
  """
  # Project and data paths
  #   Adjusted for location in src/input/configuration.py (depth: src/input/configuration.py -> input -> src -> root)
  project_root    = Path(__file__).parent.parent.parent
  data_folderpath = project_root / 'data'
  
  # SPICE kernels path
  spice_kernels_folderpath = data_folderpath / 'spice_kernels'
  lsk_filepath             = spice_kernels_folderpath / 'naif0012.tls'
  
  # TLEs folderpath
  tles_folderpath = data_folderpath / 'tles'
  
  # Horizons ephemeris folder (loader will search for compatible files)
  jpl_horizons_folderpath = data_folderpath / 'ephems'
  
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
    'jpl_horizons_folderpath'  : jpl_horizons_folderpath,
    'tles_folderpath'          : tles_folderpath,
    'lsk_filepath'             : lsk_filepath,
  }


def extract_tle_to_config(
  config               : SimpleNamespace,
  result_celestrak_tle : Optional[dict],
) -> None:
  """
  Extract TLE data from result dictionary and store on config object.
  
  Input:
  ------
    config : SimpleNamespace
      Configuration object to store TLE data on.
    result_celestrak_tle : dict | None
      Result dictionary from get_celestrak_tle containing TLE data.
      
  Output:
  -------
    None
  """
  if result_celestrak_tle and result_celestrak_tle.get('success'):
    config.tle_line_0   = result_celestrak_tle['tle_line_0']
    config.tle_line_1   = result_celestrak_tle['tle_line_1']
    config.tle_line_2   = result_celestrak_tle['tle_line_2']
    config.tle_epoch_dt = result_celestrak_tle['tle_epoch_dt']
