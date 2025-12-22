import yaml
import numpy as np
import os

from pathlib  import Path
from datetime import datetime
from types    import SimpleNamespace
from typing   import Optional

from src.input.loader        import load_supported_objects
from src.utility.time_helper import parse_time


def print_input_configuration(
  initial_state_source       : str,
  initial_state_norad_id     : Optional[str],
  initial_state_filename     : Optional[str],
  desired_timespan           : list,
  include_drag               : bool,
  compare_tle                : bool,
  compare_jpl_horizons       : bool,
  third_bodies_list          : list,
  gravity_harmonics_list     : list,
  two_body_gravity_model     : SimpleNamespace,
  include_srp                : bool,
) -> None:
  """
  Print the input configuration in a formatted table.

  Input:
  ------
    initial_state_source : str
      Source for the initial state vector ('jpl_horizons' or 'tle').
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
    gravity_harmonics_list : list
      List of gravity harmonics to include. Empty list if disabled.
    two_body_gravity_model : SimpleNamespace
      Nested namespace containing gravity model configuration.
    include_srp : bool
      Flag to enable/disable Solar Radiation Pressure.
      
  Output:
  -------
    None
  """
  # Define defaults for comparison
  defaults = {
    'initial_state_source'       : 'jpl_horizons',
    'initial_state_norad_id'     : None,
    'initial_state_filename'     : None,
    'timespan'                   : None,
    'gravity_harmonics'          : [],
    'gravity_harmonics_degree'   : None,
    'gravity_harmonics_order'    : None,
    'gravity_harmonics_filename' : 'EGM2008.gfc',
    'third_bodies'               : [],
    'include_drag'               : False,
    'include_srp'                : False,
    'compare_jpl_horizons'       : False,
    'compare_tle'                : False,
  }
  
  # Format values for display
  timespan_str     = f"{desired_timespan[0]} {desired_timespan[1]}" if desired_timespan else "None"
  harmonics_str    = ' '.join(gravity_harmonics_list) if gravity_harmonics_list else "None"
  gh_deg_order_str = f"{two_body_gravity_model.spherical_harmonics.degree} {two_body_gravity_model.spherical_harmonics.order}" if two_body_gravity_model.spherical_harmonics.degree is not None else "None"
  third_str        = ' '.join(third_bodies_list) if third_bodies_list else "None"
  
  # Build configuration entries: (name, value, default, user_set)
  entries = [
    ('initial_state_source',   initial_state_source,       defaults['initial_state_source'],       initial_state_source       != defaults['initial_state_source']),
    ('initial_state_norad_id', initial_state_norad_id,     defaults['initial_state_norad_id'],     initial_state_norad_id     is not None),
    ('initial_state_filename', initial_state_filename,     defaults['initial_state_filename'],     initial_state_filename     is not None),
    ('timespan',               timespan_str,               defaults['timespan'],                   desired_timespan           is not None),
    ('gravity_harmonics',      harmonics_str,              defaults['gravity_harmonics'],          gravity_harmonics_list     is not None and len(gravity_harmonics_list) > 0),
    ('gravity_degree_order',   gh_deg_order_str,           "None",                                 two_body_gravity_model.spherical_harmonics.degree is not None),
    ('gravity_file',           two_body_gravity_model.filename, defaults['gravity_harmonics_filename'], two_body_gravity_model.filename != defaults['gravity_harmonics_filename']),
    ('third_bodies',           third_str,                  defaults['third_bodies'],               third_bodies_list          is not None and len(third_bodies_list) > 0),
    ('include_drag',           include_drag,               defaults['include_drag'],               include_drag               != defaults['include_drag']),
    ('include_srp',            include_srp,                defaults['include_srp'],                include_srp                != defaults['include_srp']),
    ('compare_jpl_horizons',   compare_jpl_horizons,       defaults['compare_jpl_horizons'],       compare_jpl_horizons       != defaults['compare_jpl_horizons']),
    ('compare_tle',            compare_tle,                defaults['compare_tle'],                compare_tle                != defaults['compare_tle']),
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
  data_folderpath = config.spice_kernels_folderpath.parent
  
  print("\nPaths and Files Setup")
  print(f"  Output Folderpath          : {config.output_folderpath}")
  print(f"    Timestamp Folderpath     : <output_folderpath>/{config.timestamp_folderpath.relative_to(config.output_folderpath)}")
  print(f"    Figures Folderpath       : <output_folderpath>/{config.figures_folderpath.relative_to(config.output_folderpath)}")
  print(f"    Files Folderpath         : <output_folderpath>/{config.files_folderpath.relative_to(config.output_folderpath)}")
  print(f"    Log Filepath             : <output_folderpath>/{config.log_filepath.relative_to(config.output_folderpath)}")
  print(f"  Data Folderpath            : {data_folderpath}")
  print(f"    SPICE Kernels Folderpath : <data_folderpath>/{config.spice_kernels_folderpath.relative_to(data_folderpath)}")
  print(f"    LSK Filepath             : <data_folderpath>/{config.lsk_filepath.relative_to(data_folderpath)}")
  print(f"    Gravity Folderpath       : <data_folderpath>/{config.two_body_gravity_model.folderpath.relative_to(data_folderpath)}")
  print(f"    JPL Horizons Folderpath  : <data_folderpath>/{config.jpl_horizons_folderpath.relative_to(data_folderpath)}")
  print(f"    TLEs Folderpath          : <data_folderpath>/{config.tles_folderpath.relative_to(data_folderpath)}")
  print(f"    State Vectors Folderpath : <data_folderpath>/{config.state_vectors_folderpath.relative_to(data_folderpath)}")


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
    initial_state_source       = config.initial_state_source,
    initial_state_norad_id     = config.initial_state_norad_id,
    initial_state_filename     = config.initial_state_filename,
    desired_timespan           = config.desired_timespan,
    include_drag               = config.include_drag,
    compare_tle                = config.compare_tle,
    compare_jpl_horizons       = config.compare_jpl_horizons,
    third_bodies_list          = config.third_bodies_list,
    gravity_harmonics_list     = config.gravity_harmonics_list,
    two_body_gravity_model     = config.two_body_gravity_model,
    include_srp                = config.include_srp,
  )
  
  print_paths(config)


def normalize_input(
  initial_state_source : str,
  gravity_harmonics    : Optional[list] = None,
) -> tuple[str, list]:
  """
  Normalize input strings.
  
  Input:
  ------
    initial_state_source : str
      Source for the initial state vector (e.g., 'jpl-horizons').
    gravity_harmonics : list | None
      List of gravity harmonics.
  
  Output:
  -------
    initial_state_source : str
      Normalized initial state source.
    gravity_harmonics_list : list
      Normalized gravity harmonics list (uppercase).
  """
  # Normalize initial state source
  initial_state_source = initial_state_source.lower().replace('-', '_').replace(' ', '_')
  
  if 'horizons' in initial_state_source:
    initial_state_source = 'jpl_horizons'
  elif initial_state_source in ['sv', 'custom_sv', 'custom_state_vector', 'state_vector']:
    initial_state_source = 'custom_state_vector'
  
  # Normalize gravity harmonics
  gravity_harmonics_list = [h.upper() for h in gravity_harmonics] if gravity_harmonics is not None else []
  
  # Return normalized values
  return initial_state_source, gravity_harmonics_list


def build_config(
  initial_state_norad_id         : Optional[str],
  initial_state_filename         : Optional[str],
  timespan_dt                    : list[datetime],
  include_drag                   : bool           = False,
  compare_tle                    : bool           = False,
  compare_jpl_horizons           : bool           = False,
  third_bodies                   : Optional[list] = None,
  gravity_harmonics              : Optional[list] = None,
  include_srp                    : bool           = False,
  initial_state_source           : str            = 'jpl_horizons',
  gravity_harmonics_degree_order : Optional[list] = None,
  gravity_model_filename         : Optional[str]  = None,
) -> SimpleNamespace:
  """
  Parse, validate, and set up input parameters for orbit propagation.
  
  Input:
  ------
    initial_state_source : str
      Source for the initial state vector ('jpl_horizons', 'tle', or 'custom_state_vector').
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
    gravity_harmonics : list | None
      List of gravity harmonics to include (e.g., ['J2', 'J3', 'J4']).
      None or empty list disables all harmonics.
    include_srp : bool
      Flag to enable/disable Solar Radiation Pressure.
  
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
  initial_state_source, gravity_harmonics_list = normalize_input(
    initial_state_source,
    gravity_harmonics,
  )
  
  # Validate: cannot use both gravity harmonics options
  has_coefficients = gravity_harmonics_list is not None and len(gravity_harmonics_list) > 0
  has_degree_order = gravity_harmonics_degree_order is not None
  
  if has_coefficients and has_degree_order:
    raise ValueError(
      "Cannot use both --gravity-harmonics-coefficients and --gravity-harmonics-degree-order. "
      "Please use only one method to specify gravity harmonics."
    )
  
  # Handle gravity harmonics logic
  include_gravity_harmonics = len(gravity_harmonics_list) > 0

  if gravity_harmonics_degree_order is not None:
    if len(gravity_harmonics_degree_order) != 2:
      raise ValueError("--gravity-harmonics-degree-order requires exactly 2 values: DEGREE ORDER")
    gravity_harmonics_degree = gravity_harmonics_degree_order[0]
    gravity_harmonics_order  = gravity_harmonics_degree_order[1]
  else:
    gravity_harmonics_degree = None
    gravity_harmonics_order  = None

  # Handle third bodies logic
  include_third_body = third_bodies is not None and len(third_bodies) > 0
  third_bodies_list  = [b.upper() for b in third_bodies] if third_bodies is not None else []

  # Unpack timespan
  time_o_dt    = timespan_dt[0]
  time_f_dt    = timespan_dt[1]
  delta_time_s = (time_f_dt - time_o_dt).total_seconds()
  
  # Set up foldernames, folderpaths, filenames, and filepaths
  paths = setup_paths(
    initial_state_source   = initial_state_source,
    initial_state_filename = initial_state_filename,
    gravity_model_filename = gravity_model_filename,
  )

  # Initialize variables
  obj_props           = {}
  custom_state_vector = None

  # --- Logic for Custom State Vector ---
  if initial_state_source == 'custom_state_vector':
    # Validation
    if initial_state_norad_id is not None:
      print(f"[WARNING] --initial-state-norad-id ({initial_state_norad_id}) is ignored when using custom state vector.")
      # Or terminate as requested:
      raise ValueError("Argument --initial-state-norad-id is not allowed when using a custom state vector.")
      
    if compare_tle or compare_jpl_horizons:
      raise ValueError("Comparisons (TLE/Horizons) are not allowed when using a custom state vector.")
      
    # Load Custom State Vector File
    sv_filepath = paths['custom_state_vector_filepath']
      
    with open(sv_filepath, 'r') as f:
      sv_data = yaml.safe_load(f)
      
    # Extract properties
    object_name = sv_data.get('name', 'CustomObject')
    
    # Ensure defaults for mass, drag, srp if not present
    if 'mass__kg' not in sv_data:
        sv_data['mass__kg'] = 1000.0 # Default mass
    
    # Handle drag defaults
    default_drag = {'coeff': 2.2, 'area__m2': 10.0}
    if 'drag' not in sv_data or sv_data['drag'] is None:
        sv_data['drag'] = default_drag
    else:
        # Fill in missing keys in existing drag dict
        for k, v in default_drag.items():
            if k not in sv_data['drag']:
                sv_data['drag'][k] = v
    
    # Handle SRP defaults
    default_srp = {'coeff': 1.3, 'area__m2': 10.0}
    if 'srp' not in sv_data or sv_data['srp'] is None:
        sv_data['srp'] = default_srp
    else:
        # Fill in missing keys in existing srp dict
        for k, v in default_srp.items():
            if k not in sv_data['srp']:
                sv_data['srp'][k] = v

    obj_props = sv_data
    
    # Extract state
    # Support 'state' (6-element list) OR 'pos_vec__m' and 'vel_vec__m_per_s'
    if 'state' in sv_data:
        custom_state_vector = np.array(sv_data['state'])
    elif 'pos_vec__m' in sv_data and ('vel_vec__m_per_s' in sv_data or 'vec_vec__m_per_s' in sv_data):
        # Handle potential typo in YAML key
        vel_key = 'vel_vec__m_per_s' if 'vel_vec__m_per_s' in sv_data else 'vec_vec__m_per_s'
        
        pos_raw = sv_data['pos_vec__m']
        vel_raw = sv_data[vel_key]
        
        # Helper to parse "x, y, z" string or [x, y, z] list
        def parse_vec3(raw_val):
          if isinstance(raw_val, str):
            # Remove brackets if present and split
            clean = raw_val.replace('[', '').replace(']', '')
            return np.array([float(x.strip()) for x in clean.split(',')])
          elif isinstance(raw_val, list):
            return np.array([float(x) for x in raw_val])
          else:
            raise ValueError(f"Unknown format for vector: {raw_val}")

        pos = parse_vec3(pos_raw)
        vel = parse_vec3(vel_raw)
        custom_state_vector = np.concatenate((pos, vel))
    else:
      raise ValueError(f"Custom state vector file {initial_state_filename} must contain 'state' or 'pos_vec__m'/'vel_vec__m_per_s'.")

  # --- Logic for Standard Sources (Horizons/TLE) ---
  else:
    # Validate: NORAD ID required
    if not initial_state_norad_id:
      raise ValueError("Initial State NORAD ID is required for Horizons/TLE sources.")

    # Validate: norad is in supported objects
    supported_objects = load_supported_objects()
    if initial_state_norad_id not in supported_objects:
      raise ValueError(f"NORAD ID {initial_state_norad_id} is not supported. Supported IDs: {list(supported_objects.keys())}")

    # Get object properties
    obj_props = supported_objects[initial_state_norad_id]

    # Get object name
    object_name = obj_props.get('name', 'object_name')
    
    # Update paths with correct name
    paths = setup_paths()
  
  # Create nested two_body_gravity_model namespace
  two_body_gravity_model = SimpleNamespace(
    folderpath          = paths['gravity_model_folderpath'],
    filename            = paths['gravity_model_filename'],
    spherical_harmonics = SimpleNamespace(
      gp     = None,  # Set after loading
      radius = None,  # Set after loading
      degree = gravity_harmonics_degree,
      order  = gravity_harmonics_order,
      model  = None,  # Loaded later by load_files
    ),
  )

  return SimpleNamespace(
    # Store original input values for print_configuration
    initial_state_norad_id = initial_state_norad_id,
    initial_state_filename = initial_state_filename,
    desired_timespan       = timespan_dt,
    # Parsed and calculated values
    obj_props                  = obj_props,
    object_name                = object_name,
    custom_state_vector        = custom_state_vector,
    time_o_dt                  = time_o_dt,
    time_f_dt                  = time_f_dt,
    delta_time_s               = delta_time_s,
    mass                       = obj_props['mass__kg'],
    cd                         = obj_props['drag']['coeff'],
    area_drag                  = obj_props['drag']['area__m2'],
    cr                         = obj_props['srp']['coeff'],
    area_srp                   = obj_props['srp']['area__m2'],
    include_drag               = include_drag,
    compare_tle                = compare_tle,
    compare_jpl_horizons       = compare_jpl_horizons,
    include_third_body         = include_third_body,
    third_bodies_list          = third_bodies_list,
    include_gravity_harmonics  = include_gravity_harmonics,
    gravity_harmonics_list     = gravity_harmonics_list,
    two_body_gravity_model     = two_body_gravity_model,
    include_srp                = include_srp,
    initial_state_source       = initial_state_source,
    output_folderpath          = paths['output_folderpath'],
    timestamp_folderpath       = paths['timestamp_folderpath'],
    figures_folderpath         = paths['figures_folderpath'],
    files_folderpath           = paths['files_folderpath'],
    log_filepath               = paths['log_filepath'],
    spice_kernels_folderpath   = paths['spice_kernels_folderpath'],
    jpl_horizons_folderpath    = paths['jpl_horizons_folderpath'],
    tles_folderpath            = paths['tles_folderpath'],
    state_vectors_folderpath   = paths['state_vectors_folderpath'],
    lsk_filepath               = paths['lsk_filepath'],
    # Values calculated later
    tle_line_0   = None,
    tle_line_1   = None,
    tle_line_2   = None,
    tle_epoch_dt = None,
  )


def setup_paths(
  initial_state_source   : Optional[str] = None,
  initial_state_filename : Optional[str] = None,
  gravity_model_filename : Optional[str] = None,
) -> dict:
  """
  Set up all required folder paths and file names for the propagation.
  
  Input:
  ------
    initial_state_source : str | None
      Source of initial state. Used to validate custom state vector file.
    initial_state_filename : str | None
      Filename of custom state vector.
    gravity_model_filename : str | None
      Filename of gravity model. If None, defaults to EGM2008.gfc.
      
  Output:
  -------
    paths : dict
      A dictionary containing paths to output, data, SPICE kernels,
      Horizons ephemeris folder, TLEs folder, and leap seconds files.
  """
  # Check for test data override (used by pytest fixtures)
  test_data_path = os.environ.get('ORBIT_PROPAGATOR_TEST_DATA')
  
  if test_data_path:
    # Use test fixtures
    data_folderpath = Path(test_data_path)
    project_root = Path(__file__).parent.parent.parent
  else:
    # Normal operation: use project data folder
    project_root = Path(__file__).parent.parent.parent
    data_folderpath = project_root / 'data'
  
  # SPICE kernels path
  spice_kernels_folderpath = data_folderpath / 'spice_kernels'
  lsk_filepath             = spice_kernels_folderpath / 'naif0012.tls'
  
  # Gravity coefficients path
  gravity_model_folderpath = data_folderpath / 'gravity_models'
  
  if gravity_model_filename is None:
    gravity_model_filename = 'EGM2008.gfc'

  # TLEs folderpath
  tles_folderpath = data_folderpath / 'tles'

  # State Vectors folderpath
  state_vectors_folderpath = data_folderpath / 'state_vectors'
  
  custom_state_vector_filepath = None
  if initial_state_source == 'custom_state_vector':
    if not initial_state_filename:
      raise ValueError("Argument --initial-state-filename is required when using a custom state vector.")
    
    custom_state_vector_filepath = state_vectors_folderpath / initial_state_filename
    if not custom_state_vector_filepath.exists():
      raise FileNotFoundError(f"Custom state vector file not found: {custom_state_vector_filepath}")

  # Horizons ephemeris folder (loader will search for compatible files)
  jpl_horizons_folderpath = data_folderpath / 'ephems'
  
  # Define output folderpath
  timestamp_str        = datetime.now().strftime("%Y%m%d_%H%M%S")
  output_folderpath    = project_root / 'output'
  timestamp_folderpath = output_folderpath / timestamp_str
  figures_folderpath   = timestamp_folderpath / 'figures'
  files_folderpath     = timestamp_folderpath / 'files'
  log_filepath         = files_folderpath / 'output.log'
  
  # Ensure output directory exists
  figures_folderpath.mkdir(parents=True, exist_ok=True)
  files_folderpath.mkdir(parents=True, exist_ok=True)

  return {
    'output_folderpath'            : output_folderpath,
    'timestamp_folderpath'         : timestamp_folderpath,
    'figures_folderpath'           : figures_folderpath,
    'files_folderpath'             : files_folderpath,
    'log_filepath'                 : log_filepath,
    'spice_kernels_folderpath'     : spice_kernels_folderpath,
    'gravity_model_folderpath'     : gravity_model_folderpath,
    'gravity_model_filename'       : gravity_model_filename,
    'jpl_horizons_folderpath'      : jpl_horizons_folderpath,
    'tles_folderpath'              : tles_folderpath,
    'state_vectors_folderpath'     : state_vectors_folderpath,
    'custom_state_vector_filepath' : custom_state_vector_filepath,
    'lsk_filepath'                 : lsk_filepath,
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
