import yaml
import numpy as np
import os

from pathlib  import Path
from datetime import datetime
from typing   import Optional

from src.schemas.config        import OutputPaths, SimulationConfig, InitialStateConfig, ComparisonConfig
from src.schemas.gravity       import GravityModelConfig, SphericalHarmonicsConfig, ThirdBodyConfig, RelativityConfig, SolidEarthTidesConfig, OceanTidesConfig
from src.schemas.spacecraft    import SpacecraftProperties, DragConfig, SRPConfig
from src.schemas.propagation   import PropagationConfig
from src.schemas.state         import TLEData
from src.model.constants       import SOLARSYSTEMCONSTANTS
from src.input.loader          import load_supported_objects
from src.utility.string_helper import sanitize_filename


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
  two_body_gravity_model     : GravityModelConfig,
  include_srp                : bool,
  include_relativity         : bool,
  auto_download              : bool,
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
    two_body_gravity_model : GravityModelConfig
      Gravity model configuration object.
    include_srp : bool
      Flag to enable/disable Solar Radiation Pressure.
      
  Output:
  -------
    None
  """
  # Print header
  title = "Input Configuration"
  print("\n" + "-" * len(title))
  print(title)
  print("-" * len(title))
  print()

  # Progress subsection
  print("  Progress")

  # Define defaults for comparison
  print("    Define default configuration values")
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
    'include_relativity'         : False,
    'compare_jpl_horizons'       : False,
    'compare_tle'                : False,
    'auto_download'              : False,
  }

  # Format values for display
  print("    Format configuration values for display")
  timespan_str     = f"{desired_timespan[0]} {desired_timespan[1]}" if desired_timespan else "None"
  harmonics_str    = ' '.join(gravity_harmonics_list) if gravity_harmonics_list else "None"
  gh_deg_order_str = f"{two_body_gravity_model.spherical_harmonics.degree} {two_body_gravity_model.spherical_harmonics.order}" if two_body_gravity_model.spherical_harmonics.degree is not None else "None"
  third_str        = ' '.join(third_bodies_list) if third_bodies_list else "None"

  # Build configuration entries: (name, value, default, user_set)
  print("    Build configuration table entries")
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
    ('include_relativity',     include_relativity,         defaults['include_relativity'],         include_relativity         != defaults['include_relativity']),
    ('compare_jpl_horizons',   compare_jpl_horizons,       defaults['compare_jpl_horizons'],       compare_jpl_horizons       != defaults['compare_jpl_horizons']),
    ('compare_tle',            compare_tle,                defaults['compare_tle'],                compare_tle                != defaults['compare_tle']),
    ('auto_download',          auto_download,              defaults['auto_download'],              auto_download              != defaults['auto_download']),
  ]

  # Convert entries to strings for width calculation
  print("    Calculate table column widths")
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
  print()

  # Summary subsection
  print("  Summary")
  header_line = "    " + "".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
  print(header_line)
  separator_line = "    " + "".join(("-" * (col_widths[i] - min_spacing)).ljust(col_widths[i]) for i in range(len(headers)))
  print(separator_line)

  for row in rows:
    row_line = "    " + "".join(row[col_idx].ljust(col_widths[col_idx]) for col_idx in range(len(row)))
    print(row_line)


def print_paths(
  config : SimulationConfig,
) -> None:
  """
  Print the paths configuration.
  
  Input:
  ------
    config : SimulationConfig
      Configuration object containing path attributes.
      
  Output:
  -------
    None
  """
  title = "Paths and Files Setup"
  print("\n" + "-" * len(title))
  print(title)
  print("-" * len(title))
  print()

  # Calculate folder paths
  data_folderpath  = config.output_paths.spice_kernels_folderpath.parent
  input_folderpath = data_folderpath.parent / 'input'

  # Progress subsection
  print("  Progress")
  print("    Calculate data folder path")
  print()

  # Summary subsection
  print("  Summary")
  print(f"    Output Folderpath          : {config.output_paths.base_folderpath.parent}")
  print(f"      Timestamp Folderpath     : <output_folderpath>/{config.output_paths.base_folderpath.name}")
  print(f"      Figures Folderpath       : <output_folderpath>/{config.output_paths.base_folderpath.name}/{config.output_paths.figures_folderpath.name}")
  print(f"      Files Folderpath         : <output_folderpath>/{config.output_paths.base_folderpath.name}/{config.output_paths.logs_folderpath.name}")
  print(f"      Log Filepath             : <output_folderpath>/{config.output_paths.base_folderpath.name}/{config.output_paths.logs_folderpath.name}/{config.output_paths.log_filepath.name}")
  print(f"    Data Folderpath            : {data_folderpath}")
  print(f"      SPICE Kernels Folderpath : <data_folderpath>/{config.output_paths.spice_kernels_folderpath.relative_to(data_folderpath)}")
  print(f"      LSK Filepath             : <data_folderpath>/{config.output_paths.lsk_filepath.relative_to(data_folderpath)}")
  print(f"      Gravity Folderpath       : <data_folderpath>/{config.gravity.folderpath.relative_to(data_folderpath)}")
  print(f"      JPL Horizons Folderpath  : <data_folderpath>/{config.output_paths.jpl_horizons_folderpath.relative_to(data_folderpath)}")
  print(f"      TLEs Folderpath          : <data_folderpath>/{config.output_paths.tles_folderpath.relative_to(data_folderpath)}")
  print(f"    Input Folderpath           : {input_folderpath}")
  print(f"      State Vectors Folderpath : <input_folderpath>/{config.output_paths.state_vectors_folderpath.relative_to(input_folderpath)}")


def print_configuration(
  config : SimulationConfig,
) -> None:
  """
  Print the complete configuration (input arguments and paths).
  
  Input:
  ------
    config : SimulationConfig
      Configuration object containing all input and path attributes.
      
  Output:
  -------
    None
  """
  print_input_configuration(
    initial_state_source   = config.initial_state.source,
    initial_state_norad_id = config.initial_state.norad_id,
    initial_state_filename = config.initial_state.filename,
    desired_timespan       = [config.time_o_dt, config.time_f_dt],
    include_drag           = config.include_drag,
    compare_tle            = config.comparison.compare_tle,
    compare_jpl_horizons   = config.comparison.compare_jpl_horizons,
    third_bodies_list      = config.gravity.third_body.bodies,
    gravity_harmonics_list = config.gravity.spherical_harmonics.coefficients,
    two_body_gravity_model = config.gravity,
    include_srp            = config.include_srp,
    include_relativity     = config.gravity.relativity.enabled,
    auto_download          = config.auto_download,
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
  include_relativity             : bool           = False,
  include_solid_tides            : bool           = False,
  include_ocean_tides            : bool           = False,
  auto_download                  : bool           = False,
  initial_state_source           : str            = 'jpl_horizons',
  gravity_harmonics_degree_order : Optional[list] = None,
  gravity_model_filename         : Optional[str]  = None,
  atol                           : float          = 1e-15,
  rtol                           : float          = 1e-12,
  include_tracker_skyplots       : bool           = False,
  tracker_filename               : Optional[str]  = None,
  tracker_filepath               : Optional[str]  = None,
  include_tracker_on_body        : bool           = False,
) -> SimulationConfig:
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
    atol : float
      Absolute tolerance for numerical integration.
    rtol : float
      Relative tolerance for numerical integration.
  
  Output:
  -------
    config : SimulationConfig
      Configuration object containing parsed and calculated propagation parameters.
  
  Raises:
  -------
    ValueError
      If NORAD ID is not supported.
      If timespan is not provided.
  """

  # Validate required arguments
  if timespan_dt is None:
    raise ValueError(
      "timespan is required but was not provided. "
      "Please provide --timespan via command line or 'timespan' in config file."
    )

  if not isinstance(timespan_dt, (list, tuple)) or len(timespan_dt) != 2:
    raise ValueError(
      f"timespan must be a list or tuple of exactly 2 datetime objects, got: {type(timespan_dt).__name__}"
    )

  # Normalize inputs
  initial_state_source, gravity_harmonics_list = normalize_input(
    initial_state_source,
    gravity_harmonics,
  )
  
  # Validate: skyplot arguments
  if (tracker_filename is not None or tracker_filepath is not None) and not include_tracker_skyplots:
    raise ValueError("--tracker-filename or --tracker-filepath requires --include-tracker-skyplots to be set.")
  
  # Validate: cannot use both gravity harmonics options
  has_coefficients = gravity_harmonics_list is not None and len(gravity_harmonics_list) > 0
  has_degree_order = gravity_harmonics_degree_order is not None
  
  if has_coefficients and has_degree_order:
    raise ValueError(
      "Cannot use both --gravity-harmonics-coefficients and --gravity-harmonics-degree-order. "
      "Please use only one method to specify gravity harmonics."
    )
  
  # Handle gravity harmonics logic
  # include_gravity_harmonics is True if EITHER explicit coefficients OR degree/order is specified
  include_gravity_harmonics = has_coefficients or has_degree_order

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
  delta_time_epoch = (time_f_dt - time_o_dt).total_seconds()
  
  # Set up foldernames, folderpaths, filenames, and filepaths
  paths = setup_paths(
    initial_state_source      = initial_state_source,
    initial_state_filename    = initial_state_filename,
    gravity_model_filename    = gravity_model_filename,
    include_tracker_skyplots  = include_tracker_skyplots,
    tracker_filename          = tracker_filename,
    tracker_filepath          = tracker_filepath,
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
    object_name_display = sv_data.get('name', 'CustomObject')
    
    # Sanitize object name for filenames
    object_name = sanitize_filename(object_name_display)
    
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

    # Get object name (original for display, sanitized for filenames)
    object_name_display = obj_props.get('name', 'object_name')
    object_name = sanitize_filename(object_name_display)
    
    # Update paths with correct name
    paths = setup_paths(
      include_tracker_skyplots = include_tracker_skyplots,
      tracker_filename         = tracker_filename,
      tracker_filepath         = tracker_filepath,
    )
  
  # Create SpacecraftProperties object
  spacecraft = SpacecraftProperties(
    mass     = obj_props['mass__kg'],
    drag     = DragConfig(
      enabled = include_drag,
      cd      = obj_props['drag']['coeff'],
      area    = obj_props['drag']['area__m2']
    ),
    srp      = SRPConfig(
      enabled = include_srp,
      cr      = obj_props['srp']['coeff'],
      area    = obj_props['srp']['area__m2']
    ),
    norad_id = initial_state_norad_id,
    name     = object_name
  )

  # Create PropagationConfig object
  propagation_config = PropagationConfig(
    time_o_dt = time_o_dt,
    time_f_dt = time_f_dt,
    atol      = atol,
    rtol      = rtol,
  )

  # Create GravityModelConfig
  gravity_model = GravityModelConfig(
    gp                  = SOLARSYSTEMCONSTANTS.EARTH.GP,
    folderpath          = paths['gravity_model_folderpath'],
    filename            = paths['gravity_model_filename'],
    spherical_harmonics = SphericalHarmonicsConfig(
      degree       = gravity_harmonics_degree if gravity_harmonics_degree is not None else 0,
      order        = gravity_harmonics_order  if gravity_harmonics_order  is not None else 0,
      coefficients = gravity_harmonics_list,
    ),
    third_body = ThirdBodyConfig(
      enabled = include_third_body,
      bodies  = third_bodies_list,
    ),
    relativity = RelativityConfig(
      enabled = include_relativity,
    ),
    solid_tides = SolidEarthTidesConfig(
      enabled = include_solid_tides,
    ),
    ocean_tides = OceanTidesConfig(
      enabled = include_ocean_tides,
    ),
  )

  # Create InitialStateConfig
  initial_state_config = InitialStateConfig(
    source   = initial_state_source,
    norad_id = initial_state_norad_id,
    filename = initial_state_filename,
  )
  
  # Create ComparisonConfig
  comparison_config = ComparisonConfig(
    compare_jpl_horizons = compare_jpl_horizons,
    compare_tle          = compare_tle,
  )
  
  # Update output_paths with all path information
  output_paths = paths['output_paths']
  output_paths.spice_kernels_folderpath = paths['spice_kernels_folderpath']
  output_paths.lsk_filepath             = paths['lsk_filepath']
  output_paths.jpl_horizons_folderpath  = paths['jpl_horizons_folderpath']
  output_paths.tles_folderpath          = paths['tles_folderpath']
  output_paths.state_vectors_folderpath = paths['state_vectors_folderpath']

  return SimulationConfig(
    initial_state       = initial_state_config,
    time_o_dt           = time_o_dt,
    time_f_dt           = time_f_dt,
    spacecraft          = spacecraft,
    gravity             = gravity_model,
    comparison          = comparison_config,
    output_paths        = output_paths,
    object_name         = object_name,
    object_name_display = object_name_display,
    auto_download       = auto_download,
    propagation_config  = propagation_config,
  )


def setup_paths(
  initial_state_source     : Optional[str]  = None,
  initial_state_filename   : Optional[str]  = None,
  gravity_model_filename   : Optional[str]  = None,
  include_tracker_skyplots : bool           = False,
  tracker_filename         : Optional[str]  = None,
  tracker_filepath         : Optional[str]  = None,
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
    include_tracker_skyplots : bool
      Flag to enable skyplot generation.
    tracker_filename : str | None
      Tracker YAML filename (assumes input/trackers/ folder).
    tracker_filepath : str | None
      Absolute path to tracker YAML file.
      
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

  # State Vectors folderpath (in input/ folder)
  input_folderpath = project_root / 'input'
  state_vectors_folderpath = input_folderpath / 'state_vectors'
  
  custom_state_vector_filepath = None
  if initial_state_source == 'custom_state_vector':
    if not initial_state_filename:
      raise ValueError("Argument --initial-state-filename is required when using a custom state vector.")
    
    custom_state_vector_filepath = state_vectors_folderpath / initial_state_filename
    if not custom_state_vector_filepath.exists():
      raise FileNotFoundError(f"Custom state vector file not found: {custom_state_vector_filepath}")

  # Horizons ephemeris folder (loader will search for compatible files)
  jpl_horizons_folderpath = data_folderpath / 'ephems'
  
  # Tracker filepath - determine based on skyplot settings
  trackers_folderpath = input_folderpath / 'trackers'
  resolved_tracker_filepath = None
  
  if include_tracker_skyplots:
    if tracker_filepath is not None:
      # Absolute path provided
      resolved_tracker_filepath = Path(tracker_filepath)
    elif tracker_filename is not None:
      # Relative filename provided - look in input/trackers/
      resolved_tracker_filepath = trackers_folderpath / tracker_filename
    else:
      # No path specified - find first .yaml in input/trackers/
      if trackers_folderpath.exists():
        yaml_files = list(trackers_folderpath.glob('*.yaml'))
        if yaml_files:
          resolved_tracker_filepath = yaml_files[0]
        else:
          print(f"[WARNING] --include-tracker-skyplots enabled but no .yaml files found in {trackers_folderpath}")
      else:
        print(f"[WARNING] --include-tracker-skyplots enabled but trackers folder not found: {trackers_folderpath}")
  
  # Define output folderpath
  timestamp_str        = datetime.now().strftime("%Y%m%d_%H%M%S")
  output_folderpath    = project_root / 'output'
  timestamp_folderpath = output_folderpath / timestamp_str
  
  # Initialize OutputPaths
  output_paths = OutputPaths(
    base_folderpath  = timestamp_folderpath,
    logs_folderpath  = timestamp_folderpath / 'files',
    log_filepath     = timestamp_folderpath / 'files' / 'output.log',
    data_folderpath  = data_folderpath,
    tracker_filepath = resolved_tracker_filepath,
  )
  output_paths.ensure_directories()

  return {
    'output_paths'                 : output_paths,
    'output_folderpath'            : output_folderpath,
    'timestamp_folderpath'         : timestamp_folderpath,
    'figures_folderpath'           : output_paths.figures_folderpath,
    'files_folderpath'             : output_paths.logs_folderpath,
    'log_filepath'                 : output_paths.log_filepath,
    'spice_kernels_folderpath'     : spice_kernels_folderpath,
    'gravity_model_folderpath'     : gravity_model_folderpath,
    'gravity_model_filename'       : gravity_model_filename,
    'jpl_horizons_folderpath'      : jpl_horizons_folderpath,
    'tles_folderpath'              : tles_folderpath,
    'state_vectors_folderpath'     : state_vectors_folderpath,
    'custom_state_vector_filepath' : custom_state_vector_filepath,
    'lsk_filepath'                 : lsk_filepath,
    'tracker_filepath'             : tracker_filepath,
  }


def extract_tle_to_config(
  config               : SimulationConfig,
  result_celestrak_tle : Optional[TLEData],
) -> None:
  """
  Extract TLE data from TLEData object and store on config object.
  
  Input:
  ------
    config : SimulationConfig
      Configuration object to store TLE data on.
    result_celestrak_tle : TLEData | None
      TLEData object from get_celestrak_tle.
      
  Output:
  -------
    None
  """
  if result_celestrak_tle is not None:
    config.tle_line_1   = result_celestrak_tle.line_1
    config.tle_line_2   = result_celestrak_tle.line_2
    config.tle_epoch_dt = result_celestrak_tle.epoch_dt
