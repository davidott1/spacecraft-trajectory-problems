import sys
import argparse
import yaml
from pathlib import Path
from src.utility.time_helper import parse_time


def load_config_file(config_path: str) -> dict:
  """
  Load configuration from a YAML file.

  Input:
  ------
    config_path : str
      Path to the YAML config file. Can be:
      - Just filename (looks in data/configs/)
      - Relative or absolute path

  Output:
  -------
    config : dict
      Configuration dictionary loaded from YAML.
  """
  path = Path(config_path)

  # If just a filename, look in data/configs/
  if not path.is_absolute() and path.parent == Path('.'):
    path = Path(__file__).parent.parent.parent / 'data' / 'configs' / config_path

  if not path.exists():
    raise FileNotFoundError(f"Config file not found: {path}")

  with open(path, 'r') as f:
    config = yaml.safe_load(f)

  return config if config else {}


def merge_config_with_args(config: dict, args: argparse.Namespace) -> argparse.Namespace:
  """
  Merge configuration file with command-line arguments.
  CLI arguments take precedence over config file values.

  Input:
  ------
    config : dict
      Configuration dictionary from YAML file.
    args : argparse.Namespace
      Parsed command-line arguments.

  Output:
  -------
    args : argparse.Namespace
      Merged arguments with CLI overriding config file.
  """
  # Map config keys to argparse attribute names
  config_mapping = {
    'initial_state_source': 'initial_state_source',
    'initial_state_norad_id': 'initial_state_norad_id',
    'initial_state_filename': 'initial_state_filename',
    'timespan': 'timespan',
    'third_bodies': 'third_bodies',
    'gravity_harmonics_coefficients': 'gravity_harmonics_coefficients',
    'gravity_harmonics_degree_order': 'gravity_harmonics_degree_order',
    'gravity_harmonics_filename': 'gravity_model_filename',
    'include_drag': 'include_drag',
    'include_srp': 'include_srp',
    'compare_tle': 'compare_tle',
    'compare_jpl_horizons': 'compare_jpl_horizons',
    'auto_download': 'auto_download',
    'atol': 'atol',
    'rtol': 'rtol',
    'include_tracker_skyplots': 'include_tracker_skyplots',
    'tracker_filename': 'tracker_filename',
    'tracker_filepath': 'tracker_filepath',
    'include_tracker_on_body': 'include_tracker_on_body',
  }

  # Process each config key
  for config_key, arg_name in config_mapping.items():
    if config_key not in config:
      continue

    config_value = config[config_key]

    # Get the current argument value
    current_value = getattr(args, arg_name, None)

    # Special handling for different argument types
    if arg_name == 'timespan':
      if current_value is None:
        from datetime import datetime

        # Handle dict format: {start: ..., end: ...}
        if isinstance(config_value, dict):
          start_val = config_value.get('start')
          end_val = config_value.get('end')
          if start_val and end_val:
            # Handle both string and datetime objects from YAML
            if isinstance(start_val, datetime):
              start = start_val
            else:
              start = parse_time(start_val)

            if isinstance(end_val, datetime):
              end = end_val
            else:
              end = parse_time(end_val)

            setattr(args, arg_name, [start, end])

        # Handle string format: "start end"
        elif isinstance(config_value, str):
          time_parts = config_value.split()
          if len(time_parts) == 2:
            start = parse_time(time_parts[0])
            end = parse_time(time_parts[1])
            setattr(args, arg_name, [start, end])

        # Handle list format: [start, end]
        elif isinstance(config_value, (list, tuple)) and len(config_value) == 2:
          if isinstance(config_value[0], datetime):
            start = config_value[0]
          else:
            start = parse_time(config_value[0])

          if isinstance(config_value[1], datetime):
            end = config_value[1]
          else:
            end = parse_time(config_value[1])

          setattr(args, arg_name, [start, end])

    elif arg_name in ['initial_state_source', 'initial_state_norad_id', 'initial_state_filename',
                      'gravity_model_filename', 'tracker_filename', 'tracker_filepath']:
      # String arguments - only override if CLI didn't provide a value
      if current_value is None or (arg_name == 'initial_state_source' and current_value == 'horizons'):
        # Convert to string (YAML may parse NORAD ID as int)
        setattr(args, arg_name, str(config_value) if config_value is not None else None)

    elif arg_name in ['third_bodies', 'gravity_harmonics_coefficients', 'gravity_harmonics_degree_order']:
      # List arguments - only override if CLI didn't provide values
      if current_value is None or (isinstance(current_value, list) and len(current_value) == 0):
        # Handle string "val1, val2" format
        if isinstance(config_value, str):
          # Parse comma-separated string
          parsed_values = [x.strip() for x in config_value.split(',')]
          # Convert to appropriate type
          if arg_name == 'gravity_harmonics_degree_order':
            # Convert to integers for degree/order
            parsed_values = [int(x) for x in parsed_values]
          setattr(args, arg_name, parsed_values)
        else:
          setattr(args, arg_name, config_value)

    elif arg_name in ['include_drag', 'include_srp', 'compare_tle', 'compare_jpl_horizons',
                      'auto_download', 'include_tracker_skyplots', 'include_tracker_on_body']:
      # Boolean flags - only override if CLI kept the default False
      if not current_value:
        setattr(args, arg_name, config_value)

    elif arg_name in ['atol', 'rtol']:
      # Float arguments - use config if CLI kept defaults
      default_vals = {'atol': 1e-15, 'rtol': 1e-12}
      if current_value == default_vals[arg_name]:
        setattr(args, arg_name, config_value)

  return args


def parse_command_line_arguments(
) -> argparse.Namespace:
  """
  Parse command-line arguments for the orbit propagator.

  Input:
  ------
    None (reads from sys.argv)

  Output:
  -------
    args : argparse.Namespace
      Parsed command-line arguments.
  """
  parser = argparse.ArgumentParser(
    description = 'High-fidelity orbit propagator',
    formatter_class = argparse.RawDescriptionHelpFormatter,
  )

  # If no arguments provided, print help and exit
  if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

  # Configuration file argument
  parser.add_argument(
    '--config',
    dest    = 'config',
    type    = str,
    default = None,
    help    = 'Path to YAML configuration file. If just a filename, looks in data/configs/. CLI arguments override config file values.',
  )

  # Initial state arguments
  parser.add_argument(
    '--initial-state-source',
    dest    = 'initial_state_source',
    type    = str,
    choices = [
      'horizons', 'jpl-horizons', 'jpl_horizons', 'jpl horizons',
      'tle', 
      'sv', 'custom-sv', 'custom-state-vector', 'state-vector',
    ],
    default = 'horizons',
    help    = "Source for initial state vector (default: horizons).",
  )
  parser.add_argument(
    '--initial-state-norad-id',
    type=str,
    required = False,
    help     = 'NORAD ID for the initial state object. Required if initial state source is jpl-horizons or tle.'
  )
  parser.add_argument(
    '--initial-state-filename',
    type     = str,
    required = False,
    help     = 'Filename of the custom state vector .yaml file in data/state_vectors. Required if initial state source is custom-state-vector.'
  )
  
  # Time arguments
  parser.add_argument(
    '--timespan',
    dest     = 'timespan',
    type     = parse_time,
    nargs    = 2,
    metavar  = ('TIME_START', 'TIME_END'),
    required = False,
    default  = None,
    help     = "Start and end time for propagation in ISO format (e.g., '2025-10-01T00:00:00 2025-10-02T00:00:00'). Required unless provided in config file.",
  )
  
  parser.add_argument(
    '--third-bodies',
    '--include-third-bodies',
    dest    = 'third_bodies',
    nargs    = '+', # accepts 1 or more args.
    type     = str,
    choices  = ['SUN', 'MOON', 'MERCURY', 'VENUS', 'MARS', 'JUPITER', 'SATURN', 'URANUS', 'NEPTUNE', 'PLUTO',
                'sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune', 'pluto'],
    default  = None,
    help     = 'Enable third-body gravity perturbations. Options: SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, URANUS, NEPTUNE, PLUTO',
  )
  parser.add_argument(
    '--gravity-harmonics-coefficients',
    '--gravity-harmonics-coeffs',
    dest    = 'gravity_harmonics_coefficients',
    type    = str,
    nargs   = '*',
    default = [],
    help    = 'Specific gravity harmonic coefficients to include (e.g., J2 J3 J4 C22 S22). Default: none.',
  )
  parser.add_argument(
    '--gravity-harmonics-degree-order',
    dest    = 'gravity_harmonics_degree_order',
    type    = int,
    nargs   = 2,
    metavar = ('DEGREE', 'ORDER'),
    default = None,
    help    = 'Maximum degree and order for spherical harmonics (e.g. 10 10).',
  )
  parser.add_argument(
    '--gravity-harmonics-filename',
    dest    = 'gravity_model_filename',
    type    = str,
    default = None,
    help    = 'Filename of the gravity field coefficient file (e.g. EGM2008.gfc).',
  )
  parser.add_argument(
    '--include-drag',
    '--drag',
    dest    = 'include_drag',
    action  = 'store_true',
    default = False,
    help    = "Enable Atmospheric Drag (disabled by default).",
  )
  parser.add_argument(
    '--compare-tle',
    dest    = 'compare_tle',
    action  = 'store_true',
    default = False,
    help    = "Enable comparison with TLE/SGP4 propagation (disabled by default).",
  )
  parser.add_argument(
    '--compare-horizons',
    '--compare-jpl-horizons',
    dest    = 'compare_jpl_horizons',
    action  = 'store_true',
    default = False,
    help    = "Enable comparison with JPL Horizons ephemeris (disabled by default).",
  )
  parser.add_argument(
    '--include-srp',
    '--srp',
    dest    = 'include_srp',
    action  = 'store_true',
    default = False,
    help    = "Enable Solar Radiation Pressure (disabled by default).",
  )
  
  parser.add_argument(
    '--auto-download',
    dest    = 'auto_download',
    action  = 'store_true',
    default = False,
    help    = "Automatically download missing data (Horizons/TLE) without prompting.",
  )
  
  parser.add_argument(
    '--atol',
    dest    = 'atol',
    type    = float,
    default = 1e-15,
    help    = "Absolute tolerance for numerical integration (default: 1e-15).",
  )

  parser.add_argument(
    '--rtol',
    dest    = 'rtol',
    type    = float,
    default = 1e-12,
    help    = "Relative tolerance for numerical integration (default: 1e-12).",
  )
  
  parser.add_argument(
    '--include-tracker-skyplots',
    dest    = 'include_tracker_skyplots',
    action  = 'store_true',
    default = False,
    help    = 'Enable skyplot generation (disabled by default). Uses tracker file from data/trackers/.',
  )
  
  parser.add_argument(
    '--tracker-filename',
    dest     = 'tracker_filename',
    type     = str,
    required = False,
    help     = 'Tracker station YAML filename (assumes data/trackers/ folder). E.g., trackers.yaml',
  )
  
  parser.add_argument(
    '--tracker-filepath',
    dest     = 'tracker_filepath',
    type     = str,
    required = False,
    help     = 'Absolute path to tracker station YAML file.',
  )

  parser.add_argument(
    '--include-tracker-on-body',
    dest    = 'include_tracker_on_body',
    action  = 'store_true',
    default = False,
    help    = 'Show tracker location on ground track and 3D body-fixed plots (disabled by default).',
  )

  # Parse arguments
  args = parser.parse_args()

  # Load and merge config file if provided
  if args.config:
    config = load_config_file(args.config)
    args = merge_config_with_args(config, args)

  return args
