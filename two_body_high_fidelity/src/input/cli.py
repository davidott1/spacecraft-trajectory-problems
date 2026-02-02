import sys
import re
import argparse
import yaml
from pathlib import Path


class FlexibleBooleanAction(argparse.Action):
  """
  Custom argparse action that provides flexible boolean argument handling.

  Supports all these variations:
    --flag              -> True
    --flag True         -> True
    --flag true         -> True
    --flag False        -> False
    --flag false        -> False
    --enable-flag       -> True
    --enable-flag True  -> True
    --enable-flag False -> False
    --disable-flag      -> False (inverts the value)
    --disable-flag True -> False (inverts the value)
    --disable-flag False -> True (inverts the value)

  Usage:
    parser.add_argument('--include-srp', '--srp', '--enable-srp',
                        dest='include_srp', action=FlexibleBooleanAction, default=False)
    parser.add_argument('--disable-srp', dest='include_srp', action=FlexibleBooleanAction,
                        default=argparse.SUPPRESS, invert=True)
  """
  def __init__(self, option_strings, dest, nargs=None, const=None, default=None,
               type=None, choices=None, required=False, help=None, metavar=None, invert=False):
    # Store whether this is a disable flag (inverts the logic)
    self.invert = invert

    # Support both no-argument and single-argument forms
    if nargs is None:
      nargs = '?'

    # Default const:
    # - For disable flags (invert=True): const=True, then inverted to False
    # - For enable flags (invert=False): const=True
    if const is None:
      const = True

    super().__init__(
      option_strings=option_strings,
      dest=dest,
      nargs=nargs,
      const=const,
      default=default,
      type=type,
      choices=choices,
      required=required,
      help=help,
      metavar=metavar
    )

  def __call__(self, parser, namespace, values, option_string=None):
    """Process the argument value and set the destination attribute."""
    # If no value provided, use const (True for enable, False for disable)
    if values is None:
      result = self.const
    else:
      # Parse boolean from string
      if isinstance(values, str):
        result = values.lower() in ('true', '1', 'yes', 'y', 't')
      else:
        result = bool(values)

    # Invert if this is a disable flag
    if self.invert:
      result = not result

    setattr(namespace, self.dest, result)
def parse_time(time_str: str):
  """
  Parse time string to datetime object.
  Always returns a timezone-naive datetime (assumed UTC).
  """
  from dateutil import parser
  dt = parser.parse(time_str)
  # Strip timezone info to maintain consistency with internal naive-UTC convention
  return dt.replace(tzinfo=None)


def load_config_file(config_path: str) -> dict:
  """
  Load configuration from a YAML file.

  Input:
  ------
    config_path : str
      Path to the YAML config file. Can be:
      - Just filename (looks in input/configs/)
      - Relative or absolute path

  Output:
  -------
    config : dict
      Configuration dictionary loaded from YAML.
  """
  path = Path(config_path)

  # If just a filename, look in input/configs/
  if not path.is_absolute() and path.parent == Path('.'):
    path = Path(__file__).parent.parent.parent / 'input' / 'configs' / config_path

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
  # Track which arguments were explicitly set on the command line
  # by checking which appear in sys.argv
  import sys
  cli_args_set = set()
  for arg in sys.argv:
    if arg.startswith('--'):
      # Extract the flag name (handle both --flag and --flag=value formats)
      flag = arg.split('=')[0]
      cli_args_set.add(flag)

  # Map config keys to argparse attribute names and their CLI flags
  config_mapping = {
    'initial_state_source': ('initial_state_source', ['--initial-state-source']),
    'initial_state_norad_id': ('initial_state_norad_id', ['--initial-state-norad-id']),
    'initial_state_filename': ('initial_state_filename', ['--initial-state-filename']),
    'timespan': ('timespan', ['--timespan']),
    'third_bodies': ('third_bodies', ['--third-bodies', '--include-third-bodies']),
    'gravity_harmonics_coefficients': ('gravity_harmonics_coefficients', ['--gravity-harmonics-coefficients', '--gravity-harmonics-coeffs']),
    'gravity_harmonics_degree_order': ('gravity_harmonics_degree_order', ['--gravity-harmonics-degree-order']),
    'gravity_harmonics_filename': ('gravity_model_filename', ['--gravity-harmonics-filename']),
    'include_drag': ('include_drag', ['--include-drag', '--drag', '--enable-drag', '--disable-drag']),
    'include_srp': ('include_srp', ['--include-srp', '--srp', '--enable-srp', '--disable-srp']),
    'include_relativity': ('include_relativity', ['--include-relativity', '--relativity', '--enable-relativity', '--disable-relativity']),
    'include_solid_tides': ('include_solid_tides', ['--include-solid-tides', '--solid-tides', '--enable-solid-tides', '--disable-solid-tides']),
    'include_ocean_tides': ('include_ocean_tides', ['--include-ocean-tides', '--ocean-tides', '--enable-ocean-tides', '--disable-ocean-tides']),
    'compare_tle': ('compare_tle', ['--compare-tle', '--enable-compare-tle', '--disable-compare-tle']),
    'compare_jpl_horizons': ('compare_jpl_horizons', ['--compare-horizons', '--compare-jpl-horizons', '--enable-compare-horizons', '--enable-compare-jpl-horizons', '--disable-compare-horizons', '--disable-compare-jpl-horizons']),
    'auto_download': ('auto_download', ['--auto-download', '--enable-auto-download', '--disable-auto-download']),
    'atol': ('atol', ['--atol']),
    'rtol': ('rtol', ['--rtol']),
    'include_tracker_skyplots': ('include_tracker_skyplots', ['--include-tracker-skyplots', '--enable-tracker-skyplots', '--disable-tracker-skyplots']),
    'tracker_filename': ('tracker_filename', ['--tracker-filename']),
    'tracker_filepath': ('tracker_filepath', ['--tracker-filepath']),
    'include_tracker_on_body': ('include_tracker_on_body', ['--include-tracker-on-body', '--enable-tracker-on-body', '--disable-tracker-on-body']),
    'include_orbit_determination': ('include_orbit_determination', ['--include-orbit-determination', '--enable-orbit-determination', '--disable-orbit-determination']),
    'maneuver_filename': ('maneuver_filename', ['--maneuver-filename']),
    'process_noise_pos': ('process_noise_pos', ['--process-noise-pos']),
    'process_noise_vel': ('process_noise_vel', ['--process-noise-vel']),
    'process_noise_pos__m_per_s': ('process_noise_pos', ['--process-noise-pos']),
    'process_noise_vel__m_per_s2': ('process_noise_vel', ['--process-noise-vel']),
    'use_approx_jacobian': ('use_approx_jacobian', ['--use-approx-jacobian', '--approx-jacobian']),
    'use_analytic_jacobian': ('use_analytic_jacobian', ['--use-analytic-jacobian', '--analytic-jacobian']),
    'jacobian_approx_eps': ('jacobian_approx_eps', ['--jacobian-approx-eps']),
    'make_meas_from': ('make_meas_from', ['--make-meas-from']),
  }

  # Process each config key
  for config_key, (arg_name, cli_flags) in config_mapping.items():
    if config_key not in config:
      continue

    config_value = config[config_key]

    # Check if any of the CLI flags for this argument were used
    cli_was_used = any(flag in cli_args_set for flag in cli_flags)

    # If CLI was used, skip applying config value (CLI takes precedence)
    if cli_was_used:
      continue

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
                      'gravity_model_filename', 'tracker_filename', 'tracker_filepath', 'maneuver_filename']:
      # String arguments - only override if CLI didn't provide a value
      if current_value is None or (arg_name == 'initial_state_source' and current_value == 'horizons'):
        # Convert to string (YAML may parse NORAD ID as int)
        setattr(args, arg_name, str(config_value) if config_value is not None else None)

    elif arg_name in ['third_bodies', 'gravity_harmonics_coefficients', 'gravity_harmonics_degree_order']:
      # List arguments - only override if CLI didn't provide values
      if current_value is None or (isinstance(current_value, list) and len(current_value) == 0):
        # Handle string "val1, val2" or "val1 val2" format
        if isinstance(config_value, str):
          # Parse comma-separated or space-separated string
          parsed_values = [x.strip() for x in re.split(r'[,\s]+', config_value) if x.strip()]
          # Convert to appropriate type
          if arg_name == 'gravity_harmonics_degree_order':
            # Convert to integers for degree/order
            parsed_values = [int(x) for x in parsed_values]
          setattr(args, arg_name, parsed_values)
        else:
          setattr(args, arg_name, config_value)

    elif arg_name in ['include_drag', 'include_srp', 'include_relativity', 'include_solid_tides', 'include_ocean_tides', 'compare_tle', 'compare_jpl_horizons',
              'auto_download', 'include_tracker_skyplots', 'include_tracker_on_body',
              'include_orbit_determination', 'use_approx_jacobian', 'use_analytic_jacobian']:
      # Boolean flags - only override if CLI kept the default False
      # With FlexibleBooleanAction, check if the value is still False (default)
      if current_value is False:
        setattr(args, arg_name, config_value)

    elif arg_name in ['atol', 'rtol', 'jacobian_approx_eps']:
      # Float arguments - use config if CLI kept defaults
      default_vals = {'atol': 1e-15, 'rtol': 1e-12}
      if arg_name == 'jacobian_approx_eps':
        if current_value is None:
          setattr(args, arg_name, float(config_value))
      elif current_value == default_vals[arg_name]:
        setattr(args, arg_name, config_value)

    elif arg_name in ['process_noise_pos', 'process_noise_vel']:
      # Float arguments for process noise - use config if CLI didn't provide a value
      if current_value is None:
        setattr(args, arg_name, float(config_value))

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
    '--config-filename',
    '--config-filepath',
    dest    = 'config',
    type    = str,
    default = None,
    help    = 'Path to YAML configuration file. If just a filename, looks in input/configs/. CLI arguments override config file values.',
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
    help     = 'Filename of the custom state vector .yaml file in input/state_vectors. Required if initial state source is custom-state-vector.'
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
  
  def parse_third_bodies(value):
    """Parse third-bodies argument, handling None/empty to disable."""
    if value.lower() in ('none', 'null', 'disable', 'disabled', 'false'):
      return None
    valid_bodies = ['SUN', 'MOON', 'MERCURY', 'VENUS', 'MARS', 'JUPITER', 'SATURN', 'URANUS', 'NEPTUNE', 'PLUTO']
    if value.upper() not in valid_bodies:
      raise argparse.ArgumentTypeError(f"Invalid third body: {value}. Choose from {', '.join(valid_bodies)} or 'None' to disable.")
    return value

  parser.add_argument(
    '--third-bodies',
    '--include-third-bodies',
    dest    = 'third_bodies',
    nargs    = '+', # accepts 1 or more args.
    type     = parse_third_bodies,
    default  = None,
    help     = 'Enable third-body gravity perturbations. Options: SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, URANUS, NEPTUNE, PLUTO. Use "None" to explicitly disable.',
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
    '--enable-drag',
    dest    = 'include_drag',
    action  = FlexibleBooleanAction,
    default = False,
    help    = "Enable Atmospheric Drag (disabled by default). Accepts True/False or use --disable-drag to disable.",
  )
  parser.add_argument(
    '--disable-drag',
    dest    = 'include_drag',
    action  = FlexibleBooleanAction,
    default = argparse.SUPPRESS,
    invert  = True,
    help    = argparse.SUPPRESS,  # Hidden from help
  )
  parser.add_argument(
    '--compare-tle',
    '--enable-compare-tle',
    dest    = 'compare_tle',
    action  = FlexibleBooleanAction,
    default = False,
    help    = "Enable comparison with TLE/SGP4 propagation (disabled by default). Accepts True/False or use --disable-compare-tle to disable.",
  )
  parser.add_argument(
    '--disable-compare-tle',
    dest    = 'compare_tle',
    action  = FlexibleBooleanAction,
    default = argparse.SUPPRESS,
    invert  = True,
    help    = argparse.SUPPRESS,
  )
  parser.add_argument(
    '--compare-horizons',
    '--compare-jpl-horizons',
    '--enable-compare-horizons',
    '--enable-compare-jpl-horizons',
    dest    = 'compare_jpl_horizons',
    action  = FlexibleBooleanAction,
    default = False,
    help    = "Enable comparison with JPL Horizons ephemeris (disabled by default). Accepts True/False or use --disable-compare-horizons to disable.",
  )
  parser.add_argument(
    '--disable-compare-horizons',
    '--disable-compare-jpl-horizons',
    dest    = 'compare_jpl_horizons',
    action  = FlexibleBooleanAction,
    default = argparse.SUPPRESS,
    invert  = True,
    help    = argparse.SUPPRESS,
  )
  parser.add_argument(
    '--include-srp',
    '--srp',
    '--enable-srp',
    dest    = 'include_srp',
    action  = FlexibleBooleanAction,
    default = False,
    help    = "Enable Solar Radiation Pressure (disabled by default). Accepts True/False or use --disable-srp to disable.",
  )
  parser.add_argument(
    '--disable-srp',
    dest    = 'include_srp',
    action  = FlexibleBooleanAction,
    default = argparse.SUPPRESS,
    invert  = True,
    help    = argparse.SUPPRESS,
  )

  parser.add_argument(
    '--include-relativity',
    '--relativity',
    '--enable-relativity',
    dest    = 'include_relativity',
    action  = FlexibleBooleanAction,
    default = False,
    help    = "Enable general relativistic corrections (Schwarzschild) (disabled by default). Accepts True/False or use --disable-relativity to disable.",
  )
  parser.add_argument(
    '--disable-relativity',
    dest    = 'include_relativity',
    action  = FlexibleBooleanAction,
    default = argparse.SUPPRESS,
    invert  = True,
    help    = argparse.SUPPRESS,
  )

  parser.add_argument(
    '--include-solid-tides',
    '--solid-tides',
    '--enable-solid-tides',
    dest    = 'include_solid_tides',
    action  = FlexibleBooleanAction,
    default = False,
    help    = "Enable solid Earth tide corrections (IERS 2010) (disabled by default). Accepts True/False or use --disable-solid-tides to disable.",
  )
  parser.add_argument(
    '--disable-solid-tides',
    dest    = 'include_solid_tides',
    action  = FlexibleBooleanAction,
    default = argparse.SUPPRESS,
    invert  = True,
    help    = argparse.SUPPRESS,
  )

  parser.add_argument(
    '--include-ocean-tides',
    '--ocean-tides',
    '--enable-ocean-tides',
    dest    = 'include_ocean_tides',
    action  = FlexibleBooleanAction,
    default = False,
    help    = "Enable ocean tide corrections (IERS 2010) (disabled by default). Accepts True/False or use --disable-ocean-tides to disable.",
  )
  parser.add_argument(
    '--disable-ocean-tides',
    dest    = 'include_ocean_tides',
    action  = FlexibleBooleanAction,
    default = argparse.SUPPRESS,
    invert  = True,
    help    = argparse.SUPPRESS,
  )

  parser.add_argument(
    '--auto-download',
    '--enable-auto-download',
    dest    = 'auto_download',
    action  = FlexibleBooleanAction,
    default = False,
    help    = "Automatically download missing data (Horizons/TLE) without prompting. Accepts True/False or use --disable-auto-download to disable.",
  )
  parser.add_argument(
    '--disable-auto-download',
    dest    = 'auto_download',
    action  = FlexibleBooleanAction,
    default = argparse.SUPPRESS,
    invert  = True,
    help    = argparse.SUPPRESS,
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
    '--enable-tracker-skyplots',
    dest    = 'include_tracker_skyplots',
    action  = FlexibleBooleanAction,
    default = False,
    help    = 'Enable skyplot generation (disabled by default). Uses tracker file from input/trackers/. Accepts True/False or use --disable-tracker-skyplots to disable.',
  )
  parser.add_argument(
    '--disable-tracker-skyplots',
    dest    = 'include_tracker_skyplots',
    action  = FlexibleBooleanAction,
    default = argparse.SUPPRESS,
    invert  = True,
    help    = argparse.SUPPRESS,
  )
  
  parser.add_argument(
    '--tracker-filename',
    dest     = 'tracker_filename',
    type     = str,
    required = False,
    help     = 'Tracker station YAML filename (assumes input/trackers/ folder). E.g., trackers.yaml',
  )
  
  parser.add_argument(
    '--tracker-filepath',
    dest     = 'tracker_filepath',
    type     = str,
    required = False,
    help     = 'Absolute path to tracker station YAML file.',
  )

  parser.add_argument(
    '--maneuver-filename',
    dest     = 'maneuver_filename',
    type     = str,
    required = False,
    help     = 'Maneuver YAML filename (assumes input/maneuvers/ folder). E.g., example_hohmann_transfer.yaml',
  )

  parser.add_argument(
    '--include-tracker-on-body',
    '--enable-tracker-on-body',
    dest    = 'include_tracker_on_body',
    action  = FlexibleBooleanAction,
    default = False,
    help    = 'Show tracker location on ground track and 3D body-fixed plots (disabled by default). Accepts True/False or use --disable-tracker-on-body to disable.',
  )
  parser.add_argument(
    '--disable-tracker-on-body',
    dest    = 'include_tracker_on_body',
    action  = FlexibleBooleanAction,
    default = argparse.SUPPRESS,
    invert  = True,
    help    = argparse.SUPPRESS,
  )

  parser.add_argument(
    '--include-orbit-determination',
    '--enable-orbit-determination',
    dest    = 'include_orbit_determination',
    action  = FlexibleBooleanAction,
    default = False,
    help    = 'Process measurements with EKF for orbit determination. When enabled, high-fidelity plots show estimated states instead of propagated states (disabled by default). Accepts True/False or use --disable-orbit-determination to disable.',
  )
  parser.add_argument(
    '--disable-orbit-determination',
    dest    = 'include_orbit_determination',
    action  = FlexibleBooleanAction,
    default = argparse.SUPPRESS,
    invert  = True,
    help    = argparse.SUPPRESS,
  )

  parser.add_argument(
    '--process-noise-pos',
    dest    = 'process_noise_pos',
    type    = float,
    default = None,
    help    = 'Position process noise spectral density [m/s]. Represents rate of position uncertainty growth. Default: 1e-4. EKF uses Q_pos = (value * dt)^2.',
  )

  parser.add_argument(
    '--process-noise-vel',
    dest    = 'process_noise_vel',
    type    = float,
    default = None,
    help    = 'Velocity process noise spectral density [m/s^2]. Represents continuous acceleration noise. Default: 1e-7. EKF uses Q_vel = (value * sqrt(dt))^2.',
  )

  parser.add_argument(
    '--use-approx-jacobian',
    '--approx-jacobian',
    dest    = 'use_approx_jacobian',
    action  = FlexibleBooleanAction,
    default = False,
    help    = 'Use numerical Jacobian for gravity (default if neither flag is set). Accepts True/False or use --disable-approx-jacobian to disable.',
  )
  parser.add_argument(
    '--disable-approx-jacobian',
    dest    = 'use_approx_jacobian',
    action  = FlexibleBooleanAction,
    default = argparse.SUPPRESS,
    invert  = True,
    help    = argparse.SUPPRESS,
  )

  parser.add_argument(
    '--use-analytic-jacobian',
    '--analytic-jacobian',
    dest    = 'use_analytic_jacobian',
    action  = FlexibleBooleanAction,
    default = False,
    help    = 'Use analytic Jacobian (currently J2-only). Accepts True/False or use --disable-analytic-jacobian to disable.',
  )
  parser.add_argument(
    '--disable-analytic-jacobian',
    dest    = 'use_analytic_jacobian',
    action  = FlexibleBooleanAction,
    default = argparse.SUPPRESS,
    invert  = True,
    help    = argparse.SUPPRESS,
  )

  parser.add_argument(
    '--jacobian-approx-eps',
    dest    = 'jacobian_approx_eps',
    type    = float,
    default = None,
    help    = 'Relative step size for numerical Jacobian (default: 1e-6).',
  )

  parser.add_argument(
    '--make-meas-from',
    dest    = 'make_meas_from',
    type    = str,
    choices = ['jpl_ephem', 'model'],
    default = 'jpl_ephem',
    help    = 'Source for measurement simulation: jpl_ephem (default) or model (closed-loop).',
  )

  # Parse arguments
  args = parser.parse_args()

  # Load and merge config file if provided
  if args.config:
    config = load_config_file(args.config)
    args = merge_config_with_args(config, args)

  return args
