from pathlib import Path
from datetime import datetime, timedelta
from types import SimpleNamespace
from sgp4.api import Satrec
from src.utility.loader import load_supported_objects
import argparse
import sys

def parse_time(
  time_str : str,
) -> datetime:
  """
  Parse time string in multiple formats (all assumed UTC)
  
  Input:
  ------
  time_str : str
    Time string in various formats
  
  Output:
  -------
  datetime : Parsed datetime object
  
  Supported formats:
  - YYYY-MM-DD
  - YYYY-MM-DD HH:MM
  - YYYY-MM-DD HH:MM:SS
  - YYYY-MM-DD HH:MM:SS.ssssss
  - YYYY-MM-DDTHH:MM
  - YYYY-MM-DDTHH:MM:SS
  - YYYY-MM-DDTHH:MM:SS.ssssss
  - YYYY-MM-DDTHH:MM:SSZ (with trailing Z)
  """
  # Remove trailing 'Z' if present (ISO 8601 UTC designator)
  time_str = time_str.rstrip('Z')
  
  # Try different formats
  formats = [
    '%Y-%m-%dT%H:%M:%S.%f', # ISO 8601 with microseconds
    '%Y-%m-%dT%H:%M:%S',    # ISO 8601 with seconds: 2025-10-01T00:00:00
    '%Y-%m-%dT%H:%M',       # ISO 8601 without seconds: 2025-10-01T00:00
    '%Y-%m-%d %H:%M:%S.%f', # Space-separated with microseconds
    '%Y-%m-%d %H:%M:%S',    # Space-separated with seconds: 2025-10-01 00:00:00
    '%Y-%m-%d %H:%M',       # Space-separated without seconds: 2025-10-01 00:00
    '%Y-%m-%d',             # Date only: 2025-10-01
  ]
  
  for fmt in formats:
    try:
      return datetime.strptime(time_str, fmt)
    except ValueError:
      continue
  
  raise ValueError(f"Cannot parse time string: {time_str}")


def parse_and_validate_inputs(
  input_object_type      : str,
  norad_id               : str,
  timespan               : list,
  use_spice              : bool = False,
  include_third_body     : bool = False,
  include_zonal_harmonics: bool = False,
  zonal_harmonics_list   : list = None,
  include_srp            : bool = False,
) -> dict:
  """
  Parse and validate input parameters for orbit propagation.
  
  Input:
  ------
    input_object_type : str
      Type of input object (e.g., norad-id).
    norad_id : str
      NORAD catalog ID of the satellite.
    timespan : list
      Start and end time in ISO format (e.g., ['2025-10-01T00:00:00', '2025-10-02T00:00:00']).
    use_spice : bool
      Flag to enable/disable SPICE usage.
    include_third_body : bool
      Flag to enable/disable third-body gravity.
    include_zonal_harmonics : bool
      Flag to enable/disable zonal harmonics.
    zonal_harmonics_list : list
      List of zonal harmonics to include (e.g., ['J2', 'J3']).
    include_srp : bool
      Flag to enable/disable Solar Radiation Pressure.
  
  Output:
  -------
    dict
      A dictionary containing parsed and calculated propagation parameters.
  
  Raises:
  -------
    ValueError
      If NORAD ID is not supported.
  """
  # Normalize input object type
  input_object_type = input_object_type.replace('-', '_').replace(' ', '_')

  # Unpack timespan
  start_time_str = timespan[0]
  end_time_str   = timespan[1]

  # Validate conditional arguments
  if input_object_type == 'norad_id' and not norad_id:
    raise ValueError("NORAD ID is required when input-object-type is 'norad-id'")

  # Enforce dependencies: SRP requires SPICE
  if include_srp:
    use_spice = True

  # Load supported objects
  supported_objects = load_supported_objects()

  # Validate norad id input
  if norad_id not in supported_objects:
    raise ValueError(f"NORAD ID {norad_id} is not supported. Supported IDs: {list(supported_objects.keys())}")

  # Get object properties
  obj_props = supported_objects[norad_id]

  # Extract TLE lines
  tle_line1 = obj_props['tle']['line_1']
  tle_line2 = obj_props['tle']['line_2']

  # Parse TLE epoch
  satellite    = Satrec.twoline2rv(tle_line1, tle_line2)
  tle_epoch_jd = satellite.jdsatepoch + satellite.jdsatepochF
  tle_epoch_dt = datetime(2000, 1, 1, 12, 0, 0) + timedelta(days=tle_epoch_jd - 2451545.0)
  
  # Target propagation start/end times from arguments
  target_start_dt = parse_time(start_time_str)
  target_end_dt   = parse_time(end_time_str)
  delta_time      = (target_end_dt - target_start_dt).total_seconds()
  
  # Integration time bounds (seconds from TLE epoch)
  integ_time_o     = (target_start_dt - tle_epoch_dt).total_seconds()
  integ_time_f     = integ_time_o + delta_time
  delta_integ_time = integ_time_f - integ_time_o
  
  return {
    'obj_props'               : obj_props,
    'tle_line1'               : tle_line1,
    'tle_line2'               : tle_line2,
    'tle_epoch_dt'            : tle_epoch_dt,
    'tle_epoch_jd'            : tle_epoch_jd,
    'target_start_dt'         : target_start_dt,
    'target_end_dt'           : target_end_dt,
    'delta_time'              : delta_time,
    'integ_time_o'            : integ_time_o,
    'integ_time_f'            : integ_time_f,
    'delta_integ_time'        : delta_integ_time,
    'mass'                    : obj_props['mass'],
    'cd'                      : obj_props['drag']['coeff'],
    'area_drag'               : obj_props['drag']['area'],
    'cr'                      : obj_props['srp']['coeff'],
    'area_srp'                : obj_props['srp']['area'],
    'use_spice'               : use_spice,
    'include_third_body'      : include_third_body,
    'include_zonal_harmonics' : include_zonal_harmonics,
    'zonal_harmonics_list'    : zonal_harmonics_list if zonal_harmonics_list else [],
    'include_srp'             : include_srp,
  }


def get_config(inputs: dict) -> SimpleNamespace:
  """
  Create configuration object from inputs dictionary.
  
  Input:
  ------
    inputs : dict
      Dictionary of input parameters.
      
  Output:
  -------
    SimpleNamespace
      Configuration object.
  """
  return SimpleNamespace(**inputs)


def setup_paths_and_files(
  norad_id        : str,
  obj_name        : str,
  target_start_dt : datetime,
  target_end_dt   : datetime,
) -> dict:
  """
  Set up all required folder paths and file names for the propagation.
  
  Input:
  ------
    norad_id : str
      NORAD catalog ID of the satellite.
    obj_name : str
      Name of the object (e.g., 'ISS').
    target_start_dt : datetime
      Target start time as a datetime object.
    target_end_dt : datetime
      Target end time as a datetime object.
  
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
  # Adjusted for location in src/config/parser.py (depth: src/config/parser.py -> config -> src -> root)
  project_root    = Path(__file__).parent.parent.parent
  data_folderpath = project_root / 'data'
  
  # SPICE kernels path
  spice_kernels_folderpath = data_folderpath / 'spice_kernels'
  lsk_filepath             = spice_kernels_folderpath / 'naif0012.tls'
  
  # Horizons ephemeris file (dynamically named)
  start_str         = target_start_dt.strftime('%Y%m%dT%H%M%SZ')
  end_str           = target_end_dt.strftime('%Y%m%dT%H%M%SZ')
  horizons_filename = f"horizons_ephem_{norad_id}_{obj_name.lower()}_{start_str}_{end_str}_1m.csv"
  horizons_filepath = data_folderpath / 'ephems' / horizons_filename
  
  return {
    'output_folderpath'        : output_folderpath,
    'spice_kernels_folderpath' : spice_kernels_folderpath,
    'horizons_filepath'        : horizons_filepath,
    'lsk_filepath'             : lsk_filepath,
  }


def get_simulation_paths(
  norad_id        : str,
  obj_name        : str,
  target_start_dt : datetime,
  target_end_dt   : datetime,
) -> tuple[Path, Path, Path, Path]:
  """
  Get paths for output, SPICE kernels, Horizons ephemeris, and leap seconds.
  
  Input:
  ------
    norad_id : str
      NORAD ID.
    obj_name : str
      Object name.
    target_start_dt : datetime
      Start time.
    target_end_dt : datetime
      End time.
      
  Output:
  -------
    tuple[Path, Path, Path, Path]
      (output_folderpath, spice_kernels_folderpath, horizons_filepath, lsk_filepath)
  """
  # Set up paths and files
  folderpaths_filepaths = setup_paths_and_files(
    norad_id        = norad_id,
    obj_name        = obj_name,
    target_start_dt = target_start_dt,
    target_end_dt   = target_end_dt,
  )
  
  return (
    folderpaths_filepaths['output_folderpath'],
    folderpaths_filepaths['spice_kernels_folderpath'],
    folderpaths_filepaths['horizons_filepath'],
    folderpaths_filepaths['lsk_filepath'],
  )


def print_input_table(
  args   : argparse.Namespace,
  parser : argparse.ArgumentParser,
) -> None:
  """
  Print a table of input arguments, their values, and status.
  """
  # Identify explicit actions from sys.argv order
  explicit_actions = []
  seen_dests = set()
  
  # Map option strings to actions for quick lookup
  opt_map = {}
  for action in parser._actions:
    for opt in action.option_strings:
      opt_map[opt] = action

  # Scan sys.argv to find order of explicit arguments
  for arg in sys.argv[1:]:
    if arg.startswith('-'):
      # Handle --opt=val syntax
      opt_name = arg.split('=')[0]
      if opt_name in opt_map:
        action = opt_map[opt_name]
        if action.dest not in seen_dests and action.dest != 'help':
          explicit_actions.append(action)
          seen_dests.add(action.dest)

  # Add remaining actions (defaults or required ones not yet seen)
  all_actions = [a for a in parser._actions if a.dest != 'help']
  remaining_actions = [a for a in all_actions if a.dest not in seen_dests]
  
  # Combine lists
  display_actions = explicit_actions + remaining_actions

  # Prepare data for printing
  rows = []
  headers = ["Argument", "Value", "Default", "Explicit"]
  col_widths = [len(h) for h in headers]
  data_widths = [0] * len(headers)

  for action in display_actions:
    val = getattr(args, action.dest, None)
    
    # Determine if explicitly set
    is_explicit = (action.dest in seen_dests) or action.required
    
    # Fallback check for explicit if not found in simple scan (e.g. abbreviations)
    if not is_explicit:
       for opt in action.option_strings:
        for arg in sys.argv:
          if arg == opt or arg.startswith(opt + "="):
            is_explicit = True
            break
        if is_explicit:
          break
        
    # Format value
    if isinstance(val, list):
      val_str = " ".join(map(str, val))
    else:
      val_str = str(val)
      
    if len(val_str) > 42:
      val_str = val_str[:39] + "..."

    # Format default value
    if isinstance(action.default, list):
      default_str = " ".join(map(str, action.default))
    else:
      default_str = str(action.default)
      
    row = [action.dest, val_str, default_str, str(is_explicit)]
    rows.append(row)
    
    # Update column widths
    for i, item in enumerate(row):
      col_widths[i] = max(col_widths[i], len(item))
      data_widths[i] = max(data_widths[i], len(item))

  # Print table
  print("\nInput Configuration")
  
  # Print headers
  header_fmt = f"  {{:<{col_widths[0]}}}    {{:<{col_widths[1]}}}    {{:<{col_widths[2]}}}    {{:<{col_widths[3]}}}"
  print(header_fmt.format(*headers))
  
  # Print separators
  separators = ["-" * w for w in col_widths]
  print(header_fmt.format(*separators))
  
  # Print rows
  for row in rows:
    print(header_fmt.format(*row))


def parse_command_line_arguments() -> argparse.Namespace:
  """
  Parse command-line arguments for the orbit propagation script.
  
  Output:
  -------
    argparse.Namespace
      An object containing the parsed arguments.
  """
  parser = argparse.ArgumentParser(description="Run high-fidelity orbit propagation.")
  
  # If no arguments provided, print help and exit
  if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)
  
  # Object definition arguments
  parser.add_argument(
    '--input-object-type',
    type     = str,
    choices  = ['norad-id', 'norad_id', 'norad id'],
    required = True,
    help     = "Type of input object identifier.",
  )
  parser.add_argument(
    '--norad-id',
    type = str,
    help = "NORAD Catalog ID of the satellite (e.g., '25544' for ISS).",
  )
  
  # Time arguments
  parser.add_argument(
    '--timespan',
    nargs    = 2,
    metavar  = ('TIME_START', 'TIME_END'),
    required = True,
    help     = "Start and end time for propagation in ISO format (e.g., '2025-10-01T00:00:00 2025-10-02T00:00:00').",
  )
  
  # Optional arguments
  parser.add_argument(
    '--include-spice',
    dest    = 'use_spice',
    action  = 'store_true',
    default = False,
    help    = "Enable SPICE functionality (disabled by default).",
  )
  parser.add_argument(
    '--include-third-body',
    dest    = 'include_third_body',
    action  = 'store_true',
    default = False,
    help    = "Enable third-body gravity (disabled by default).",
  )
  parser.add_argument(
    '--include-zonal-harmonics',
    dest    = 'include_zonal_harmonics',
    action  = 'store_true',
    default = False,
    help    = "Enable zonal harmonics (disabled by default).",
  )
  parser.add_argument(
    '--zonal-harmonics',
    dest    = 'zonal_harmonics_list',
    nargs   = '+',
    choices = ['J2', 'J3', 'J4'],
    default = ['J2'],
    help    = "List of zonal harmonics to include (default: J2).",
  )
  parser.add_argument(
    '--include-srp',
    dest    = 'include_srp',
    action  = 'store_true',
    default = False,
    help    = "Enable Solar Radiation Pressure (disabled by default).",
  )
  parser.add_argument(
    '--use-horizons-initial-guess',
    dest    = 'use_horizons_initial',
    action  = 'store_true',
    default = True,
    help    = "Use Horizons ephemeris for initial state (default: True).",
  )
  parser.add_argument(
    '--use-tle-initial-guess',
    dest   = 'use_horizons_initial',
    action = 'store_false',
    help   = "Use TLE for initial state (disables --use-horizons-initial-guess).",
  )

  # Parse arguments
  args = parser.parse_args()
  
  # Print input summary
  print_input_table(args, parser)
  
  return args