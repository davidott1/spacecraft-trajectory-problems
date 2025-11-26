import argparse
import sys
from datetime           import datetime


def parse_time(
  time_str : str,
) -> datetime:
  """
  Parse a time string in ISO format.
  
  Accepted formats include:
  - "YYYY-MM-DD" (e.g., "2025-10-01")
  - "YYYY-MM-DDTHH:MM:SS" (e.g., "2025-10-01T12:00:00")
  - "YYYY-MM-DD HH:MM:SS" (e.g., "2025-10-01 12:00:00")
  
  Output:
  -------
    datetime.datetime
      An object representing the parsed time.
  """
  try:
    return datetime.fromisoformat(time_str)
  except ValueError:
    # Fallback for space separator if fromisoformat fails (older python versions)
    # or other common formats
    try:
      return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
      pass
      
    raise ValueError(f"Cannot parse time string: {time_str}")


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


def parse_command_line_arguments(
) -> argparse.Namespace:
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
    '--initial-state',
    '--initial-state-source',
    dest    = 'initial_state_source',
    type    = str,
    choices = ['horizons', 'jpl-horizons', 'jpl_horizons', 'jpl horizons', 'tle'],
    default = 'horizons',
    help    = "Source for initial state vector (default: horizons).",
  )

  # Parse arguments
  args = parser.parse_args()
  
  # Print input summary
  print_input_table(args, parser)
  
  return args
