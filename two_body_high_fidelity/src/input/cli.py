import argparse
import sys
from datetime import datetime


def parse_time(
  time_str : str,
) -> datetime:
  """
  Parse a time string into a datetime object.
  
  Accepted formats include:
  - ISO 8601 with 'T' separator: "2025-10-01T00:00:00"
  - ISO 8601 with 'Z' suffix: "2025-10-01T00:00:00Z"
  - Space-separated: "2025-10-01 00:00:00"
  - With microseconds: "2025-10-01 00:00:00.123456"
  
  The 'Z' suffix (UTC indicator) is stripped before parsing for 
  Python < 3.11 compatibility.
  
  Input:
  ------
    time_str : str
      Time string to parse.
      
  Output:
  -------
    datetime
      Parsed datetime object.
  """
  # Handle 'Z' suffix for Python < 3.11 compatibility
  if time_str.endswith('Z'):
    time_str = time_str[:-1]

  try:
    return datetime.fromisoformat(time_str)
  except ValueError:
    # Fallback for other formats if needed
    formats = [
      "%Y-%m-%d %H:%M:%S",
      "%Y-%m-%d %H:%M:%S.%f",
    ]
    for fmt in formats:
      try:
        return datetime.strptime(time_str, fmt)
      except ValueError:
        continue
    raise ValueError(f"Cannot parse time string: {time_str}")


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
    '--spice',
    dest    = 'include_spice',
    action  = 'store_true',
    default = False,
    help    = "Enable SPICE functionality (disabled by default).",
  )
  parser.add_argument(
    '--third-bodies',
    '--include-third-bodies',
    dest    = 'third_bodies',
    nargs   = '+', # accepts 1 or more args.
    choices = ['SUN', 'MOON', 'sun', 'moon'],
    default = None,
    help    = "Enable third-body gravity. Required args: SUN MOON (e.g. --third-bodies SUN).",
  )
  parser.add_argument(
    '--zonal-harmonics',
    '--include-zonal-harmonics',
    dest    = 'zonal_harmonics',
    nargs   = '+', # accepts 1 or more args.
    choices = ['J2', 'J3', 'J4'],
    default = None,
    help    = "Enable zonal harmonics. Required args: J2 J3 J4 (e.g. --zonal-harmonics J2).",
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
  
  return args
