import sys
import argparse
from src.utility.time_helper import parse_time


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
    required = True,
    help     = "Start and end time for propagation in ISO format (e.g., '2025-10-01T00:00:00 2025-10-02T00:00:00').",
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
    '--gravity-harmonics',
    type    = str,
    nargs   = '*',
    default = [],
    help    = 'Gravity harmonics to include (e.g., J2 J3 J4 C22 S22). Default: none.',
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
    '--gravity-harmonics-file',
    dest    = 'gravity_harmonics_file',
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
  
  
  # Parse arguments
  args = parser.parse_args()
  
  return args
