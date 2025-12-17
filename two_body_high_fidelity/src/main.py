"""
High-Fidelity Orbit Propagator

Description:
  This script propagates the orbit of a satellite using a high-fidelity numerical
  integration model. It takes a NORAD ID, a start time, and an end time as input.
  The initial state is derived from either JPL Horizons or a Two-Line Element (TLE) set.

  The propagation includes the following forces:
  - Earth's gravity (including J2, J3, J4 zonal harmonics)
  - Atmospheric drag
  - Solar Radiation Pressure (SRP)
  - Third-body gravity from the Sun and Moon

  The script performs the following steps:
  1. Loads a reference ephemeris from JPL Horizons (if available).
  2. Derives an initial state from the selected source for the specified start time.
  3. Propagates the orbit using the high-fidelity model.
  4. Propagates the orbit using the SGP4 model for comparison.
  5. Generates and saves plots comparing the trajectories and their errors.

Usage:

  Argument                     Required   Description
  ---------------------------  --------   --------------------------------------------------
  --initial-state-source       No         Source of initial state (jpl_horizons or tle)
  --initial-state-norad-id     Yes        NORAD ID for the initial state object
  --initial-state-filename     No         Filename for custom state vector (required if source is sv)
  --timespan                   Yes        Start and end time (ISO format)
  --gravity-harmonics          No         Enable gravity harmonics (e.g. J2 J3 J4 C22 S22)
  --third-bodies               No         Enable third-body gravity (requires arguments e.g. sun)
  --srp                        No         Enable Solar Radiation Pressure
  --drag                       No         Enable Atmospheric Drag
  

  Example Commands:
    python -m src.main \
      --initial-state-norad-id <id> \
      --timespan <start> <end> \
      [--initial-state-source jpl_horizons] \
      [--gravity-harmonics J2 J3 J4 C22 S22] \
      [--third-bodies sun moon mercury venus mars jupiter saturn uranus neptune pluto] \
      [--srp] \
      [--drag]
      
    python -m src.main \
      --initial-state-source jpl_horizons \
      --initial-state-norad-id 25544 \
      --timespan 2025-10-01T00:00:00 2025-10-02T00:00:00 \
      --gravity-harmonics J2 J3 J4 \
      --third-bodies sun moon \
      --srp \
      --drag
"""
from typing                            import Optional
from datetime                          import datetime, timedelta

from src.plot.trajectory               import generate_plots
from src.propagation.propagator        import run_propagations
from src.input.loader                  import unload_files, load_files, get_horizons_ephemeris, get_celestrak_tle
from src.utility.printer               import print_results_summary
from src.input.cli                     import parse_command_line_arguments
from src.input.configuration           import build_config, print_configuration, extract_tle_to_config
from src.propagation.state_initializer import get_initial_state
from src.utility.logger                import start_logging, stop_logging

def check_data_availability(
  result              : Optional[dict],
  data_name           : str,
  needs_initial_state : bool,
  needs_comparison    : bool,
  initial_state_flag  : str,
  comparison_flag     : str,
) -> Optional[dict]:
  """
  Check if required data is available and return error dict if not.
  
  Input:
  ------
    result : dict | None
      Result dictionary from data loading (e.g., Horizons or TLE).
    data_name : str
      Human-readable name of the data source (e.g., "JPL Horizons ephemeris", "TLE").
    needs_initial_state : bool
      Whether this data is needed for initial state.
    needs_comparison : bool
      Whether this data is needed for comparison.
    initial_state_flag : str
      CLI flag for initial state source (e.g., "--initial-state-source jpl_horizons").
    comparison_flag : str
      CLI flag for comparison (e.g., "--compare-jpl-horizons").
      
  Output:
  -------
    error : dict | None
      Error dictionary with 'success': False if data unavailable, None if data is available.
  """
  # Check if data loading failed
  data_failed = result is None or not result.get('success', False)
  
  if not data_failed:
    return None  # Data is available, no error
  
  # Build reason string
  reasons = []
  if needs_initial_state:
    reasons.append(f"initial state ({initial_state_flag})")
  if needs_comparison:
    reasons.append(f"comparison ({comparison_flag})")
  
  if not reasons:
    return None  # Data not actually needed
  
  if len(reasons) > 1:
    reason_str = reasons[0] + "\n            and " + reasons[1]
  else:
    reason_str = reasons[0]
  
  print(f"\n    [ERROR] {data_name} is required for {reason_str} but is unavailable.")
  
  return {'success': False, 'message': f'{data_name} unavailable'}


def main(
  initial_state_norad_id : Optional[str],
  initial_state_filename : Optional[str],
  timespan               : list[datetime],
  include_drag           : bool           = False,
  compare_tle            : bool           = False,
  compare_jpl_horizons   : bool           = False,
  third_bodies           : Optional[list] = None,
  gravity_harmonics      : Optional[list] = None,
  include_srp            : bool           = False,
  initial_state_source   : str            = 'jpl_horizons',
) -> dict:
  """
  Main function to run the high-fidelity orbit propagation.
  
  This function orchestrates the orbit propagation process. It builds the
  configuration, loads necessary data (SPICE kernels, ephemerides), determines
  the initial state, runs the high-fidelity and SGP4 propagations, and finally
  generates results and plots.
  
  Input:
  ------
    initial_state_norad_id : str
      NORAD Catalog ID of the satellite.
    timespan : list[datetime]
      Start and end time for propagation as datetime objects.
    include_drag : bool
      Flag to enable/disable Atmospheric Drag.
    compare_tle : bool
      Flag to enable/disable comparison with TLE propagation.
    compare_jpl_horizons : bool
      Flag to enable/disable comparison with Horizons ephemeris.
    third_bodies : list | None
      List of third bodies to include (e.g., ['SUN', 'MOON']). None means disabled.
    gravity_harmonics : list | None
      List of gravity harmonics to include (e.g., ['J2', 'J3', 'J4']).
      None means disabled. Empty list means no harmonics.
    include_srp : bool
      Flag to enable/disable Solar Radiation Pressure.
    initial_state_source : str
      Source for the initial state vector ('jpl_horizons' or 'tle').
  
  Output:
  -------
    result : dict
      Dictionary containing the results of the high-fidelity propagation.
  """
  
  # Process inputs and setup
  config = build_config(
    initial_state_norad_id,
    initial_state_filename,
    timespan,
    include_drag,
    compare_tle,
    compare_jpl_horizons,
    third_bodies,
    gravity_harmonics,
    include_srp,
    initial_state_source,
  )

  # Start logging to file
  logger = start_logging(
    config.log_filepath,
  )
  
  # Print input configuration and paths
  print_configuration(config)

  # Load files (SPICE is always required)
  load_files(
    True,
    config.spice_kernels_folderpath,
    config.lsk_filepath,
  )

  # Get Horizons ephemeris (only if needed for initial state or comparison)
  result_jpl_horizons_ephemeris = None
  if config.initial_state_source == 'jpl_horizons' or config.compare_jpl_horizons:
    result_jpl_horizons_ephemeris = get_horizons_ephemeris(
      jpl_horizons_folderpath = config.jpl_horizons_folderpath,
      desired_time_o_dt       = config.time_o_dt,
      desired_time_f_dt       = config.time_f_dt,
      norad_id                = config.initial_state_norad_id,
      object_name             = config.object_name,
    )
    
    # Check if Horizons data is required but unavailable
    error = check_data_availability(
      result              = result_jpl_horizons_ephemeris,
      data_name           = "JPL Horizons ephemeris",
      needs_initial_state = config.initial_state_source == 'jpl_horizons',
      needs_comparison    = config.compare_jpl_horizons,
      initial_state_flag  = "--initial-state-source jpl_horizons",
      comparison_flag     = "--compare-jpl-horizons",
    )
    if error:
      unload_files(True)
      stop_logging(logger)
      return error

  # Get Celestrak TLE (if needed for initial state or comparison)
  result_celestrak_tle = None
  if config.initial_state_source == 'tle' or config.compare_tle:
    result_celestrak_tle = get_celestrak_tle(
      norad_id        = config.initial_state_norad_id,
      object_name     = config.object_name,
      tles_folderpath = config.tles_folderpath,
      desired_time_o_dt = config.time_o_dt,
      desired_time_f_dt = config.time_f_dt,
    )
    
    # Extract TLE data to config
    extract_tle_to_config(config, result_celestrak_tle)
    
    # Check if TLE data is required but unavailable
    error = check_data_availability(
      result              = result_celestrak_tle,
      data_name           = "TLE",
      needs_initial_state = config.initial_state_source == 'tle',
      needs_comparison    = config.compare_tle,
      initial_state_flag  = "--initial-state-source tle",
      comparison_flag     = "--compare-tle",
    )
    if error:
      unload_files(True)
      stop_logging(logger)
      return error

  # Determine initial state (from Horizons, TLE, or Custom SV)
  if config.initial_state_source == 'custom_state_vector':
    print(f"  Initial State")
    print(f"    Source : Custom State Vector File")
    print(f"    File   : {config.initial_state_filename}")
    initial_state = config.custom_state_vector
  else:
    initial_state = get_initial_state(
      tle_line_1                    = config.tle_line_1,
      tle_line_2                    = config.tle_line_2,
      time_o_dt                     = config.time_o_dt,
      result_jpl_horizons_ephemeris = result_jpl_horizons_ephemeris,
      initial_state_source          = config.initial_state_source,
      to_j2000                      = True,
    )

  # Run propagations: high-fidelity and SGP4
  result_high_fidelity_propagation, result_sgp4_propagation = run_propagations(
    initial_state                 = initial_state,
    time_o_dt                     = config.time_o_dt,
    time_f_dt                     = config.time_f_dt,
    mass                          = config.mass,
    include_drag                  = config.include_drag,
    compare_tle                   = config.compare_tle,
    compare_jpl_horizons          = config.compare_jpl_horizons,
    cd                            = config.cd,
    area_drag                     = config.area_drag,
    cr                            = config.cr,
    area_srp                      = config.area_srp,
    include_third_body            = config.include_third_body,
    third_bodies_list             = config.third_bodies_list,
    include_gravity_harmonics     = config.include_gravity_harmonics,
    gravity_harmonics_list        = config.gravity_harmonics_list,
    include_srp                   = config.include_srp,
    spice_kernels_folderpath      = config.spice_kernels_folderpath,
    result_jpl_horizons_ephemeris = result_jpl_horizons_ephemeris,
    tle_line_1                    = config.tle_line_1,
    tle_line_2                    = config.tle_line_2,
  )
  
  # Display results and create plots
  print_results_summary( 
    result_high_fidelity_propagation,
  )
  
  # Generate plots
  generate_plots(
    result_jpl_horizons_ephemeris    = result_jpl_horizons_ephemeris,
    result_high_fidelity_propagation = result_high_fidelity_propagation,
    result_sgp4_propagation          = result_sgp4_propagation,
    time_o_dt                        = config.time_o_dt,
    figures_folderpath               = config.figures_folderpath,
    compare_jpl_horizons             = config.compare_jpl_horizons,
    compare_tle                      = config.compare_tle,
    object_name                      = config.object_name,
  )
  
  # Unload all files (SPICE kernels)
  unload_files(True)
  
  # Stop logging
  stop_logging(logger)
  
  # Return high-fidelity propagation results
  return result_high_fidelity_propagation


if __name__ == "__main__":
  # Parse command-line arguments
  args = parse_command_line_arguments()
  
  # Run main function
  main(
    args.initial_state_norad_id,
    args.initial_state_filename,
    args.timespan,
    args.include_drag,
    args.compare_tle,
    args.compare_jpl_horizons,
    args.third_bodies,
    args.gravity_harmonics,
    args.include_srp,
    args.initial_state_source,
  )