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
  --input-object-type          Yes        Type of input object (e.g., norad-id)
  --norad-id                   Yes*       NORAD ID (required for norad-id type)
  --timespan                   Yes        Start and end time (ISO format)
  --initial-state-source       No         Source of initial state (jpl_horizons or tle)
  --zonal-harmonics            No         Enable zonal harmonics (requires arguments e.g. J2)
  --spice                      No         Enable SPICE functionality
  --third-bodies               No         Enable third-body gravity (requires arguments e.g. sun)
  --srp                        No         Enable Solar Radiation Pressure
  --drag                       No         Enable Atmospheric Drag
  

  Example Commands:
    python -m src.main \
      --input-object-type <type> \
      --norad-id <id> \
      --timespan <start> <end> \
      [--initial-state-source jpl_horizons] \
      [--zonal-harmonics J2 J3 J4] \
      [--third-bodies sun moon] \
      [--srp] \
      [--spice] \
      [--drag]
      
    python -m src.main \
      --input-object-type norad-id \
      --norad-id 25544 \
      --timespan 2025-10-01T00:00:00 2025-10-02T00:00:00 \
      --initial-state-source jpl_horizons \
      --zonal-harmonics J2 J3 J4 \
      --third-bodies sun moon \
      --srp \
      --spice \
      --drag
"""
from typing                            import Optional
from datetime                          import datetime, timedelta

from src.plot.trajectory               import generate_plots
from src.propagation.propagator        import run_propagations
from src.propagation.utility           import determine_actual_times
from src.input.loader                  import unload_files, load_files, get_horizons_ephemeris, get_celestrak_tle
from src.utility.printer               import print_results_summary
from src.input.cli                     import parse_command_line_arguments
from src.input.configuration           import build_config, print_configuration, extract_tle_to_config
from src.propagation.state_initializer import get_initial_state
from src.utility.logger                import start_logging, stop_logging

def main(
  input_object_type    : str,
  norad_id             : str,
  timespan             : list,
  include_spice        : bool           = False,
  include_drag         : bool           = False,
  compare_tle          : bool           = False,
  compare_jpl_horizons : bool           = False,
  third_bodies         : Optional[list] = None,
  zonal_harmonics      : Optional[list] = None,
  include_srp          : bool           = False,
  initial_state_source : str            = 'jpl_horizons',
) -> dict:
  """
  Main function to run the high-fidelity orbit propagation.
  
  This function orchestrates the orbit propagation process. It builds the
  configuration, loads necessary data (SPICE kernels, ephemerides), determines
  the initial state, runs the high-fidelity and SGP4 propagations, and finally
  generates results and plots.
  
  Input:
  ------
    input_object_type : str
      Type of input object (e.g., 'norad-id').
    norad_id : str
      NORAD Catalog ID of the satellite.
    timespan : list
      Start and end time for propagation in ISO format.
    include_spice : bool
      Flag to enable/disable SPICE usage.
    include_drag : bool
      Flag to enable/disable Atmospheric Drag.
    compare_tle : bool
      Flag to enable/disable comparison with TLE propagation.
    compare_jpl_horizons : bool
      Flag to enable/disable comparison with Horizons ephemeris.
    third_bodies : list | None
      List of third bodies to include (e.g., ['SUN', 'MOON']). None means disabled.
    zonal_harmonics : list | None
      List of specific zonal harmonics to include (e.g., ['J2', 'J3', 'J4']).
      None means disabled. Empty list means default (J2).
    include_srp : bool
      Flag to enable/disable Solar Radiation Pressure.
    initial_state_source : str
      Source for the initial state vector ('jpl_horizons' or 'tle').
  
  Output:
  -------
    dict
      Dictionary containing the results of the high-fidelity propagation.
  """
  
  # Process inputs and setup
  config = build_config(
    input_object_type,
    norad_id,
    timespan,
    include_spice,
    include_drag,
    compare_tle,
    compare_jpl_horizons,
    third_bodies,
    zonal_harmonics,
    include_srp,
    initial_state_source,
  )

  # Start logging to file
  logger = start_logging(
    config.log_filepath,
  )
  
  # Print input configuration and paths
  print_configuration(config)

  # Load files
  load_files(
    config.include_spice,
    config.spice_kernels_folderpath,
    config.lsk_filepath,
  )

  # Get Horizons ephemeris (only if needed for initial state or comparison)
  result_jpl_horizons_ephemeris = None
  if config.initial_state_source == 'jpl_horizons' or config.compare_jpl_horizons:
    result_jpl_horizons_ephemeris = get_horizons_ephemeris(
      jpl_horizons_folderpath = config.jpl_horizons_folderpath,
      desired_time_o_dt       = config.desired_time_o_dt,
      desired_time_f_dt       = config.desired_time_f_dt,
      norad_id                = config.norad_id,
      object_name             = config.object_name,
    )

  # Get Celestrak TLE (if needed for initial state or comparison)
  result_celestrak_tle = None
  if config.initial_state_source == 'tle' or config.compare_tle:
    result_celestrak_tle = get_celestrak_tle(
      norad_id          = config.norad_id,
      object_name       = config.object_name,
      tles_folderpath   = config.tles_folderpath,
      desired_time_o_dt = config.desired_time_o_dt,
      desired_time_f_dt = config.desired_time_f_dt,
    )
    
    # Extract TLE data to config
    extract_tle_to_config(config, result_celestrak_tle)

  # Determine actual times if Horizons is available (for grid alignment)
  actual_time_o_dt, actual_time_f_dt = determine_actual_times(
    result_jpl_horizons_ephemeris = result_jpl_horizons_ephemeris,
    desired_time_o_dt             = config.desired_time_o_dt,
    desired_time_f_dt             = config.desired_time_f_dt
  )

  # Determine initial state (from Horizons if available, else TLE)
  initial_state = get_initial_state(
    tle_line_1                    = config.tle_line_1,
    tle_line_2                    = config.tle_line_2,
    desired_time_o_dt             = config.desired_time_o_dt,
    result_jpl_horizons_ephemeris = result_jpl_horizons_ephemeris,
    initial_state_source          = config.initial_state_source,
    to_j2000                      = True,
  )

  # Run propagations: high-fidelity and SGP4
  result_high_fidelity_propagation, result_sgp4_propagation = run_propagations(
    initial_state                 = initial_state,
    desired_time_o_dt             = config.desired_time_o_dt,
    desired_time_f_dt             = config.desired_time_f_dt,
    actual_time_o_dt              = actual_time_o_dt,
    actual_time_f_dt              = actual_time_f_dt,
    mass                          = config.mass,
    include_drag                  = config.include_drag,
    compare_tle                   = config.compare_tle,
    cd                            = config.cd,
    area_drag                     = config.area_drag,
    cr                            = config.cr,
    area_srp                      = config.area_srp,
    use_spice                     = config.include_spice,
    include_third_body            = config.include_third_body,
    third_bodies_list             = config.third_bodies_list,
    include_zonal_harmonics       = config.include_zonal_harmonics,
    zonal_harmonics_list          = config.zonal_harmonics_list,
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
    desired_time_o_dt                = config.desired_time_o_dt,
    figures_folderpath               = config.figures_folderpath,
    compare_jpl_horizons             = config.compare_jpl_horizons,
    compare_tle                      = config.compare_tle,
    object_name                      = config.object_name,
  )
  
  # Unload all files (SPICE kernels)
  unload_files(config.include_spice)
  
  # Stop logging
  stop_logging(logger)
  
  # Return high-fidelity propagation results
  return result_high_fidelity_propagation


if __name__ == "__main__":
  # Parse command-line arguments
  args = parse_command_line_arguments()
  
  # Run main function
  main(
    args.input_object_type,
    args.norad_id,
    args.timespan,
    args.include_spice,
    args.include_drag,
    args.compare_tle,
    args.compare_jpl_horizons,
    args.third_bodies,
    args.zonal_harmonics,
    args.include_srp,
    args.initial_state_source,
  )