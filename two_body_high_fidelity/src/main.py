"""
High-Fidelity Orbit Propagator

Description:
  This script propagates the orbit of a satellite using a high-fidelity numerical
  integration model. It takes a NORAD ID, a start time, and an end time as input.
  The initial state is derived from a hardcoded Two-Line Element (TLE) set.

  The propagation includes the following forces:
  - Earth's gravity (including J2, J3, J4 zonal harmonics)
  - Atmospheric drag
  - Solar Radiation Pressure (SRP)
  - Third-body gravity from the Sun and Moon

  The script performs the following steps:
  1. Loads a reference ephemeris from JPL Horizons (if available).
  2. Derives an initial state from the TLE for the specified start time.
  3. Propagates the orbit using the high-fidelity model.
  4. Propagates the orbit using the SGP4 model for comparison.
  5. Generates and saves plots comparing the trajectories and their errors.

Usage:

  Argument                     Required   Description
  ---------------------------  --------   --------------------------------------------------
  --input-object-type          Yes        Type of input object (e.g., norad-id)
  --norad-id                   Yes*       NORAD ID (required for norad-id type)
  --timespan                   Yes        Start and end time (ISO format)
  --include-zonal-harmonics    No         Enable zonal harmonics
  --zonal-harmonics            No         List of zonal harmonics: J2 (default), J3, and J4
  --include-spice              No         Enable SPICE functionality
  --include-third-body         No         Enable third-body gravity
  --include-srp                No         Enable Solar Radiation Pressure

  Example Commands:
    python -m src.main \
      --input-object-type <type> \
      --norad-id <id> \
      --timespan <start> <end> \
      [--include-spice] \
      [--include-third-body] \
      [--include-srp] \
      [--include-zonal-harmonics] \
      [--zonal-harmonics <J2|J3|J4>]

    python -m src.main \
      --input-object-type norad-id \
      --norad-id 25544 \
      --timespan 2025-10-01T00:00:00 2025-10-02T00:00:00 \
      --include-zonal-harmonics \
      --zonal-harmonics J2 J3 J4 \
      --include-third-body \
      --include-srp \
      --include-spice


    
"""


from pathlib import Path

from src.plot.trajectory             import generate_plots
from src.propagation.propagator      import run_propagations
from src.load.loader                 import unload_files, load_files
from src.utility.printer             import print_results_summary
from src.load.parser                 import parse_command_line_arguments
from src.load.configurer             import build_config
from src.load.loader                 import get_horizons_ephemeris
from src.model.initial_state_guesser import get_initial_state
from typing import Optional


def main(
  input_object_type          : str,
  norad_id                   : str,
  timespan                   : list,
  use_spice                  : bool           = False,
  include_third_body         : bool           = False,
  include_zonal_harmonics    : bool           = False,
  zonal_harmonics_list       : Optional[list] = None,
  include_srp                : bool           = False,
  use_horizons_initial_guess : bool           = True,
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
    use_spice : bool
      Flag to enable/disable SPICE usage.
    include_third_body : bool
      Flag to enable/disable third-body gravity forces.
    include_zonal_harmonics : bool
      Flag to enable/disable zonal harmonic gravity terms.
    zonal_harmonics_list : list
      List of specific zonal harmonics to include (e.g., ['J2', 'J3', 'J4']).
    include_srp : bool
      Flag to enable/disable Solar Radiation Pressure.
    use_horizons_initial : bool
      Flag to use JPL Horizons ephemeris for the initial state vector.
      If False, the TLE is used.
  
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
    use_spice,
    include_third_body,
    include_zonal_harmonics,
    zonal_harmonics_list,
    include_srp,
    use_horizons_initial_guess,
  )

  # Load files
  load_files(
    config.use_spice,
    config.spice_kernels_folderpath,
    config.lsk_filepath,
  )

  # Get Horizons ephemeris
  result_horizons_ephemeris = get_horizons_ephemeris(
    horizons_filepath = config.horizons_filepath,
    target_start_dt   = config.target_start_dt,
    target_end_dt     = config.target_end_dt,
  )

  # Determine initial state (from Horizons if available, else TLE)
  initial_state = get_initial_state(
    tle_line1                  = config.tle_line1,
    tle_line2                  = config.tle_line2,
    integ_time_o               = config.integ_time_o,
    result_horizons            = result_horizons_ephemeris,
    use_horizons_initial_guess = config.use_horizons_initial_guess,
    to_j2000                   = True,
  )

  # Run propagations: high-fidelity and SGP4 at Horizons times
  result_high_fidelity_propagation, result_sgp4_propagation = run_propagations(
    initial_state            = initial_state,
    integ_time_o             = config.integ_time_o,
    integ_time_f             = config.integ_time_f,
    target_start_dt          = config.target_start_dt,
    target_end_dt            = config.target_end_dt,
    mass                     = config.mass,
    cd                       = config.cd,
    area_drag                = config.area_drag,
    cr                       = config.cr,
    area_srp                 = config.area_srp,
    use_spice                = config.use_spice,
    include_third_body       = config.include_third_body,
    include_zonal_harmonics  = config.include_zonal_harmonics,
    zonal_harmonics_list     = config.zonal_harmonics_list,
    include_srp              = config.include_srp,
    spice_kernels_folderpath = config.spice_kernels_folderpath,
    result_horizons          = result_horizons_ephemeris, # type: ignore
    tle_line1                = config.tle_line1,
    tle_line2                = config.tle_line2,
  )
  
  # Display results and create plots
  print_results_summary(
    result_horizons_ephemeris,
    result_high_fidelity_propagation,
    result_sgp4_propagation,
  )
  
  # Generate plots
  generate_plots(
    result_horizons         = result_horizons_ephemeris,
    result_high_fidelity    = result_high_fidelity_propagation,
    result_sgp4_at_horizons = result_sgp4_propagation,
    target_start_dt         = config.target_start_dt,
    output_folderpath       = config.output_folderpath,
  )
  
  # Unload all files (SPICE kernels)
  unload_files(config.use_spice)
  
  return result_high_fidelity_propagation


if __name__ == "__main__":
  args = parse_command_line_arguments()
  main(
    args.input_object_type,
    args.norad_id,
    args.timespan,
    args.use_spice,
    args.include_third_body,
    args.include_zonal_harmonics,
    args.zonal_harmonics_list,
    args.include_srp,
    args.use_horizons_initial_guess,
  )