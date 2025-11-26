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


from pathlib         import Path

from src.plot.trajectory             import generate_plots
from src.propagation.propagator      import run_propagations
from src.utility.loader              import load_spice_files, unload_spice_files
from src.utility.printer             import print_results_summary
from src.config.parser               import parse_and_validate_inputs, get_config, get_simulation_paths, parse_command_line_arguments
from src.propagation.horizons_loader import get_horizons_ephemeris
from src.initialization.initializer  import get_initial_state


def main(
  input_object_type       : str,
  norad_id                : str,
  timespan                : list,
  use_spice               : bool = False,
  include_third_body      : bool = False,
  include_zonal_harmonics : bool = False,
  zonal_harmonics_list    : list = None,
  include_srp             : bool = False,
  use_horizons_initial    : bool = True,
) -> dict:
  """
  Main function to run the high-fidelity orbit propagation.
  
  This function propagates an orbit using a high-fidelity dynamics model. The
  initial state is derived from a TLE, then propagated with detailed force
  models. The result is compared with SGP4 and JPL Horizons ephemeris.
  
  Input:
  ------
    input_object_type : str
      Type of input object.
    norad_id : str
      NORAD Catalog ID of the satellite.
    timespan : list
      Start and end time for propagation in ISO format.
    use_spice : bool
      Flag to enable/disable SPICE usage.
    include_third_body : bool
      Flag to enable/disable third-body gravity.
    include_zonal_harmonics : bool
      Flag to enable/disable zonal harmonics.
    zonal_harmonics_list : list
      List of zonal harmonics to include.
    include_srp : bool
      Flag to enable/disable Solar Radiation Pressure.
    use_horizons_initial : bool
      Flag to use Horizons for initial state (default: True).
  
  Output:
  -------
    None
  """
  # Process inputs and setup
  inputs_dict = parse_and_validate_inputs(input_object_type, norad_id, timespan, use_spice, include_third_body, include_zonal_harmonics, zonal_harmonics_list, include_srp)
  config      = get_config(inputs_dict)

  # Set up paths and files
  output_folderpath, spice_kernels_folderpath, horizons_filepath, lsk_filepath = get_simulation_paths(
    norad_id        = norad_id,
    obj_name        = config.obj_props['name'],
    target_start_dt = config.target_start_dt,
    target_end_dt   = config.target_end_dt,
  )

  print("\nLoad Files")
  print(f"  Project Folderpath : {Path.cwd()}")

  # Load spice files if SPICE is enabled
  load_spice_files(config.use_spice, spice_kernels_folderpath, lsk_filepath)

  # Get Horizons ephemeris
  result_horizons = get_horizons_ephemeris(
    horizons_filepath = horizons_filepath,
    target_start_dt   = config.target_start_dt,
    target_end_dt     = config.target_end_dt,
  )

  # Determine initial state (from Horizons if available, else TLE)
  initial_state = get_initial_state(
    tle_line1            = config.tle_line1,
    tle_line2            = config.tle_line2,
    integ_time_o         = config.integ_time_o,
    result_horizons      = result_horizons,
    use_horizons_initial = use_horizons_initial,
    to_j2000             = True,
  )

  # Run propagations: high-fidelity and SGP4 at Horizons times
  result_high_fidelity, result_sgp4_at_horizons = run_propagations(
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
    spice_kernels_folderpath = spice_kernels_folderpath,
    result_horizons          = result_horizons, # type: ignore
    tle_line1                = config.tle_line1,
    tle_line2                = config.tle_line2,
  )
  
  # Display results and create plots
  print_results_summary(result_horizons, result_high_fidelity, result_sgp4_at_horizons)
  
  # Create plots
  generate_plots(
    result_horizons         = result_horizons,
    result_high_fidelity    = result_high_fidelity,
    result_sgp4_at_horizons = result_sgp4_at_horizons,
    target_start_dt         = config.target_start_dt,
    output_folderpath       = output_folderpath,
  )
  
  # Unload all SPICE kernels if they were loaded
  unload_spice_files(config.use_spice)
  
  return result_high_fidelity


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
    args.use_horizons_initial,
  )