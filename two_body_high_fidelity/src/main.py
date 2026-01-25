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
from typing                            import Optional, Union
from datetime                          import datetime

import numpy as np

from src.plot.plot_generator           import generate_plots
from src.propagation.propagator        import run_propagations
from src.input.loader                  import unload_files, load_files, get_horizons_ephemeris, get_celestrak_tle
from src.utility.printer               import final_print
from src.input.cli                     import parse_command_line_arguments
from src.input.configuration           import build_config, print_configuration, extract_tle_to_config
from src.propagation.state_initializer import get_initial_state
from src.utility.logger                import start_logging, stop_logging
from src.schemas.propagation           import PropagationConfig, PropagationResult
from src.schemas.state                 import TLEData
from src.schemas.spacecraft            import ManeuversConfig

from src.orbit_determination.ekf_processor         import process_measurements_with_ekf, apply_rts_smoother
from src.orbit_determination.measurement_simulator import MeasurementSimulator
from src.model.dynamics                            import AccelerationSTMDot, GeneralStateEquationsOfMotion


def check_data_availability(
  result              : Union[TLEData, PropagationResult, None],
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
    result : TLEData | PropagationResult | None
      Result from data loading (e.g., Horizons or TLE).
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
  # For TLEData, it's either a valid object or None
  # For PropagationResult, check the success attribute
  if result is None:
    data_failed = True
  elif isinstance(result, TLEData):
    data_failed = False  # TLEData is always valid if it exists
  elif isinstance(result, PropagationResult):
    data_failed = not result.success
  else:
    data_failed = True  # Unknown type
  
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
  initial_state_norad_id         : Optional[str],
  initial_state_filename         : Optional[str],
  timespan                       : list[datetime],
  include_drag                   : bool           = False,
  compare_tle                    : bool           = False,
  compare_jpl_horizons           : bool           = False,
  third_bodies                   : Optional[list] = None,
  gravity_harmonics              : Optional[list] = None,
  include_srp                    : bool           = False,
  include_relativity             : bool           = False,
  include_solid_tides            : bool           = False,
  include_ocean_tides            : bool           = False,
  auto_download                  : bool           = False,
  initial_state_source           : str            = 'jpl_horizons',
  gravity_harmonics_degree_order : Optional[list] = None,
  gravity_model_filename         : Optional[str]  = None,
  atol                           : float          = 1e-15,
  rtol                           : float          = 1e-12,
  include_tracker_skyplots       : bool           = False,
  tracker_filename               : Optional[str]  = None,
  tracker_filepath               : Optional[str]  = None,
  include_tracker_on_body        : bool           = False,
  maneuver_filename              : Optional[str]  = None,
  include_orbit_determination    : bool           = False,
  process_noise_pos              : Optional[float] = None,
  process_noise_vel              : Optional[float] = None,
) -> PropagationResult:
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
    gravity_harmonics_degree_order : list | None
      Degree and order for gravity harmonics (e.g., [4, 4]). None means disabled.
    gravity_model_filename : str | None
      Filename for custom gravity harmonics file. None means disabled.
    atol : float
      Absolute tolerance for numerical integration.
    rtol : float
      Relative tolerance for numerical integration.
  
  Output:
  -------
    result : PropagationResult
      Object containing the results of the high-fidelity propagation.
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
    include_relativity,
    include_solid_tides,
    include_ocean_tides,
    auto_download,
    initial_state_source,
    gravity_harmonics_degree_order,
    gravity_model_filename,
    atol,
    rtol,
    include_tracker_skyplots,
    tracker_filename,
    tracker_filepath,
    include_tracker_on_body,
    maneuver_filename,
    include_orbit_determination,
    process_noise_pos,
    process_noise_vel,
  )

  # Start logging to file
  logger = start_logging(
    config.output_paths.log_filepath,
  )
  
  # Print input configuration and paths
  print_configuration(config)

  # Load files: SPICE, spherical harmonics coefficients, trackers, maneuvers
  # Note: Tracker azimuth normalization happens inside load_files()
  spherical_harmonics_model, trackers, maneuvers_from_file = load_files(
    spice_kernels_folderpath  = config.output_paths.spice_kernels_folderpath,
    lsk_filepath              = config.output_paths.lsk_filepath,
    gravity_model_folderpath  = config.gravity.folderpath,
    gravity_model_filename    = config.gravity.filename,
    gravity_model_degree      = config.gravity.spherical_harmonics.degree,
    gravity_model_order       = config.gravity.spherical_harmonics.order,
    gravity_coefficient_names = config.gravity.spherical_harmonics.coefficients if config.gravity.spherical_harmonics.enabled else None,
    tracker_filepath          = config.output_paths.tracker_filepath,
    maneuver_filename         = maneuver_filename,
  )

  # Update gravity model with loaded values
  if spherical_harmonics_model is not None:
    config.gravity.spherical_harmonics.model  = spherical_harmonics_model
    config.gravity.spherical_harmonics.gp     = spherical_harmonics_model.gp
    config.gravity.spherical_harmonics.radius = spherical_harmonics_model.radius

  # Update spacecraft with loaded maneuvers
  if maneuvers_from_file:
    config.spacecraft.maneuvers._items = maneuvers_from_file

  # Get Horizons ephemeris (only if needed for initial state or comparison)
  result_jpl_horizons_ephemeris = None
  if config.initial_state.source == 'jpl_horizons' or config.comparison.compare_jpl_horizons:
    result_jpl_horizons_ephemeris = get_horizons_ephemeris(
      jpl_horizons_folderpath = config.output_paths.jpl_horizons_folderpath,
      desired_time_o_dt       = config.time_o_dt,
      desired_time_f_dt       = config.time_f_dt,
      norad_id                = config.initial_state.norad_id,
      object_name             = config.object_name,
      auto_download           = config.auto_download,
    )
    
    # Check if Horizons data is required but unavailable
    error = check_data_availability(
      result              = result_jpl_horizons_ephemeris,
      data_name           = "JPL Horizons ephemeris",
      needs_initial_state = config.initial_state.source == 'jpl_horizons',
      needs_comparison    = config.comparison.compare_jpl_horizons,
      initial_state_flag  = "--initial-state-source jpl_horizons",
      comparison_flag     = "--compare-jpl-horizons",
    )
    if error:
      unload_files()
      stop_logging(logger)
      return error

  # Get Celestrak TLE (if needed for initial state or comparison)
  result_celestrak_tle = None
  if config.initial_state.source == 'tle' or config.comparison.compare_tle:
    result_celestrak_tle = get_celestrak_tle(
      norad_id          = config.initial_state.norad_id,
      object_name       = config.object_name,
      tles_folderpath   = config.output_paths.tles_folderpath,
      desired_time_o_dt = config.time_o_dt,
      desired_time_f_dt = config.time_f_dt,
      auto_download     = config.auto_download,
    )
    
    # Extract TLE data to config
    extract_tle_to_config(config, result_celestrak_tle)
    
    # Check if TLE data is required but unavailable
    error = check_data_availability(
      result              = result_celestrak_tle,
      data_name           = "TLE",
      needs_initial_state = config.initial_state.source == 'tle',
      needs_comparison    = config.comparison.compare_tle,
      initial_state_flag  = "--initial-state-source tle",
      comparison_flag     = "--compare-tle",
    )
    if error:
      unload_files()
      stop_logging(logger)
      return error

  # Determine initial state: JPL Horizons, TLE, or Custom State Vector
  initial_state, initial_epoch_dt = get_initial_state(
    tle_line_1                    = config.tle_line_1,
    tle_line_2                    = config.tle_line_2,
    time_o_dt                     = config.time_o_dt,
    result_jpl_horizons_ephemeris = result_jpl_horizons_ephemeris,
    initial_state_source          = config.initial_state.source,
    custom_state_vector           = config.initial_state.state,
    initial_state_filename        = config.initial_state.filename,
  )

  # Create PropagationConfig for the propagator using actual initial epoch
  propagation_config = PropagationConfig(
    time_o_dt = initial_epoch_dt,
    time_f_dt = config.time_f_dt,
  )

  # Run propagations: high-fidelity and SGP4
  result_high_fidelity_propagation, result_sgp4_propagation = run_propagations(
    initial_state                 = initial_state,
    propagation_config            = propagation_config,
    spacecraft                    = config.spacecraft,
    compare_tle                   = config.comparison.compare_tle,
    compare_jpl_horizons          = config.comparison.compare_jpl_horizons,
    result_jpl_horizons_ephemeris = result_jpl_horizons_ephemeris,
    tle_line_1                    = config.tle_line_1,
    tle_line_2                    = config.tle_line_2,
    two_body_gravity_model        = config.gravity,
  )

  # Orbit Determination: Process measurements with EKF and RTS smoother
  od_filter_states = None
  od_filter_covariances = None
  od_smoother_states = None
  od_smoother_covariances = None
  od_estimation_times = None
  od_measurement_times = None
  od_residual_data = None
  if include_orbit_determination and trackers is not None and len(trackers) > 0:
    
    # Use JPL Horizons as truth for measurement simulation
    if result_jpl_horizons_ephemeris is not None and result_jpl_horizons_ephemeris.success:
      section_title = "Orbit Determination with Extended Kalman Filter"
      print("\n" + "-" * len(section_title))
      print(section_title)
      print("-" * len(section_title))

      # Use first tracker for orbit determination
      tracker_od = trackers[0]
      print(f"\n  Using tracker: {tracker_od.name}")

      # Simulate measurements from JPL Horizons truth
      print(f"  Simulating measurements from JPL Horizons ephemeris")
      simulator = MeasurementSimulator(result_jpl_horizons_ephemeris, tracker_od, config.time_o_dt)
      noise_config = simulator.get_tracker_noise_config()
      measurements = simulator.simulate(noise_config=noise_config, seed=42, include_rates=True)

      # Filter measurements to only include times when tracker has visibility
      measurements.truth    = measurements.get_visible_truth()
      measurements.measured = measurements.get_visible_measured()

      # Store measurement times for plotting (only visible times)
      od_measurement_times = measurements.measured.delta_time_epoch.copy()

      # Use ephemeris initial state as initial guess (this is what we would have from propagation)
      print(f"  Using ephemeris initial state as initial guess")
      initial_guess = initial_state.copy()

      # Create high-fidelity dynamics model for EKF propagation
      print(f"  Initializing high-fidelity dynamics for EKF")
      od_acceleration = AccelerationSTMDot(
        gravity_config = config.gravity,
        spacecraft     = config.spacecraft,
      )
      od_dynamics = GeneralStateEquationsOfMotion(acceleration=od_acceleration)

      # Construct Process Noise (Q) matrix from configuration
      q_pos_sigma = config.orbit_determination.process_noise_pos
      q_vel_sigma = config.orbit_determination.process_noise_vel

      print(f"  Using process noise: pos={q_pos_sigma:.1e}, vel={q_vel_sigma:.1e}")
      
      od_process_noise = np.diag([
        q_pos_sigma**2, q_pos_sigma**2, q_pos_sigma**2,
        q_vel_sigma**2, q_vel_sigma**2, q_vel_sigma**2
      ])

      # Process with EKF
      print(f"  Processing {len(measurements.measured.delta_time_epoch)} measurements with EKF")
      od_filter_states, od_filter_covariances, od_estimation_times, od_residual_data = process_measurements_with_ekf(
        measurements       = measurements,
        tracker            = tracker_od,
        initial_state      = initial_guess,
        epoch_dt_utc       = config.time_o_dt,
        ephemeris_times    = result_jpl_horizons_ephemeris.time_grid.deltas,
        propagation_times  = None,  # Use ephemeris_times
        initial_covariance = None,  # Use defaults
        process_noise      = od_process_noise,
        dynamics           = od_dynamics,  # High-fidelity dynamics
      )

      print(f"  ✓ EKF filtering complete")
      print(f"    Initial position uncertainty: ±{100.0:.0f} m (1-sigma)")
      print(f"    Initial velocity uncertainty: ±{1.0:.1f} m/s (1-sigma)")

      # Compute final filter uncertainties
      final_cov = od_filter_covariances[:, :, -1]
      final_pos_sigma = (final_cov[0, 0] + final_cov[1, 1] + final_cov[2, 2])**0.5 / 3**0.5
      final_vel_sigma = (final_cov[3, 3] + final_cov[4, 4] + final_cov[5, 5])**0.5 / 3**0.5
      print(f"    Final filter position uncertainty: ±{final_pos_sigma:.1f} m (1-sigma)")

      print(f"    Final filter velocity uncertainty: ±{final_vel_sigma:.4f} m/s (1-sigma)")

      # Apply RTS smoother to get smoothed estimates
      print(f"\n  Applying RTS smoother")
      od_smoother_states, od_smoother_covariances = apply_rts_smoother(
        filter_result        = od_filter_states,
        filtered_covariances = od_filter_covariances,
        estimation_times     = od_estimation_times,
        epoch_dt_utc         = config.time_o_dt,
        dynamics             = od_dynamics,
      )

      # Compute final smoother uncertainties
      final_smooth_cov = od_smoother_covariances[:, :, -1]
      final_smooth_pos_sigma = (final_smooth_cov[0, 0] + final_smooth_cov[1, 1] + final_smooth_cov[2, 2])**0.5 / 3**0.5
      final_smooth_vel_sigma = (final_smooth_cov[3, 3] + final_smooth_cov[4, 4] + final_smooth_cov[5, 5])**0.5 / 3**0.5
      print(f"  ✓ RTS smoothing complete")
      print(f"    Final smoother position uncertainty: ±{final_smooth_pos_sigma:.1f} m (1-sigma)")
      print(f"    Final smoother velocity uncertainty: ±{final_smooth_vel_sigma:.4f} m/s (1-sigma)")

      # Replace high-fidelity propagation result with smoothed OD estimates
      result_high_fidelity_propagation = od_smoother_states

  # Generate plots
  generate_plots(
    result_jpl_horizons_ephemeris    = result_jpl_horizons_ephemeris,
    result_high_fidelity_propagation = result_high_fidelity_propagation,
    result_sgp4_propagation          = result_sgp4_propagation,
    time_o_dt                        = config.time_o_dt,
    figures_folderpath               = config.output_paths.figures_folderpath,
    compare_jpl_horizons             = config.comparison.compare_jpl_horizons,
    compare_tle                      = config.comparison.compare_tle,
    object_name                      = config.object_name,
    object_name_display              = config.object_name_display,
    trackers                         = trackers,
    include_tracker_on_body          = include_tracker_on_body,
    od_filter_states                 = od_filter_states,
    od_filter_covariances            = od_filter_covariances,
    od_smoother_states               = od_smoother_states,
    od_smoother_covariances          = od_smoother_covariances,
    od_estimation_times              = od_estimation_times,
    od_measurement_times             = od_measurement_times,
    od_residual_data                 = od_residual_data,
    include_orbit_determination      = include_orbit_determination,
  )
  
  # Cleanup: Unload all files (SPICE kernels)
  unload_files()

  # Cleanup: final print
  final_print()
  
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
    args.gravity_harmonics_coefficients,
    args.include_srp,
    args.include_relativity,
    args.include_solid_tides,
    args.include_ocean_tides,
    args.auto_download,
    args.initial_state_source,
    args.gravity_harmonics_degree_order,
    args.gravity_model_filename,
    args.atol,
    args.rtol,
    args.include_tracker_skyplots,
    args.tracker_filename,
    args.tracker_filepath,
    args.include_tracker_on_body,
    args.maneuver_filename,
    args.include_orbit_determination,
    args.process_noise_pos,
    args.process_noise_vel,
  )