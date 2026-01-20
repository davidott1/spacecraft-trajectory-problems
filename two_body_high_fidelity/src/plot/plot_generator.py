"""
Plot generation orchestration functions.

This module contains high-level functions that orchestrate the generation
of multiple plots and manage saving them to disk.
"""
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib  import Path
from typing   import Optional

from src.plot.plot_3d                          import plot_3d_trajectories, plot_3d_trajectories_body_fixed, plot_3d_trajectory_sun_centered
from src.plot.plot_timeseries                  import plot_time_series, plot_time_series_error
from src.plot.plot_groundtrack                 import plot_ground_track
from src.plot.plot_skyplot                     import plot_skyplot, plot_pass_timeseries, plot_measurement_errors, plot_error_skyplot
from src.plot.plot_covariance                  import plot_covariance_timeseries, plot_covariance_components
from src.schemas.propagation                   import PropagationResult
from src.schemas.state                         import TrackerStation
from src.orbit_determination.measurement_simulator import MeasurementSimulator


def generate_error_plots(
  result_jpl_horizons_ephemeris    : Optional[PropagationResult],
  result_high_fidelity_propagation : PropagationResult,
  result_sgp4_propagation          : Optional[PropagationResult],
  time_o_dt                        : datetime,
  figures_folderpath               : Path,
  compare_jpl_horizons             : bool,
  compare_tle                      : bool,
  object_name                      : str = "object",
  object_name_display              : str = "Object",
) -> dict:
  """
  Generate and save error comparison plots.
  
  Returns:
  --------
    dict : Dictionary with error plot filenames organized by comparison type.
  """
  error_files = {
    'hf_vs_horizons': [],
    'hf_vs_sgp4': [],
    'sgp4_vs_horizons': [],
  }
  
  # If neither comparison is requested, do nothing
  if not (compare_jpl_horizons or compare_tle):
    return error_files

  # Define availability flags
  has_horizons      = result_jpl_horizons_ephemeris is not None and result_jpl_horizons_ephemeris.success
  has_high_fidelity = result_high_fidelity_propagation.success
  has_sgp4          = result_sgp4_propagation is not None and result_sgp4_propagation.success

  # Check for pre-computed ephemeris-time data
  has_hf_at_ephem   = has_high_fidelity and result_high_fidelity_propagation.at_ephem_times is not None
  has_sgp4_at_ephem = has_sgp4 and result_sgp4_propagation.at_ephem_times is not None

  # Lowercase name for filenames
  name_lower = object_name.lower()

  # High-Fidelity Relative To JPL Horizons (compare at ephemeris times)
  if compare_jpl_horizons and has_horizons and has_hf_at_ephem:
    # Build result dict for comparison using pre-computed at_ephem_times data
    # Note: plot_time_series_error expects dicts or objects. 
    # at_ephem_times is a dict in PropagationResult.
    hf_at_ephem = result_high_fidelity_propagation.at_ephem_times
    
    fig_err_ts = plot_time_series_error(
      result_ref  = result_jpl_horizons_ephemeris,
      result_comp = hf_at_ephem,
      epoch       = time_o_dt,
      use_ric     = True,
    )
    title = f'Error Time Series: High-Fidelity vs JPL Horizons - {object_name_display}'
    fig_err_ts.suptitle(title, fontsize=14)
    filename = f'error_timeseries_cart_coe_mee_high_fidelity_rel_jpl_horizons_{name_lower}.png'
    fig_err_ts.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    error_files['hf_vs_horizons'].append(filename)
    plt.close(fig_err_ts)

  # High-Fidelity Relative To SGP4 (compare at equal grid times)
  if compare_tle and has_high_fidelity and has_sgp4:
    # Use equal grid data (main plot_delta_time, state, coe)
    fig_err_ts = plot_time_series_error(
      result_ref  = result_sgp4_propagation,
      result_comp = result_high_fidelity_propagation,
      epoch       = time_o_dt,
      use_ric     = True,
    )
    title = f'Error Time Series: High-Fidelity vs SGP4 - {object_name_display}'
    fig_err_ts.suptitle(title, fontsize=14)
    filename = f'error_timeseries_high_fidelity_rel_sgp4_{name_lower}.png'
    fig_err_ts.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    error_files['hf_vs_sgp4'].append(filename)
    plt.close(fig_err_ts)

  # SGP4 Relative To JPL Horizons (compare at ephemeris times)
  if compare_jpl_horizons and compare_tle and has_horizons and has_sgp4_at_ephem:
    # Build result dict for comparison using pre-computed at_ephem_times data
    sgp4_at_ephem = result_sgp4_propagation.at_ephem_times
    
    fig_err_ts = plot_time_series_error(
      result_ref  = result_jpl_horizons_ephemeris,
      result_comp = sgp4_at_ephem,
      epoch       = time_o_dt,
      use_ric     = True,
    )
    title = f'RIC Errors: SGP4 vs JPL Horizons - {object_name_display}'
    fig_err_ts.suptitle(title, fontsize=14)
    filename = f'error_timeseries_sgp4_rel_jpl_horizons_{name_lower}.png'
    fig_err_ts.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    error_files['sgp4_vs_horizons'].append(filename)
    plt.close(fig_err_ts)
  
  return error_files


def generate_3d_and_time_series_plots(
  result_jpl_horizons_ephemeris    : Optional[PropagationResult],
  result_high_fidelity_propagation : PropagationResult,
  result_sgp4_propagation          : Optional[PropagationResult],
  time_o_dt                        : datetime,
  figures_folderpath               : Path,
  compare_jpl_horizons             : bool,
  compare_tle                      : bool,
  object_name                      : str = "object",
  object_name_display              : str = "Object",
  trackers                         : Optional[list['TrackerStation']] = None,
  include_tracker_on_body          : bool = False,
) -> dict:
  """
  Generate and save 3D trajectory and time series plots.
  
  Input:
  ------
    result_horizons : PropagationResult | None
      Horizons ephemeris result.
    result_high_fidelity : PropagationResult
      High-fidelity propagation result.
    result_sgp4 : PropagationResult | None
      SGP4 propagation result.
    time_o_dt : datetime
      Simulation start time (for plot labels).
    figures_folderpath : Path
      Directory to save plots.
    compare_jpl_horizons : bool
      Flag to enable comparison with Horizons.
    compare_tle : bool
      Flag to enable comparison with TLE/SGP4.
    object_name : str
      Sanitized name of the object for filenames.
    object_name_display : str
      Original name of the object for plot titles.
      
  Output:
  -------
    dict : Dictionary with plot filenames organized by source type.
  """
  plot_files = {
    'jpl_horizons': {},
    'high_fidelity': {},
    'sgp4': {},
  }

  # Lowercase name for filenames
  name_lower = object_name.lower()

  # Horizons plots
  if compare_jpl_horizons and result_jpl_horizons_ephemeris and result_jpl_horizons_ephemeris.success:
    fig1 = plot_3d_trajectories(result_jpl_horizons_ephemeris, epoch=time_o_dt, frame="J2000")
    fig1.suptitle(f'3D Inertial - {object_name_display} - JPL Horizons', fontsize=16)
    filename = f'3d_j2000_earth_centered_jpl_horizons_{name_lower}.png'
    fig1.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    plot_files['jpl_horizons']['3d_inertial'] = filename
    plt.close(fig1)

    fig2 = plot_time_series(result_jpl_horizons_ephemeris, epoch=time_o_dt)
    fig2.suptitle(f'Time Series - {object_name_display} - JPL Horizons', fontsize=16)
    filename = f'timeseries_cart_coe_mee_jpl_horizons_{name_lower}.png'
    fig2.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    plot_files['jpl_horizons']['time_series'] = filename
    plt.close(fig2)

    # Body-fixed 3D plot for Horizons
    fig_ef = plot_3d_trajectories_body_fixed(result_jpl_horizons_ephemeris, epoch_dt_utc=time_o_dt, trackers=trackers, include_tracker_on_body=include_tracker_on_body)
    fig_ef.suptitle(f'3D Body-Fixed - {object_name_display} - JPL Horizons', fontsize=16)
    filename = f'3d_iau_earth_jpl_horizons_{name_lower}.png'
    fig_ef.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    plot_files['jpl_horizons']['3d_body_fixed'] = filename
    plt.close(fig_ef)

    # Ground track plot for Horizons
    gt_title = f'Ground Track - {object_name_display} - JPL Horizons'
    fig_gt = plot_ground_track(result_jpl_horizons_ephemeris, epoch_dt_utc=time_o_dt, title_text=gt_title, trackers=trackers, include_tracker_on_body=include_tracker_on_body)
    filename = f'groundtrack_jpl_horizons_{name_lower}.png'
    fig_gt.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    plot_files['jpl_horizons']['ground_track'] = filename
    plt.close(fig_gt)
  
  # High-fidelity plots
  if result_high_fidelity_propagation.success:
    fig3 = plot_3d_trajectories(result_high_fidelity_propagation, epoch=time_o_dt, frame="J2000")
    fig3.suptitle(f'3D Inertial - {object_name_display} - High-Fidelity', fontsize=16)
    filename = f'3d_j2000_earth_centered_high_fidelity_{name_lower}.png'
    fig3.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    plot_files['high_fidelity']['3d_inertial'] = filename
    plt.close(fig3)

    fig4 = plot_time_series(result_high_fidelity_propagation, epoch=time_o_dt)
    fig4.suptitle(f'Time Series - {object_name_display} - High-Fidelity', fontsize=16)
    filename = f'timeseries_cart_coe_mee_high_fidelity_{name_lower}.png'
    fig4.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    plot_files['high_fidelity']['time_series'] = filename
    plt.close(fig4)

    # Body-fixed 3D plot
    fig_ef = plot_3d_trajectories_body_fixed(result_high_fidelity_propagation, epoch_dt_utc=time_o_dt, trackers=trackers, include_tracker_on_body=include_tracker_on_body)
    fig_ef.suptitle(f'3D Body-Fixed - {object_name_display} - High-Fidelity', fontsize=16)
    filename = f'3d_iau_earth_high_fidelity_{name_lower}.png'
    fig_ef.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    plot_files['high_fidelity']['3d_body_fixed'] = filename
    plt.close(fig_ef)

    # Ground track plot
    gt_title = f'Ground Track - {object_name_display} - High-Fidelity'
    fig_gt = plot_ground_track(result_high_fidelity_propagation, epoch_dt_utc=time_o_dt, title_text=gt_title, trackers=trackers, include_tracker_on_body=include_tracker_on_body)
    filename = f'groundtrack_high_fidelity_{name_lower}.png'
    fig_gt.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    plot_files['high_fidelity']['ground_track'] = filename
    plt.close(fig_gt)

    # 3D plot Sun-centered trajectory
    fig_moon = plot_3d_trajectory_sun_centered(result_high_fidelity_propagation, epoch=time_o_dt)
    fig_moon.suptitle(f'3D Inertial - {object_name_display} - High-Fidelity', fontsize=16)
    filename = f'3d_j2000_sun_centered_high_fidelity_{name_lower}.png'
    fig_moon.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    plot_files['high_fidelity']['3d_sun_centered'] = filename
    plt.close(fig_moon)
  
  # SGP4 plots
  if compare_tle and result_sgp4_propagation and result_sgp4_propagation.success:
    # 3D trajectory plot
    fig_sgp4_3d = plot_3d_trajectories(result_sgp4_propagation, epoch=time_o_dt, frame="J2000")
    fig_sgp4_3d.suptitle(f'3D Inertial - {object_name_display} - SGP4', fontsize=16)
    filename = f'3d_j2000_earth_centered_sgp4_{name_lower}.png'
    fig_sgp4_3d.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    plot_files['sgp4']['3d_inertial'] = filename
    plt.close(fig_sgp4_3d)
    
    # Time series plot
    fig_sgp4_ts = plot_time_series(result_sgp4_propagation, epoch=time_o_dt)
    fig_sgp4_ts.suptitle(f'Time Series - {object_name_display} - SGP4', fontsize=16)
    filename = f'timeseries_cart_coe_mee_sgp4_{name_lower}.png'
    fig_sgp4_ts.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    plot_files['sgp4']['time_series'] = filename
    plt.close(fig_sgp4_ts)

    # Ground track plot for SGP4
    gt_title = f'Ground Track - {object_name_display} - SGP4'
    fig_gt_sgp4 = plot_ground_track(result_sgp4_propagation, epoch_dt_utc=time_o_dt, title_text=gt_title, trackers=trackers, include_tracker_on_body=include_tracker_on_body)
    filename = f'groundtrack_sgp4_{name_lower}.png'
    fig_gt_sgp4.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    plot_files['sgp4']['ground_track'] = filename
    plt.close(fig_gt_sgp4)

  return plot_files


def generate_plots(
  result_jpl_horizons_ephemeris    : Optional[PropagationResult],
  result_high_fidelity_propagation : PropagationResult,
  result_sgp4_propagation          : Optional[PropagationResult],
  time_o_dt                        : datetime,
  figures_folderpath               : Path,
  compare_jpl_horizons             : bool = False,
  compare_tle                      : bool = False,
  object_name                      : str  = "object",
  object_name_display              : str  = "Object",
  trackers                         : Optional[list['TrackerStation']] = None,
  include_tracker_on_body          : bool = False,
  od_covariances                   : Optional['np.ndarray'] = None,
  od_estimation_times              : Optional['np.ndarray'] = None,
  od_measurement_times             : Optional['np.ndarray'] = None,
  include_orbit_determination      : bool = False,
) -> None:
  """
  Generate and save all simulation plots.

  Input:
  ------
    result_horizons : PropagationResult | None
      Horizons ephemeris result.
    result_high_fidelity : PropagationResult
      High-fidelity propagation result.
    result_sgp4 : PropagationResult | None
      SGP4 propagation result.
    time_o_dt : datetime
      Simulation start time (for plot labels).
    figures_folderpath : Path
      Directory to save plots.
    compare_jpl_horizons : bool
      Flag to enable comparison with Horizons.
    compare_tle : bool
      Flag to enable comparison with TLE/SGP4.
    object_name : str
      Sanitized name of the object for filenames.
    object_name_display : str
      Original name of the object for plot titles.
    tracker : TrackerStation | None
      Tracker station object with normalized azimuth values.
    include_tracker_on_body : bool
      Flag to show tracker location on ground track and 3D body-fixed plots.

  Output:
  -------
    None
  """
  title = "Generate and Save Plots"
  print("\n" + "-" * len(title))
  print(title)
  print("-" * len(title))

  print()
  print("  Progress")

  # Generate 3D and time series plots
  print("    Generate 3D-trajectory, time-series, and groundtrack plots")
  plot_files = generate_3d_and_time_series_plots(
    result_jpl_horizons_ephemeris    = result_jpl_horizons_ephemeris,
    result_high_fidelity_propagation = result_high_fidelity_propagation,
    result_sgp4_propagation          = result_sgp4_propagation,
    time_o_dt                        = time_o_dt,
    figures_folderpath               = figures_folderpath,
    compare_jpl_horizons             = compare_jpl_horizons,
    compare_tle                      = compare_tle,
    object_name                      = object_name,
    object_name_display              = object_name_display,
    trackers                         = trackers,
    include_tracker_on_body          = include_tracker_on_body,
  )

  # Generate error plots only if a comparison was requested
  error_files = {'hf_vs_horizons': [], 'hf_vs_sgp4': [], 'sgp4_vs_horizons': []}
  if compare_jpl_horizons or compare_tle:
    print("    Generate error plots")
    error_files = generate_error_plots(
      result_jpl_horizons_ephemeris    = result_jpl_horizons_ephemeris,
      result_high_fidelity_propagation = result_high_fidelity_propagation,
      result_sgp4_propagation          = result_sgp4_propagation,
      time_o_dt                        = time_o_dt,
      figures_folderpath               = figures_folderpath,
      compare_jpl_horizons             = compare_jpl_horizons,
      compare_tle                      = compare_tle,
      object_name                      = object_name,
      object_name_display              = object_name_display,
    )

  # Generate skyplots for each tracker
  skyplot_files = {}  # Dict of tracker_name -> list of filenames
  if trackers is not None and len(trackers) > 0:
    print("    Generate skyplots")

    name_lower = object_name.lower().replace(' ', '_').replace('-', '_')

    for tracker in trackers:
      try:
        tracker_name_sanitized = tracker.name.lower().replace(' ', '_').replace('-', '_')

        # Collect filenames for this tracker
        filenames = []

        # Generate skyplot for high-fidelity propagation
        if result_high_fidelity_propagation.success:
          skyplot_title = f'Skyplot - {object_name_display} - High-Fidelity - {tracker.name}'
          fig_skyplot = plot_skyplot(
            result       = result_high_fidelity_propagation,
            tracker      = tracker,
            epoch_dt_utc = time_o_dt,
            title_text   = skyplot_title,
          )
          filename = f'skyplot_{tracker_name_sanitized}_high_fidelity_{name_lower}.png'
          fig_skyplot.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
          plt.close(fig_skyplot)
          filenames.append(filename)

        # Generate skyplot for SGP4 if available
        if compare_tle and result_sgp4_propagation and result_sgp4_propagation.success:
          skyplot_title = f'Skyplot - {object_name_display} - SGP4 - {tracker.name}'
          fig_skyplot_sgp4 = plot_skyplot(
            result       = result_sgp4_propagation,
            tracker      = tracker,
            epoch_dt_utc = time_o_dt,
            title_text   = skyplot_title,
          )
          filename = f'skyplot_{tracker_name_sanitized}_sgp4_{name_lower}.png'
          fig_skyplot_sgp4.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
          plt.close(fig_skyplot_sgp4)
          filenames.append(filename)

        # Generate skyplot for JPL Horizons if available
        if compare_jpl_horizons and result_jpl_horizons_ephemeris and result_jpl_horizons_ephemeris.success:
          # Generate simulated measurements with uncertainty for JPL Horizons
          simulator = MeasurementSimulator(result_jpl_horizons_ephemeris, tracker, time_o_dt)
          noise_config = simulator.get_tracker_noise_config()
          measurements = simulator.simulate(noise_config=noise_config, seed=42, include_rates=True)

          # Generate regular skyplot
          skyplot_title = f'Skyplot - {object_name_display} - JPL Horizons - {tracker.name}'
          fig_skyplot_horizons = plot_skyplot(
            result       = result_jpl_horizons_ephemeris,
            tracker      = tracker,
            epoch_dt_utc = time_o_dt,
            title_text   = skyplot_title,
            measurements = measurements,
          )
          filename = f'skyplot_{tracker_name_sanitized}_jpl_horizons_{name_lower}.png'
          fig_skyplot_horizons.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
          plt.close(fig_skyplot_horizons)
          filenames.append(filename)

          # Generate error skyplot
          error_skyplot_title = f'Measurement Errors - {object_name_display} - JPL Horizons - {tracker.name}'
          fig_error_skyplot = plot_measurement_errors(
            measurements = measurements,
            tracker      = tracker,
            epoch_dt_utc = time_o_dt,
            title_text   = error_skyplot_title,
          )
          error_filename = f'error_skyplot_{tracker_name_sanitized}_jpl_horizons_{name_lower}.png'
          fig_error_skyplot.savefig(figures_folderpath / error_filename, dpi=300, bbox_inches='tight')
          plt.close(fig_error_skyplot)
          filenames.append(error_filename)

        skyplot_files[tracker.name] = filenames

      except Exception as e:
        print(f"      [WARNING] Failed to generate skyplot for {tracker.name}: {e}")

  # Generate error skyplot for each tracker (Measured vs Truth from JPL Horizons)
  error_skyplot_files = {}  # Dict of tracker_name -> list of filenames
  if trackers is not None and len(trackers) > 0 and compare_jpl_horizons:
    has_horizons = result_jpl_horizons_ephemeris is not None and result_jpl_horizons_ephemeris.success

    if has_horizons:
      print("    Generate error skyplots (Measured vs Truth)")

      name_lower = object_name.lower().replace(' ', '_').replace('-', '_')

      for tracker in trackers:
        try:
          tracker_name_sanitized = tracker.name.lower().replace(' ', '_').replace('-', '_')

          # Generate simulated measurements with noise for JPL Horizons
          simulator = MeasurementSimulator(result_jpl_horizons_ephemeris, tracker, time_o_dt)
          noise_config = simulator.get_tracker_noise_config()
          measurements = simulator.simulate(noise_config=noise_config, seed=42, include_rates=True)

          # Generate error skyplot (measured - truth)
          error_title = f'Measurement Error - {object_name_display} - {tracker.name}'
          fig_error = plot_error_skyplot(
            measurements = measurements,
            epoch_dt_utc = time_o_dt,
            title_text   = error_title,
          )
          if fig_error is not None:
            filename = f'error_skyplot_{tracker_name_sanitized}_jpl_horizons_meas_rel_truth_{name_lower}.png'
            fig_error.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
            plt.close(fig_error)
            error_skyplot_files[tracker.name] = [filename]

        except Exception as e:
          import traceback
          print(f"      [WARNING] Failed to generate error skyplot for {tracker.name}: {e}")
          traceback.print_exc()

  # Generate pass time-series plots for each tracker
  pass_timeseries_files = {}  # Dict of tracker_name -> list of filenames
  if trackers is not None and len(trackers) > 0:
    print("    Generate pass time-series plots")

    name_lower = object_name.lower().replace(' ', '_').replace('-', '_')

    for tracker in trackers:
      try:
        tracker_name_sanitized = tracker.name.lower().replace(' ', '_').replace('-', '_')

        # Collect filenames for this tracker
        filenames = []

        # Generate pass time-series for high-fidelity propagation
        if result_high_fidelity_propagation.success:
          pass_ts_title = f'Pass Time Series - {object_name_display} - High-Fidelity - {tracker.name}'
          fig_pass_ts = plot_pass_timeseries(
            result       = result_high_fidelity_propagation,
            tracker      = tracker,
            epoch_dt_utc = time_o_dt,
            title_text   = pass_ts_title,
          )
          if fig_pass_ts is not None:
            filename = f'time_series_meas_{tracker_name_sanitized}_high_fidelity_{name_lower}.png'
            fig_pass_ts.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
            plt.close(fig_pass_ts)
            filenames.append(filename)

        # Generate pass time-series for SGP4 if available
        if compare_tle and result_sgp4_propagation and result_sgp4_propagation.success:
          pass_ts_title = f'Pass Time Series - {object_name_display} - SGP4 - {tracker.name}'
          fig_pass_ts_sgp4 = plot_pass_timeseries(
            result       = result_sgp4_propagation,
            tracker      = tracker,
            epoch_dt_utc = time_o_dt,
            title_text   = pass_ts_title,
          )
          if fig_pass_ts_sgp4 is not None:
            filename = f'time_series_meas_{tracker_name_sanitized}_sgp4_{name_lower}.png'
            fig_pass_ts_sgp4.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
            plt.close(fig_pass_ts_sgp4)
            filenames.append(filename)

        # Generate pass time-series for JPL Horizons if available
        if compare_jpl_horizons and result_jpl_horizons_ephemeris and result_jpl_horizons_ephemeris.success:
          pass_ts_title = f'Pass Time Series - {object_name_display} - JPL Horizons - {tracker.name}'
          fig_pass_ts_horizons = plot_pass_timeseries(
            result       = result_jpl_horizons_ephemeris,
            tracker      = tracker,
            epoch_dt_utc = time_o_dt,
            title_text   = pass_ts_title,
          )
          if fig_pass_ts_horizons is not None:
            filename = f'time_series_meas_{tracker_name_sanitized}_jpl_horizons_{name_lower}.png'
            fig_pass_ts_horizons.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
            plt.close(fig_pass_ts_horizons)
            filenames.append(filename)

        pass_timeseries_files[tracker.name] = filenames

      except Exception as e:
        print(f"      [WARNING] Failed to generate pass time-series for {tracker.name}: {e}")

  # Generate covariance plots if orbit determination was performed
  covariance_files = []
  if include_orbit_determination and od_covariances is not None and od_estimation_times is not None:
    print("    Generate covariance plots")

    name_lower = object_name.lower().replace(' ', '_').replace('-', '_')

    try:
      # Use the dedicated covariance time array (includes pre-update and post-update times for sawtooth)
      delta_time_epoch = od_estimation_times

      # Plot 1: Covariance time series (RSS uncertainties with 1-sigma and 3-sigma)
      cov_ts_title = f'State Uncertainty Evolution - {object_name_display} - Orbit Determination'
      fig_cov_ts = plot_covariance_timeseries(
        covariances       = od_covariances,
        delta_time_epoch  = delta_time_epoch,
        title_text        = cov_ts_title,
        measurement_times = od_measurement_times,
      )
      filename = f'covariance_timeseries_od_{name_lower}.png'
      fig_cov_ts.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
      plt.close(fig_cov_ts)
      covariance_files.append(filename)

      # Plot 2: Covariance components (individual x,y,z and vx,vy,vz)
      cov_comp_title = f'State Uncertainty Components - {object_name_display} - Orbit Determination'
      fig_cov_comp = plot_covariance_components(
        covariances       = od_covariances,
        delta_time_epoch  = delta_time_epoch,
        title_text        = cov_comp_title,
        measurement_times = od_measurement_times,
      )
      filename = f'covariance_components_od_{name_lower}.png'
      fig_cov_comp.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
      plt.close(fig_cov_comp)
      covariance_files.append(filename)

    except Exception as e:
      print(f"      [WARNING] Failed to generate covariance plots: {e}")

  print()
  print("  Summary")
  print(f"    Figure Folderpath : {figures_folderpath}")
  print()
  
  # Print 3D-Trajectory, Time-Series, and Groundtrack Plots
  print("    3D-Trajectory, Time-Series, and Groundtrack Plots")
  
  if plot_files['jpl_horizons']:
    print("      JPL-Horizons-Ephemeris Plots")
    if '3d_inertial' in plot_files['jpl_horizons']:
      print(f"        3D Inertial    : <figures_folderpath>/{plot_files['jpl_horizons']['3d_inertial']}")
    if 'time_series' in plot_files['jpl_horizons']:
      print(f"        Time Series    : <figures_folderpath>/{plot_files['jpl_horizons']['time_series']}")
    if '3d_body_fixed' in plot_files['jpl_horizons']:
      print(f"        3D Body-Fixed  : <figures_folderpath>/{plot_files['jpl_horizons']['3d_body_fixed']}")
    if 'ground_track' in plot_files['jpl_horizons']:
      print(f"        Ground Track   : <figures_folderpath>/{plot_files['jpl_horizons']['ground_track']}")
  
  if plot_files['high_fidelity']:
    print("      High-Fidelity-Model Plots")
    if '3d_inertial' in plot_files['high_fidelity']:
      print(f"        3D Inertial    : <figures_folderpath>/{plot_files['high_fidelity']['3d_inertial']}")
    if 'time_series' in plot_files['high_fidelity']:
      print(f"        Time Series    : <figures_folderpath>/{plot_files['high_fidelity']['time_series']}")
    if '3d_body_fixed' in plot_files['high_fidelity']:
      print(f"        3D Body-Fixed  : <figures_folderpath>/{plot_files['high_fidelity']['3d_body_fixed']}")
    if 'ground_track' in plot_files['high_fidelity']:
      print(f"        Ground Track   : <figures_folderpath>/{plot_files['high_fidelity']['ground_track']}")
    if '3d_sun_centered' in plot_files['high_fidelity']:
      print(f"        3D Sun-Centered: <figures_folderpath>/{plot_files['high_fidelity']['3d_sun_centered']}")
  
  if plot_files['sgp4']:
    print("      SGP4-Model Plots")
    if '3d_inertial' in plot_files['sgp4']:
      print(f"        3D Inertial    : <figures_folderpath>/{plot_files['sgp4']['3d_inertial']}")
    if 'time_series' in plot_files['sgp4']:
      print(f"        Time Series    : <figures_folderpath>/{plot_files['sgp4']['time_series']}")
    if 'ground_track' in plot_files['sgp4']:
      print(f"        Ground Track   : <figures_folderpath>/{plot_files['sgp4']['ground_track']}")
  
  # Print Error Plots
  has_error_plots = any(error_files[k] for k in error_files)
  if has_error_plots:
    print()
    print("    Error Cartesian-COE-MEE Plots")
    if error_files['hf_vs_horizons']:
      print("      High-Fidelity Relative To JPL Horizons")
      for filename in error_files['hf_vs_horizons']:
        print(f"        Time-Series Error : <figures_folderpath>/{filename}")
    if error_files['hf_vs_sgp4']:
      print("      High-Fidelity Relative To SGP4")
      for filename in error_files['hf_vs_sgp4']:
        print(f"        Time-Series Error : <figures_folderpath>/{filename}")
    if error_files['sgp4_vs_horizons']:
      print("      SGP4 Relative To JPL Horizons")
      for filename in error_files['sgp4_vs_horizons']:
        print(f"        Time-Series Error : <figures_folderpath>/{filename}")
  
  # Print Skyplots
  if skyplot_files:
    print()
    print("    Skyplots")
    for tracker_name, filenames in skyplot_files.items():
      print(f"      Tracker {tracker_name}")
      for filename in filenames:
        print(f"        <figures_folderpath>/{filename}")

  # Print Error Skyplots
  if error_skyplot_files:
    print()
    print("    Error Skyplots")
    for tracker_name, filenames in error_skyplot_files.items():
      print(f"      Tracker {tracker_name}")
      for filename in filenames:
        print(f"        <figures_folderpath>/{filename}")

  # Print Pass Time-Series
  if pass_timeseries_files:
    print()
    print("    Pass Time-Series")
    for tracker_name, filenames in pass_timeseries_files.items():
      if filenames:
        print(f"      Tracker {tracker_name}")
        for filename in filenames:
          print(f"        <figures_folderpath>/{filename}")

  # Print Covariance Plots
  if covariance_files:
    print()
    print("    Covariance Plots (Orbit Determination)")
    for filename in covariance_files:
      print(f"      <figures_folderpath>/{filename}")
