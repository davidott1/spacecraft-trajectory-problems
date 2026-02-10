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
from scipy    import interpolate

from src.plot.plot_3d                          import plot_3d_trajectories, plot_3d_trajectories_body_fixed, plot_3d_trajectory_sun_centered
from src.plot.plot_timeseries                  import plot_time_series, plot_time_series_error
from src.plot.plot_groundtrack                 import plot_ground_track
from src.plot.plot_skyplot                     import plot_skyplot, plot_pass_timeseries, plot_measurement_errors, plot_error_skyplot
from src.plot.plot_covariance                  import plot_covariance_combined, plot_covariance_filter_vs_smoother
from src.plot.plot_od_comparison               import plot_filter_smoother_error_comparison, plot_filter_smoother_rss_comparison, plot_filter_smoother_full_error_comparison, plot_mcreynolds_consistency
from src.plot.plot_residual_ratio              import plot_measurement_residual_ratio, plot_innovation_covariance_evolution
from src.schemas.propagation                   import PropagationResult, TimeGrid
from src.schemas.state                         import TrackerStation, ClassicalOrbitalElements, ModifiedEquinoctialElements
from src.orbit_determination.measurement_simulator import MeasurementSimulator
from src.model.orbit_converter                 import OrbitConverter
from src.model.constants                       import SOLARSYSTEMCONSTANTS


def _interpolate_result_to_times(source_result: PropagationResult, target_times: np.ndarray) -> PropagationResult:
  """
  Interpolate a PropagationResult to a new time grid.

  Input:
  ------
    source_result : PropagationResult
      The source result with states and time_grid to interpolate from.
    target_times : np.ndarray
      The target times (in seconds from epoch) to interpolate to.

  Output:
  -------
    interpolated_result : PropagationResult
      A new PropagationResult with states interpolated to target_times.
  """
  source_times = source_result.time_grid.deltas
  source_states = source_result.state

  # Create interpolators for each state component (6 states)
  interpolated_states = np.zeros((6, len(target_times)))
  for i in range(6):
    interp_func = interpolate.interp1d(
      source_times, source_states[i, :],
      kind='cubic', fill_value='extrapolate'
    )
    interpolated_states[i, :] = interp_func(target_times)

  # Compute orbital elements for interpolated states
  n_times = len(target_times)
  coe_sma  = np.zeros(n_times)
  coe_ecc  = np.zeros(n_times)
  coe_inc  = np.zeros(n_times)
  coe_raan = np.zeros(n_times)
  coe_aop  = np.zeros(n_times)
  coe_ta   = np.zeros(n_times)
  coe_ea   = np.zeros(n_times)
  coe_ma   = np.zeros(n_times)
  mee_p    = np.zeros(n_times)
  mee_f    = np.zeros(n_times)
  mee_g    = np.zeros(n_times)
  mee_h    = np.zeros(n_times)
  mee_k    = np.zeros(n_times)
  mee_L    = np.zeros(n_times)

  for i in range(n_times):
    coe = OrbitConverter.pv_to_coe(
      interpolated_states[0:3, i] * 1000.0,  # km -> m
      interpolated_states[3:6, i] * 1000.0,  # km/s -> m/s
      gp = SOLARSYSTEMCONSTANTS.EARTH.GP,
    )
    coe_sma[i]  = coe.sma  if coe.sma  is not None else 0.0
    coe_ecc[i]  = coe.ecc  if coe.ecc  is not None else 0.0
    coe_inc[i]  = coe.inc  if coe.inc  is not None else 0.0
    coe_raan[i] = coe.raan if coe.raan is not None else 0.0
    coe_aop[i]  = coe.aop  if coe.aop  is not None else 0.0
    coe_ta[i]   = coe.ta   if coe.ta   is not None else 0.0
    coe_ea[i]   = coe.ea   if coe.ea   is not None else 0.0
    coe_ma[i]   = coe.ma   if coe.ma   is not None else 0.0

    mee = OrbitConverter.pv_to_mee(
      interpolated_states[0:3, i] * 1000.0,
      interpolated_states[3:6, i] * 1000.0,
      gp = SOLARSYSTEMCONSTANTS.EARTH.GP,
    )
    mee_p[i] = mee.p
    mee_f[i] = mee.f
    mee_g[i] = mee.g
    mee_h[i] = mee.h
    mee_k[i] = mee.k
    mee_L[i] = mee.L

  coe_time_series = ClassicalOrbitalElements(
    sma=coe_sma, ecc=coe_ecc, inc=coe_inc, raan=coe_raan,
    aop=coe_aop, ma=coe_ma, ta=coe_ta, ea=coe_ea,
  )
  mee_time_series = ModifiedEquinoctialElements(
    p=mee_p, f=mee_f, g=mee_g, h=mee_h, k=mee_k, L=mee_L,
  )

  # Create new time grid using source's initial time and computing final from target deltas
  from datetime import timedelta
  initial_time = source_result.time_grid.initial
  final_time = initial_time + timedelta(seconds=float(target_times[-1]))
  new_time_grid = TimeGrid(
    initial=initial_time,
    final=final_time,
    deltas=target_times,
  )

  return PropagationResult(
    success=True,
    state=interpolated_states,
    time_grid=new_time_grid,
    coe=coe_time_series,
    mee=mee_time_series,
    at_ephem_times=None,
  )


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

    # Check if time grids match, if not, interpolate to JPL Horizons grid
    jpl_times = result_jpl_horizons_ephemeris.time_grid.deltas
    hf_times = hf_at_ephem.time_grid.deltas

    if len(hf_times) != len(jpl_times) or not np.allclose(hf_times, jpl_times):
      # Time grids don't match - interpolate HF to JPL grid
      print(f"      Interpolating HF model to JPL Horizons time grid ({len(hf_times)} -> {len(jpl_times)} points)")
      hf_at_ephem = _interpolate_result_to_times(hf_at_ephem, jpl_times)
    
    try:
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
    except Exception as e:
      print(f"      [WARNING] HF vs Horizons error plot failed: {e}")

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
  include_orbit_determination      : bool = False,
  od_smoother_states               : Optional[PropagationResult] = None,
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

    # Smoother time series (if orbit determination is enabled)
    if include_orbit_determination and od_smoother_states is not None:
      fig_smoother = plot_time_series(od_smoother_states, epoch=time_o_dt)
      fig_smoother.suptitle(f'Time Series - {object_name_display} - High-Fidelity Smoother', fontsize=16)
      filename_smoother = f'timeseries_cart_coe_mee_high_fidelity_smoother_{name_lower}.png'
      fig_smoother.savefig(figures_folderpath / filename_smoother, dpi=300, bbox_inches='tight')
      plot_files['high_fidelity']['time_series_smoother'] = filename_smoother
      plt.close(fig_smoother)

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
  od_filter_states                 : Optional[PropagationResult] = None,
  od_filter_covariances            : Optional['np.ndarray'] = None,
  od_smoother_states               : Optional[PropagationResult] = None,
  od_smoother_covariances          : Optional['np.ndarray'] = None,
  od_estimation_times              : Optional['np.ndarray'] = None,
  od_measurement_times             : Optional['np.ndarray'] = None,
  od_residual_data                 : Optional[dict] = None,
  od_propagator                    : Optional[object] = None,
  od_process_noise                 : Optional['np.ndarray'] = None,
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
    include_orbit_determination      = include_orbit_determination,
    od_smoother_states               = od_smoother_states,
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
            filename = f'timeseries_meas_{tracker_name_sanitized}_high_fidelity_{name_lower}.png'
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
            filename = f'timeseries_meas_{tracker_name_sanitized}_sgp4_{name_lower}.png'
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
            filename = f'timeseries_meas_{tracker_name_sanitized}_jpl_horizons_{name_lower}.png'
            fig_pass_ts_horizons.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
            plt.close(fig_pass_ts_horizons)
            filenames.append(filename)

        pass_timeseries_files[tracker.name] = filenames

      except Exception as e:
        print(f"      [WARNING] Failed to generate pass time-series for {tracker.name}: {e}")

  # Generate measurement residual ratio plots if orbit determination was performed
  residual_ratio_files = []
  if include_orbit_determination and od_residual_data is not None:
    print("    Generate measurement residual ratio plots")

    name_lower = object_name.lower().replace(' ', '_').replace('-', '_')

    try:
      residuals = od_residual_data.get('residuals', [])
      innovation_covariances = od_residual_data.get('innovation_covariances', [])
      residual_times = od_residual_data.get('measurement_times', np.array([]))

      if len(residuals) > 0:
        # Generate residual ratio plot
        residual_title = f'Measurement Residual Ratio - {object_name_display} - EKF'
        fig_residual = plot_measurement_residual_ratio(
          residuals              = residuals,
          innovation_covariances = innovation_covariances,
          measurement_times      = residual_times,
          title_text             = residual_title,
        )
        filename_residual = f'timeseries_residual_ratio_{name_lower}.png'
        fig_residual.savefig(figures_folderpath / filename_residual, dpi=300, bbox_inches='tight')
        plt.close(fig_residual)
        residual_ratio_files.append(filename_residual)

        # Also plot innovation covariance evolution to diagnose filter behavior
        innov_cov_title = f'Innovation Covariance Evolution - {object_name_display}'
        fig_innov_cov = plot_innovation_covariance_evolution(
          innovation_covariances = od_residual_data['innovation_covariances'],
          measurement_times      = od_residual_data['measurement_times'],
          title_text             = innov_cov_title,
        )
        filename_innov_cov = f'timeseries_innovation_covariance_{name_lower}.png'
        fig_innov_cov.savefig(figures_folderpath / filename_innov_cov, dpi=300, bbox_inches='tight')
        plt.close(fig_innov_cov)
        residual_ratio_files.append(filename_innov_cov)

    except Exception as e:
      print(f"      [WARNING] Failed to generate residual ratio plots: {e}")

  # Generate covariance plots if orbit determination was performed
  covariance_files = []
  od_comparison_files = []
  if include_orbit_determination and od_filter_covariances is not None and od_estimation_times is not None:
    print("    Generate orbit determination plots")

    name_lower = object_name.lower().replace(' ', '_').replace('-', '_')

    try:
      # Use the dedicated covariance time array (includes pre-update and post-update times for sawtooth)
      delta_time_epoch = od_estimation_times

      # Filter covariance plot
      cov_title_filter = f'Filter Uncertainty - {object_name_display} - EKF'
      fig_cov_filter = plot_covariance_combined(
        covariances       = od_filter_covariances,
        delta_time_epoch  = delta_time_epoch,
        title_text        = cov_title_filter,
        measurement_times = od_measurement_times,
      )
      filename_filter = f'timeseries_cov_filter_{name_lower}.png'
      fig_cov_filter.savefig(figures_folderpath / filename_filter, dpi=300, bbox_inches='tight')
      plt.close(fig_cov_filter)
      covariance_files.append(filename_filter)

      # Smoother covariance plot (if available)
      if od_smoother_covariances is not None:
        cov_title_smoother = f'Smoother Uncertainty - {object_name_display} - RTS'
        fig_cov_smoother = plot_covariance_combined(
          covariances       = od_smoother_covariances,
          delta_time_epoch  = delta_time_epoch,
          title_text        = cov_title_smoother,
          measurement_times = od_measurement_times,
        )
        filename_smoother = f'timeseries_cov_smoother_{name_lower}.png'
        fig_cov_smoother.savefig(figures_folderpath / filename_smoother, dpi=300, bbox_inches='tight')
        plt.close(fig_cov_smoother)
        covariance_files.append(filename_smoother)

        # Filter vs Smoother covariance comparison
        cov_title_comparison = f'Filter vs Smoother Uncertainty - {object_name_display}'
        fig_cov_comp = plot_covariance_filter_vs_smoother(
          filter_covariances   = od_filter_covariances,
          smoother_covariances = od_smoother_covariances,
          delta_time_epoch     = delta_time_epoch,
          title_text           = cov_title_comparison,
          measurement_times    = od_measurement_times,
        )
        filename_comp = f'timeseries_cov_filter_vs_smoother_{name_lower}.png'
        fig_cov_comp.savefig(figures_folderpath / filename_comp, dpi=300, bbox_inches='tight')
        plt.close(fig_cov_comp)
        covariance_files.append(filename_comp)

    except Exception as e:
      print(f"      [WARNING] Failed to generate covariance plots: {e}")

  # Generate filter vs smoother error comparison plots
  if (include_orbit_determination and
      od_filter_states is not None and
      od_smoother_states is not None and
      result_jpl_horizons_ephemeris is not None and
      result_jpl_horizons_ephemeris.success):
    print("    Generate filter vs smoother error comparison plots")

    name_lower = object_name.lower().replace(' ', '_').replace('-', '_')

    # Check if filter and smoother have at_ephem_times
    has_filter_at_ephem = od_filter_states.at_ephem_times is not None
    has_smoother_at_ephem = od_smoother_states.at_ephem_times is not None

    if has_filter_at_ephem and has_smoother_at_ephem:
      # Use at_ephem_times for smooth curves (EXACT same approach as first plot)
      filter_at_ephem = od_filter_states.at_ephem_times
      smoother_at_ephem = od_smoother_states.at_ephem_times

      # Check if time grids match, if not, interpolate to JPL Horizons grid
      jpl_times = result_jpl_horizons_ephemeris.time_grid.deltas
      filter_times = filter_at_ephem.time_grid.deltas

      if len(filter_times) != len(jpl_times) or not np.allclose(filter_times, jpl_times):
        # Time grids don't match - interpolate filter/smoother to JPL grid
        print(f"      Interpolating filter/smoother results to JPL Horizons time grid ({len(filter_times)} -> {len(jpl_times)} points)")
        filter_at_ephem = _interpolate_result_to_times(filter_at_ephem, jpl_times)
        smoother_at_ephem = _interpolate_result_to_times(smoother_at_ephem, jpl_times)

      try:
        # Generate filter/smoother error plot with both solutions overlaid
        fig_err_ts = plot_time_series_error(
          result_ref      = result_jpl_horizons_ephemeris,
          result_comp     = filter_at_ephem,
          epoch           = time_o_dt,
          use_ric         = True,
          result_smoother = smoother_at_ephem,
        )
        title = f'Error Time Series: Filter/Smoother vs JPL Horizons - {object_name_display}'
        fig_err_ts.suptitle(title, fontsize=14)
        filename = f'error_timeseries_cart_coe_mee_high_fidelity_filter_smoother_rel_jpl_horizons_{name_lower}.png'
        fig_err_ts.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
        od_comparison_files.append(filename)
        plt.close(fig_err_ts)
      except Exception as e:
        print(f"      [WARNING] Filter/smoother error plot failed: {e}")

  # Generate McReynolds consistency plot
  consistency_files = []
  if (include_orbit_determination and
      od_filter_states is not None and
      od_smoother_states is not None and
      od_filter_covariances is not None and
      od_smoother_covariances is not None):
    print("    Generate McReynolds consistency plot")

    name_lower = object_name.lower().replace(' ', '_').replace('-', '_')

    try:
      # Generate McReynolds consistency plot
      consistency_title = f'McReynolds Consistency Test - {object_name_display}'
      fig_consistency = plot_mcreynolds_consistency(
        filter_result        = od_filter_states,
        smoother_result      = od_smoother_states,
        filter_covariances   = od_filter_covariances,
        smoother_covariances = od_smoother_covariances,
        epoch                = time_o_dt,
        truth_result         = result_jpl_horizons_ephemeris if (result_jpl_horizons_ephemeris and result_jpl_horizons_ephemeris.success) else None,
        title_text           = consistency_title,
      )
      filename_consistency = f'mcreynolds_consistency_{name_lower}.png'
      fig_consistency.savefig(figures_folderpath / filename_consistency, dpi=300, bbox_inches='tight')
      plt.close(fig_consistency)
      consistency_files.append(filename_consistency)

    except Exception as e:
      print(f"      [WARNING] Failed to generate McReynolds consistency plot: {e}")

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

  # Print Pass Time-Series (only if there are actual files)
  if pass_timeseries_files:
    # Check if there are any actual files to print
    has_files = any(filenames for filenames in pass_timeseries_files.values())
    if has_files:
      print()
      print("    Pass Time-Series")
      for tracker_name, filenames in pass_timeseries_files.items():
        if filenames:
          print(f"      Tracker {tracker_name}")
          for filename in filenames:
            print(f"        <figures_folderpath>/{filename}")

  # Print Measurement Residual Ratio Plots
  if residual_ratio_files:
    print()
    print("    Measurement Residual Ratio Plots (Orbit Determination)")
    for filename in residual_ratio_files:
      print(f"      <figures_folderpath>/{filename}")

  # Print Covariance Plots
  if covariance_files:
    print()
    print("    Covariance Plots (Orbit Determination)")
    for filename in covariance_files:
      print(f"      <figures_folderpath>/{filename}")

  # Print Filter vs Smoother Comparison Plots
  if od_comparison_files:
    print()
    print("    Filter vs Smoother Comparison Plots")
    for filename in od_comparison_files:
      print(f"      <figures_folderpath>/{filename}")

  # Print McReynolds Consistency Plots
  if consistency_files:
    print()
    print("    McReynolds Consistency Plots")
    for filename in consistency_files:
      print(f"      <figures_folderpath>/{filename}")
