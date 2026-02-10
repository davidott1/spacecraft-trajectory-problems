"""
Orbit Determination Comparison Plotting
========================================

Functions for visualizing filter vs smoother performance in orbit determination.
"""
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from datetime import datetime
from typing import Optional
from scipy import stats

from src.schemas.propagation import PropagationResult
from src.model.frame_converter import FrameConverter
from src.model.constants import CONVERTER


def plot_filter_smoother_error_comparison(
  truth_result     : PropagationResult,
  filter_result    : PropagationResult,
  smoother_result  : PropagationResult,
  epoch            : datetime,
  title_text       : str = "Filter vs Smoother Error",
  use_ric          : bool = True,
) -> Figure:
  """
  Plot filter and smoother errors relative to truth with RSS and components.

  Creates a 1x2 grid showing:
    Left: Position errors (RSS and R/I/C components) for filter and smoother
    Right: Velocity errors (RSS and R/I/C components) for filter and smoother

  Input:
  ------
    truth_result : PropagationResult
      Truth trajectory (e.g., JPL Horizons).
    filter_result : PropagationResult
      Forward-filtered EKF estimates.
    smoother_result : PropagationResult
      Backward-smoothed RTS estimates.
    epoch : datetime
      Reference epoch for time axis.
    title_text : str
      Title for the figure.
    use_ric : bool
      If True, compute errors in RIC frame. If False, use XYZ inertial.

  Output:
  -------
    fig : Figure
      Matplotlib figure containing error comparison plots.
  """
  # Extract time and states
  time_truth = truth_result.time_grid.deltas
  time_filter = filter_result.time_grid.deltas
  time_smoother = smoother_result.time_grid.deltas

  # Verify time grids match
  if not (np.allclose(time_truth, time_filter) and np.allclose(time_truth, time_smoother)):
    raise ValueError("Time grids don't match between truth, filter, and smoother!")

  time = time_truth / 60.0  # Convert to minutes

  state_truth = truth_result.state
  state_filter = filter_result.state
  state_smoother = smoother_result.state

  n_points = len(time_truth)

  if use_ric:
    # Compute RIC frame errors
    pos_error_filter = np.zeros((3, n_points))
    vel_error_filter = np.zeros((3, n_points))
    pos_error_smoother = np.zeros((3, n_points))
    vel_error_smoother = np.zeros((3, n_points))

    for i in range(n_points):
      # Reference position and velocity (truth)
      ref_pos = state_truth[0:3, i]
      ref_vel = state_truth[3:6, i]

      # Rotation matrix from inertial to RIC
      R_inertial_to_ric = FrameConverter.xyz_to_ric(ref_pos, ref_vel)

      # Filter errors in inertial frame
      pos_error_filter_inertial = state_filter[0:3, i] - ref_pos
      vel_error_filter_inertial = state_filter[3:6, i] - ref_vel

      # Smoother errors in inertial frame
      pos_error_smoother_inertial = state_smoother[0:3, i] - ref_pos
      vel_error_smoother_inertial = state_smoother[3:6, i] - ref_vel

      # Transform to RIC
      pos_error_filter[:, i] = R_inertial_to_ric @ pos_error_filter_inertial
      vel_error_filter[:, i] = R_inertial_to_ric @ vel_error_filter_inertial
      pos_error_smoother[:, i] = R_inertial_to_ric @ pos_error_smoother_inertial
      vel_error_smoother[:, i] = R_inertial_to_ric @ vel_error_smoother_inertial

    component_labels = ['R', 'I', 'C']
  else:
    # Use XYZ inertial frame errors
    pos_error_filter = state_filter[0:3, :] - state_truth[0:3, :]
    vel_error_filter = state_filter[3:6, :] - state_truth[3:6, :]
    pos_error_smoother = state_smoother[0:3, :] - state_truth[0:3, :]
    vel_error_smoother = state_smoother[3:6, :] - state_truth[3:6, :]
    component_labels = ['X', 'Y', 'Z']

  # Compute RSS magnitudes
  pos_rss_filter = np.linalg.norm(pos_error_filter, axis=0)
  vel_rss_filter = np.linalg.norm(vel_error_filter, axis=0)
  pos_rss_smoother = np.linalg.norm(pos_error_smoother, axis=0)
  vel_rss_smoother = np.linalg.norm(vel_error_smoother, axis=0)

  # Create figure with 1x2 grid
  fig, axes = plt.subplots(1, 2, figsize=(16, 6))

  # Component colors
  colors = ['r', 'g', 'b']

  # Left: Position errors (RSS + components)
  ax1 = axes[0]
  # RSS lines (thick)
  ax1.semilogy(time, pos_rss_filter, 'k-', linewidth=3.0, alpha=0.6, label='Mag Filter')
  ax1.semilogy(time, pos_rss_smoother, 'k--', linewidth=3.0, alpha=0.6, label='Mag Smoother')
  # Component lines (thin) - absolute values for log scale
  for i in range(3):
    ax1.semilogy(time, np.abs(pos_error_filter[i, :]), color=colors[i], linestyle='-',
                 linewidth=1.2, alpha=0.5, label=f'{component_labels[i]} Filter')
    ax1.semilogy(time, np.abs(pos_error_smoother[i, :]), color=colors[i], linestyle='--',
                 linewidth=1.2, alpha=0.5, label=f'{component_labels[i]} Smoother')
  ax1.set_xlabel('Time [min]', fontsize=12)
  ax1.set_ylabel('Position Error [m]', fontsize=12)
  ax1.set_title('Position Error vs Time', fontsize=13, fontweight='bold')
  ax1.grid(True, which='both', linestyle=':', alpha=0.5)
  ax1.legend(loc='best', fontsize=9, ncol=2)

  # Right: Velocity errors (RSS + components)
  ax2 = axes[1]
  # RSS lines (thick)
  ax2.semilogy(time, vel_rss_filter, 'k-', linewidth=3.0, alpha=0.6, label='Mag Filter')
  ax2.semilogy(time, vel_rss_smoother, 'k--', linewidth=3.0, alpha=0.6, label='Mag Smoother')
  # Component lines (thin) - absolute values for log scale
  for i in range(3):
    ax2.semilogy(time, np.abs(vel_error_filter[i, :]), color=colors[i], linestyle='-',
                 linewidth=1.2, alpha=0.5, label=f'{component_labels[i]} Filter')
    ax2.semilogy(time, np.abs(vel_error_smoother[i, :]), color=colors[i], linestyle='--',
                 linewidth=1.2, alpha=0.5, label=f'{component_labels[i]} Smoother')
  ax2.set_xlabel('Time [min]', fontsize=12)
  ax2.set_ylabel('Velocity Error [m/s]', fontsize=12)
  ax2.set_title('Velocity Error vs Time', fontsize=13, fontweight='bold')
  ax2.grid(True, which='both', linestyle=':', alpha=0.5)
  ax2.legend(loc='best', fontsize=9, ncol=2)

  # Overall title
  fig.suptitle(title_text, fontsize=14, fontweight='bold')
  fig.tight_layout(rect=[0, 0, 1, 0.96])

  return fig


# Keep for backwards compatibility but mark as deprecated
def plot_filter_smoother_rss_comparison(
  truth_result    : PropagationResult,
  filter_result   : PropagationResult,
  smoother_result : PropagationResult,
  epoch           : datetime,
  title_text      : str = "Filter vs Smoother RSS Error",
  use_ric         : bool = True,
) -> Figure:
  """
  DEPRECATED: Use plot_filter_smoother_error_comparison instead.
  This function now just calls the combined version.
  """
  return plot_filter_smoother_error_comparison(
    truth_result    = truth_result,
    filter_result   = filter_result,
    smoother_result = smoother_result,
    epoch           = epoch,
    title_text      = title_text,
    use_ric         = use_ric,
  )


def plot_filter_smoother_full_error_comparison(
  truth_result     : PropagationResult,
  filter_result    : PropagationResult,
  smoother_result  : PropagationResult,
  epoch            : datetime,
  title_text       : str = "Filter vs Smoother Error",
  use_ric          : bool = True,
) -> Figure:
  """
  Plot comprehensive filter and smoother errors (cart, COE, MEE) relative to truth.

  Shows magnitude-only lines for cart (filter=solid, smoother=dashed) and
  all COE and MEE error components vs time, matching the format of plot_time_series_error.

  Input:
  ------
    truth_result : PropagationResult
      Truth trajectory (e.g., JPL Horizons).
    filter_result : PropagationResult
      Forward-filtered EKF estimates.
    smoother_result : PropagationResult
      Backward-smoothed RTS estimates.
    epoch : datetime
      Reference epoch for time axis.
    title_text : str
      Title for the figure.
    use_ric : bool
      If True, compute errors in RIC frame. If False, use XYZ inertial.

  Output:
  -------
    fig : Figure
      Matplotlib figure containing comprehensive error comparison plots.
  """
  # Extract time and states
  time_truth = truth_result.time_grid.deltas
  time_filter = filter_result.time_grid.deltas
  time_smoother = smoother_result.time_grid.deltas

  # Verify time grids match
  if not (np.allclose(time_truth, time_filter) and np.allclose(time_truth, time_smoother)):
    raise ValueError("Time grids don't match between truth, filter, and smoother!")

  time = time_truth  # Time in seconds

  state_truth = truth_result.state
  state_filter = filter_result.state
  state_smoother = smoother_result.state
  coe_truth = truth_result.coe
  coe_filter = filter_result.coe
  coe_smoother = smoother_result.coe
  mee_truth = truth_result.mee
  mee_filter = filter_result.mee
  mee_smoother = smoother_result.mee

  n_points = len(time_truth)

  if use_ric:
    # Compute RIC frame errors
    pos_error_filter = np.zeros((3, n_points))
    vel_error_filter = np.zeros((3, n_points))
    pos_error_smoother = np.zeros((3, n_points))
    vel_error_smoother = np.zeros((3, n_points))

    for i in range(n_points):
      ref_pos = state_truth[0:3, i]
      ref_vel = state_truth[3:6, i]
      R_inertial_to_ric = FrameConverter.xyz_to_ric(ref_pos, ref_vel)

      pos_error_filter_inertial = state_filter[0:3, i] - ref_pos
      vel_error_filter_inertial = state_filter[3:6, i] - ref_vel
      pos_error_smoother_inertial = state_smoother[0:3, i] - ref_pos
      vel_error_smoother_inertial = state_smoother[3:6, i] - ref_vel

      pos_error_filter[:, i] = R_inertial_to_ric @ pos_error_filter_inertial
      vel_error_filter[:, i] = R_inertial_to_ric @ vel_error_filter_inertial
      pos_error_smoother[:, i] = R_inertial_to_ric @ pos_error_smoother_inertial
      vel_error_smoother[:, i] = R_inertial_to_ric @ vel_error_smoother_inertial

    pos_ylabel = 'Position Error (RIC)\n[m]'
    vel_ylabel = 'Velocity Error (RIC)\n[m/s]'
  else:
    pos_error_filter = state_filter[0:3, :] - state_truth[0:3, :]
    vel_error_filter = state_filter[3:6, :] - state_truth[3:6, :]
    pos_error_smoother = state_smoother[0:3, :] - state_truth[0:3, :]
    vel_error_smoother = state_smoother[3:6, :] - state_truth[3:6, :]
    pos_ylabel = 'Position Error (XYZ)\n[m]'
    vel_ylabel = 'Velocity Error (XYZ)\n[m/s]'

  # Compute magnitudes
  pos_mag_filter = np.linalg.norm(pos_error_filter, axis=0)
  vel_mag_filter = np.linalg.norm(vel_error_filter, axis=0)
  pos_mag_smoother = np.linalg.norm(pos_error_smoother, axis=0)
  vel_mag_smoother = np.linalg.norm(vel_error_smoother, axis=0)

  # Create figure with 6x3 grid
  fig = plt.figure(figsize=(24, 10))

  # LEFT COLUMN: Position and Velocity Magnitude Errors Only
  # Position error magnitude (rows 0-2, column 0)
  ax_pos = plt.subplot2grid((6, 3), (0, 0), rowspan=3)
  ax_pos.plot(time, pos_mag_filter, 'k-', label='Mag Filter', linewidth=2)
  ax_pos.plot(time, pos_mag_smoother, 'k--', label='Mag Smoother', linewidth=2)
  ax_pos.tick_params(labelbottom=False)
  ax_pos.set_ylabel(pos_ylabel)
  ax_pos.legend()
  ax_pos.grid(True)
  ax_pos.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Velocity error magnitude (rows 3-5, column 0)
  ax_vel = plt.subplot2grid((6, 3), (3, 0), rowspan=3, sharex=ax_pos)
  ax_vel.plot(time, vel_mag_filter, 'k-', label='Mag Filter', linewidth=2)
  ax_vel.plot(time, vel_mag_smoother, 'k--', label='Mag Smoother', linewidth=2)
  ax_vel.set_xlabel('Time\n[s]')
  ax_vel.set_ylabel(vel_ylabel)
  ax_vel.legend()
  ax_vel.grid(True)
  ax_vel.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # MIDDLE COLUMN: Classical Orbital Elements Errors
  # SMA error (row 0, column 1)
  ax_sma = plt.subplot2grid((6, 3), (0, 1), sharex=ax_pos)
  if coe_truth.sma is not None and coe_filter.sma is not None and coe_smoother.sma is not None:
    sma_error_filter = coe_filter.sma - coe_truth.sma
    sma_error_smoother = coe_smoother.sma - coe_truth.sma
    ax_sma.plot(time, sma_error_filter, 'b-', label='Filter', linewidth=1.5)
    ax_sma.plot(time, sma_error_smoother, 'b--', label='Smoother', linewidth=1.5)
  ax_sma.tick_params(labelbottom=False)
  ax_sma.set_ylabel('SMA Error\n[m]')
  ax_sma.legend(fontsize=8)
  ax_sma.grid(True)
  ax_sma.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # ECC error (row 1, column 1)
  ax_ecc = plt.subplot2grid((6, 3), (1, 1), sharex=ax_pos)
  if coe_truth.ecc is not None and coe_filter.ecc is not None and coe_smoother.ecc is not None:
    ecc_error_filter = coe_filter.ecc - coe_truth.ecc
    ecc_error_smoother = coe_smoother.ecc - coe_truth.ecc
    ax_ecc.plot(time, ecc_error_filter, 'b-', label='Filter', linewidth=1.5)
    ax_ecc.plot(time, ecc_error_smoother, 'b--', label='Smoother', linewidth=1.5)
  ax_ecc.tick_params(labelbottom=False)
  ax_ecc.set_ylabel('ECC Error\n[-]')
  ax_ecc.legend(fontsize=8)
  ax_ecc.grid(True)
  ax_ecc.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # INC error (row 2, column 1)
  ax_inc = plt.subplot2grid((6, 3), (2, 1), sharex=ax_pos)
  if coe_truth.inc is not None and coe_filter.inc is not None and coe_smoother.inc is not None:
    inc_error_filter = (coe_filter.inc - coe_truth.inc) * CONVERTER.DEG_PER_RAD
    inc_error_smoother = (coe_smoother.inc - coe_truth.inc) * CONVERTER.DEG_PER_RAD
    ax_inc.plot(time, inc_error_filter, 'b-', label='Filter', linewidth=1.5)
    ax_inc.plot(time, inc_error_smoother, 'b--', label='Smoother', linewidth=1.5)
  ax_inc.tick_params(labelbottom=False)
  ax_inc.set_ylabel('INC Error\n[deg]')
  ax_inc.legend(fontsize=8)
  ax_inc.grid(True)
  ax_inc.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # RAAN error (row 3, column 1)
  ax_raan = plt.subplot2grid((6, 3), (3, 1), sharex=ax_pos)
  if coe_truth.raan is not None and coe_filter.raan is not None and coe_smoother.raan is not None:
    raan_error_filter = (coe_filter.raan - coe_truth.raan) * CONVERTER.DEG_PER_RAD
    raan_error_smoother = (coe_smoother.raan - coe_truth.raan) * CONVERTER.DEG_PER_RAD
    ax_raan.plot(time, raan_error_filter, 'b-', label='Filter', linewidth=1.5)
    ax_raan.plot(time, raan_error_smoother, 'b--', label='Smoother', linewidth=1.5)
  ax_raan.tick_params(labelbottom=False)
  ax_raan.set_ylabel('RAAN Error\n[deg]')
  ax_raan.legend(fontsize=8)
  ax_raan.grid(True)
  ax_raan.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # AOP error (row 4, column 1)
  ax_aop = plt.subplot2grid((6, 3), (4, 1), sharex=ax_pos)
  if coe_truth.aop is not None and coe_filter.aop is not None and coe_smoother.aop is not None:
    aop_error_filter = (coe_filter.aop - coe_truth.aop) * CONVERTER.DEG_PER_RAD
    aop_error_smoother = (coe_smoother.aop - coe_truth.aop) * CONVERTER.DEG_PER_RAD
    ax_aop.plot(time, aop_error_filter, 'b-', label='Filter', linewidth=1.5)
    ax_aop.plot(time, aop_error_smoother, 'b--', label='Smoother', linewidth=1.5)
  ax_aop.tick_params(labelbottom=False)
  ax_aop.set_ylabel('AOP Error\n[deg]')
  ax_aop.legend(fontsize=8)
  ax_aop.grid(True)
  ax_aop.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # TA error (row 5, column 1)
  ax_ta = plt.subplot2grid((6, 3), (5, 1), sharex=ax_pos)
  if coe_truth.ta is not None and coe_filter.ta is not None and coe_smoother.ta is not None:
    ta_error_filter = (coe_filter.ta - coe_truth.ta) * CONVERTER.DEG_PER_RAD
    ta_error_smoother = (coe_smoother.ta - coe_truth.ta) * CONVERTER.DEG_PER_RAD
    ax_ta.plot(time, ta_error_filter, 'b-', label='Filter', linewidth=1.5)
    ax_ta.plot(time, ta_error_smoother, 'b--', label='Smoother', linewidth=1.5)
  ax_ta.set_xlabel('Time\n[s]')
  ax_ta.set_ylabel('TA Error\n[deg]')
  ax_ta.legend(fontsize=8)
  ax_ta.grid(True)
  ax_ta.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # RIGHT COLUMN: Modified Equinoctial Elements Errors
  # p error (row 0, column 2)
  ax_p = plt.subplot2grid((6, 3), (0, 2), sharex=ax_pos)
  if mee_truth.p is not None and mee_filter.p is not None and mee_smoother.p is not None:
    p_error_filter = mee_filter.p - mee_truth.p
    p_error_smoother = mee_smoother.p - mee_truth.p
    ax_p.plot(time, p_error_filter, 'b-', label='Filter', linewidth=1.5)
    ax_p.plot(time, p_error_smoother, 'b--', label='Smoother', linewidth=1.5)
  ax_p.tick_params(labelbottom=False)
  ax_p.set_ylabel('p Error\n[m]')
  ax_p.legend(fontsize=8)
  ax_p.grid(True)
  ax_p.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # f error (row 1, column 2)
  ax_f = plt.subplot2grid((6, 3), (1, 2), sharex=ax_pos)
  if mee_truth.f is not None and mee_filter.f is not None and mee_smoother.f is not None:
    f_error_filter = mee_filter.f - mee_truth.f
    f_error_smoother = mee_smoother.f - mee_truth.f
    ax_f.plot(time, f_error_filter, 'b-', label='Filter', linewidth=1.5)
    ax_f.plot(time, f_error_smoother, 'b--', label='Smoother', linewidth=1.5)
  ax_f.tick_params(labelbottom=False)
  ax_f.set_ylabel('f Error\n[-]')
  ax_f.legend(fontsize=8)
  ax_f.grid(True)
  ax_f.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # g error (row 2, column 2)
  ax_g = plt.subplot2grid((6, 3), (2, 2), sharex=ax_pos)
  if mee_truth.g is not None and mee_filter.g is not None and mee_smoother.g is not None:
    g_error_filter = mee_filter.g - mee_truth.g
    g_error_smoother = mee_smoother.g - mee_truth.g
    ax_g.plot(time, g_error_filter, 'b-', label='Filter', linewidth=1.5)
    ax_g.plot(time, g_error_smoother, 'b--', label='Smoother', linewidth=1.5)
  ax_g.tick_params(labelbottom=False)
  ax_g.set_ylabel('g Error\n[-]')
  ax_g.legend(fontsize=8)
  ax_g.grid(True)
  ax_g.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # h error (row 3, column 2)
  ax_h = plt.subplot2grid((6, 3), (3, 2), sharex=ax_pos)
  if mee_truth.h is not None and mee_filter.h is not None and mee_smoother.h is not None:
    h_error_filter = mee_filter.h - mee_truth.h
    h_error_smoother = mee_smoother.h - mee_truth.h
    ax_h.plot(time, h_error_filter, 'b-', label='Filter', linewidth=1.5)
    ax_h.plot(time, h_error_smoother, 'b--', label='Smoother', linewidth=1.5)
  ax_h.tick_params(labelbottom=False)
  ax_h.set_ylabel('h Error\n[-]')
  ax_h.legend(fontsize=8)
  ax_h.grid(True)
  ax_h.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # k error (row 4, column 2)
  ax_k = plt.subplot2grid((6, 3), (4, 2), sharex=ax_pos)
  if mee_truth.k is not None and mee_filter.k is not None and mee_smoother.k is not None:
    k_error_filter = mee_filter.k - mee_truth.k
    k_error_smoother = mee_smoother.k - mee_truth.k
    ax_k.plot(time, k_error_filter, 'b-', label='Filter', linewidth=1.5)
    ax_k.plot(time, k_error_smoother, 'b--', label='Smoother', linewidth=1.5)
  ax_k.tick_params(labelbottom=False)
  ax_k.set_ylabel('k Error\n[-]')
  ax_k.legend(fontsize=8)
  ax_k.grid(True)
  ax_k.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # L error (row 5, column 2)
  ax_L = plt.subplot2grid((6, 3), (5, 2), sharex=ax_pos)
  if mee_truth.L is not None and mee_filter.L is not None and mee_smoother.L is not None:
    L_error_filter = (mee_filter.L - mee_truth.L) * CONVERTER.DEG_PER_RAD
    L_error_smoother = (mee_smoother.L - mee_truth.L) * CONVERTER.DEG_PER_RAD
    ax_L.plot(time, L_error_filter, 'b-', label='Filter', linewidth=1.5)
    ax_L.plot(time, L_error_smoother, 'b--', label='Smoother', linewidth=1.5)
  ax_L.set_xlabel('Time\n[s]')
  ax_L.set_ylabel('L Error\n[deg]')
  ax_L.legend(fontsize=8)
  ax_L.grid(True)
  ax_L.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Overall title
  fig.suptitle(title_text, fontsize=14, fontweight='bold')
  fig.tight_layout(rect=[0, 0, 1, 0.96])

  return fig


def plot_mcreynolds_consistency(
  filter_result        : PropagationResult,
  smoother_result      : PropagationResult,
  filter_covariances   : np.ndarray,
  smoother_covariances : np.ndarray,
  epoch                : datetime,
  truth_result         : Optional[PropagationResult] = None,
  title_text           : str = "McReynolds Filter/Smoother Consistency Test",
) -> Figure:
  """
  Plot McReynolds consistency metric in RIC frame.

  The McReynolds test computes:
    χ = (x̂_filter - x̂_smoother) / sqrt(P_filter - P_smoother)

  for each state component in the RIC (Radial-In-track-Cross-track) frame.
  Under the assumption that both filter and smoother are consistent, these
  normalized residuals should behave like zero-mean Gaussian random variables
  with unit variance.

  Input:
  ------
    filter_result : PropagationResult
      Forward-filtered EKF estimates.
    smoother_result : PropagationResult
      Backward-smoothed RTS estimates.
    filter_covariances : np.ndarray (6, 6, N)
      Forward-filtered covariance matrices.
    smoother_covariances : np.ndarray (6, 6, N)
      Backward-smoothed covariance matrices.
    epoch : datetime
      Reference epoch for time axis.
    truth_result : PropagationResult, optional
      Truth trajectory for RIC frame computation. If None, uses filter trajectory.
    title_text : str
      Title for the figure.

  Output:
  -------
    fig : Figure
      Matplotlib figure with 2x1 grid showing position and velocity consistency.
  """
  # Extract time and states
  time_filter = filter_result.time_grid.deltas
  time_smoother = smoother_result.time_grid.deltas

  # Verify time grids match
  if not np.allclose(time_filter, time_smoother):
    raise ValueError("Time grids don't match between filter and smoother!")

  time = time_filter / 60.0  # Convert to minutes

  state_filter = filter_result.state
  state_smoother = smoother_result.state

  # Check dimensions first
  n_cov_filter = filter_covariances.shape[2]
  n_cov_smoother = smoother_covariances.shape[2]
  n_state_filter = state_filter.shape[1]
  n_state_smoother = state_smoother.shape[1]

  # Use minimum to avoid index errors
  n_all = min(len(time_filter), n_cov_filter, n_cov_smoother, n_state_filter, n_state_smoother)

  # Filter out pre-update (predicted) entries at measurement times.
  # The EKF stores [pre-update, post-update] at the same time for measurement times.
  # Pre-update has P_predicted (inflated), which makes P_f - P_s artificially large.
  # Keep only post-update entries (second of each duplicate pair) and non-measurement entries.
  keep_mask = np.ones(n_all, dtype=bool)
  for i in range(n_all - 1):
    if np.abs(time_filter[i + 1] - time_filter[i]) < 1e-6:
      # This is a pre-update entry (next entry is post-update at same time)
      keep_mask[i] = False

  keep_indices = np.where(keep_mask)[0]
  n_points = len(keep_indices)
  n_filtered = n_all - n_points
  print(f"    [DEBUG] McReynolds: n_all={n_all}, n_filtered={n_filtered}, n_points={n_points}")

  # Diagnostic: compute McReynolds in INERTIAL frame (no RIC) for debugging
  chi_inertial_diag = np.zeros((6, n_points))
  for i in range(n_points):
    idx = keep_indices[i]
    dx = state_filter[:, idx] - state_smoother[:, idx]
    dP = np.diag(filter_covariances[:, :, idx] - smoother_covariances[:, :, idx])
    dP_safe = np.abs(dP) + 1e-20
    chi_inertial_diag[:, i] = dx / np.sqrt(dP_safe)
  for j in range(6):
    labels_6 = ['x','y','z','vx','vy','vz']
    m = np.mean(chi_inertial_diag[j,:])
    s = np.std(chi_inertial_diag[j,:])
    print(f"    [DEBUG] Inertial {labels_6[j]}: mean={m:.4f}, std={s:.4f}")
  # Also print sample P_f, P_s, diff at midpoint
  mid = keep_indices[n_points // 2]
  Pf_diag = np.diag(filter_covariances[:, :, mid])
  Ps_diag = np.diag(smoother_covariances[:, :, mid])
  print(f"    [DEBUG] Mid-point Pf diag: {Pf_diag}")
  print(f"    [DEBUG] Mid-point Ps diag: {Ps_diag}")
  print(f"    [DEBUG] Mid-point Pf-Ps  : {Pf_diag - Ps_diag}")
  dx_mid = state_filter[:, mid] - state_smoother[:, mid]
  print(f"    [DEBUG] Mid-point dx     : {dx_mid}")
  print(f"    [DEBUG] Mid-point dx^2   : {dx_mid**2}")

  # Remap time to use only kept indices
  time = time[keep_indices]

  # Use truth for RIC frame if provided, otherwise use filter
  # Make sure truth has same number of points
  if truth_result is not None:
    ref_states = truth_result.state
    # If truth has different size, use filter instead
    if ref_states.shape[1] < n_all:
      ref_states = state_filter
  else:
    ref_states = state_filter

  # Compute consistency metric in RIC frame
  chi_pos = np.zeros((3, n_points))
  chi_vel = np.zeros((3, n_points))
  pos_var_diff_hist = np.zeros((3, n_points))
  vel_var_diff_hist = np.zeros((3, n_points))

  for i in range(n_points):
    # Map to original index (skipping pre-update entries)
    idx = keep_indices[i]

    # Reference position and velocity for RIC frame
    ref_pos = ref_states[0:3, idx]
    ref_vel = ref_states[3:6, idx]

    # Rotation matrix from inertial to RIC
    R_inertial_to_ric = FrameConverter.xyz_to_ric(ref_pos, ref_vel)

    # State differences in inertial frame
    pos_diff_inertial = state_filter[0:3, idx] - state_smoother[0:3, idx]
    vel_diff_inertial = state_filter[3:6, idx] - state_smoother[3:6, idx]

    # Transform state differences to RIC
    pos_diff_ric = R_inertial_to_ric @ pos_diff_inertial
    vel_diff_ric = R_inertial_to_ric @ vel_diff_inertial

    # Covariance differences in inertial frame
    P_diff = filter_covariances[:, :, idx] - smoother_covariances[:, :, idx]

    # Transform covariance difference to RIC frame
    # Build 6x6 rotation matrix: R_6x6 = [[R, 0], [0, R]]
    R_6x6 = np.zeros((6, 6))
    R_6x6[0:3, 0:3] = R_inertial_to_ric
    R_6x6[3:6, 3:6] = R_inertial_to_ric

    # P_ric = R_6x6 @ P_inertial @ R_6x6^T
    P_diff_ric = R_6x6 @ P_diff @ R_6x6.T

    # Extract diagonal elements (variances)
    pos_var_diff = np.diag(P_diff_ric[0:3, 0:3])
    vel_var_diff = np.diag(P_diff_ric[3:6, 3:6])

    pos_var_diff_hist[:, i] = pos_var_diff
    vel_var_diff_hist[:, i] = vel_var_diff

    # Ensure positive variance difference (use absolute value to avoid NaN)
    pos_var_diff_safe = np.abs(pos_var_diff)
    vel_var_diff_safe = np.abs(vel_var_diff)

    # Avoid division by zero
    epsilon = 1e-20

    # Compute consistency metric
    chi_pos[:, i] = pos_diff_ric / np.sqrt(pos_var_diff_safe + epsilon)
    chi_vel[:, i] = vel_diff_ric / np.sqrt(vel_var_diff_safe + epsilon)

  # Create figure with GridSpec for timeseries (left) and histograms (right)
  fig = plt.figure(figsize=(16, 10))
  gs = GridSpec(2, 2, figure=fig, width_ratios=[3, 1], hspace=0.15, wspace=0.05)

  # Create axes
  ax_pos = fig.add_subplot(gs[0, 0])
  ax_pos_hist = fig.add_subplot(gs[0, 1], sharey=ax_pos)
  ax_vel = fig.add_subplot(gs[1, 0])
  ax_vel_hist = fig.add_subplot(gs[1, 1], sharey=ax_vel)

  # Colors for R, I, C components
  colors = ['r', 'g', 'b']
  labels = ['R (Radial)', 'I (In-track)', 'C (Cross-track)']

  # ===== TOP PLOT: Position consistency timeseries =====
  for i in range(3):
    ax_pos.plot(time, chi_pos[i, :], color=colors[i], linewidth=1.5, alpha=0.8, label=labels[i])

  # Mark non-positive variance differences (Pf - Ps <= 0)
  for i in range(3):
    bad_idx = np.where(pos_var_diff_hist[i, :] <= 0.0)[0]
    if bad_idx.size > 0:
      ax_pos.scatter(time[bad_idx], chi_pos[i, bad_idx], s=18, marker='x', color='k', alpha=0.7, zorder=5)

  # Reference lines
  ax_pos.axhline(y=0, color='k', linestyle='-', linewidth=1.2, alpha=0.7)
  ax_pos.axhline(y=1, color='gray', linestyle='--', linewidth=1.0, alpha=0.6)
  ax_pos.axhline(y=-1, color='gray', linestyle='--', linewidth=1.0, alpha=0.6)
  ax_pos.axhline(y=2, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)
  ax_pos.axhline(y=-2, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)
  ax_pos.axhline(y=3, color='gray', linestyle='-.', linewidth=1.0, alpha=0.4)
  ax_pos.axhline(y=-3, color='gray', linestyle='-.', linewidth=1.0, alpha=0.4)

  ax_pos.set_ylabel('Position Consistency χ [-]', fontsize=12)
  ax_pos.set_yscale('symlog', linthresh=3)
  ax_pos.grid(True, linestyle=':', alpha=0.5)
  ax_pos.legend(loc='best', fontsize=10)
  # Set ylim to show full data range with some padding
  pos_max = np.max(np.abs(chi_pos)) * 1.1
  ax_pos.set_ylim(-pos_max, pos_max)
  # Custom yticks: linear region (-3 to 3) and log powers of 10
  linear_ticks = [-3, -2, -1, 0, 1, 2, 3]
  log_ticks = []
  log_labels = []
  for exp in range(1, 10):
    val = 10**exp
    if val < pos_max:
      log_ticks.extend([-val, val])
      log_labels.extend([f'$-10^{exp}$', f'$10^{exp}$'])
  all_ticks = sorted(linear_ticks + log_ticks)
  all_labels = [str(t) if t in linear_ticks else (f'$-10^{{{int(np.log10(abs(t)))}}}$' if t < 0 else f'$10^{{{int(np.log10(t))}}}$') for t in all_ticks]
  ax_pos.set_yticks(all_ticks)
  ax_pos.set_yticklabels(all_labels)
  ax_pos.tick_params(labelbottom=False)

  # ===== TOP HISTOGRAM: Position consistency distribution =====
  # Combine all position components for histogram
  chi_pos_all = chi_pos.flatten()
  def _symlog_bins(max_abs: float, linthresh: float = 3.0, n_linear: int = 21, n_log: int = 5) -> np.ndarray:
    max_abs = max(max_abs, linthresh)
    linear_edges = np.linspace(-linthresh, linthresh, n_linear)
    if max_abs <= linthresh:
      return linear_edges
    log_edges = np.logspace(np.log10(linthresh), np.log10(max_abs), n_log)
    neg_edges = -log_edges[::-1]
    return np.concatenate([neg_edges, linear_edges[1:-1], log_edges])

  pos_bins = _symlog_bins(np.max(np.abs(chi_pos_all)))
  ax_pos_hist.hist(chi_pos_all, bins=pos_bins, orientation='horizontal', density=True,
                   alpha=0.5, color='steelblue', edgecolor='black', linewidth=0.5)

  # Overlay theoretical normal distribution N(0,1)
  y_range = np.linspace(-5, 5, 100)
  normal_pdf = stats.norm.pdf(y_range, loc=0, scale=1)
  ax_pos_hist.plot(normal_pdf, y_range, 'b-', linewidth=2, label='N(0,1)')

  # Add statistics text
  mean_val = np.mean(chi_pos_all)
  std_val = np.std(chi_pos_all)
  print(f"    [DEBUG] McReynolds Position Consistency: mean={mean_val:.2f}, std={std_val:.2f}")
  ax_pos_hist.text(0.95, 0.95, f'μ={mean_val:.2f}\nσ={std_val:.2f}',
                   transform=ax_pos_hist.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

  ax_pos_hist.set_xlabel('Density', fontsize=10)
  ax_pos_hist.grid(True, alpha=0.3, axis='x')
  ax_pos_hist.legend(loc='lower right', fontsize=9)
  ax_pos_hist.tick_params(labelleft=False)

  # ===== BOTTOM PLOT: Velocity consistency timeseries =====
  for i in range(3):
    ax_vel.plot(time, chi_vel[i, :], color=colors[i], linewidth=1.5, alpha=0.8, label=labels[i])

  # Mark non-positive variance differences (Pf - Ps <= 0)
  for i in range(3):
    bad_idx = np.where(vel_var_diff_hist[i, :] <= 0.0)[0]
    if bad_idx.size > 0:
      ax_vel.scatter(time[bad_idx], chi_vel[i, bad_idx], s=18, marker='x', color='k', alpha=0.7, zorder=5)

  # Reference lines
  ax_vel.axhline(y=0, color='k', linestyle='-', linewidth=1.2, alpha=0.7)
  ax_vel.axhline(y=1, color='gray', linestyle='--', linewidth=1.0, alpha=0.6)
  ax_vel.axhline(y=-1, color='gray', linestyle='--', linewidth=1.0, alpha=0.6)
  ax_vel.axhline(y=2, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)
  ax_vel.axhline(y=-2, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)
  ax_vel.axhline(y=3, color='gray', linestyle='-.', linewidth=1.0, alpha=0.4)
  ax_vel.axhline(y=-3, color='gray', linestyle='-.', linewidth=1.0, alpha=0.4)

  ax_vel.set_xlabel('Time [min]', fontsize=12)
  ax_vel.set_ylabel('Velocity Consistency χ [-]', fontsize=12)
  ax_vel.set_yscale('symlog', linthresh=3)
  ax_vel.grid(True, linestyle=':', alpha=0.5)
  ax_vel.legend(loc='best', fontsize=10)
  # Set ylim to show full data range with some padding
  vel_max = np.max(np.abs(chi_vel)) * 1.1
  ax_vel.set_ylim(-vel_max, vel_max)
  # Custom yticks: linear region (-3 to 3) and log powers of 10
  linear_ticks = [-3, -2, -1, 0, 1, 2, 3]
  log_ticks = []
  for exp in range(1, 10):
    val = 10**exp
    if val < vel_max:
      log_ticks.extend([-val, val])
  all_ticks = sorted(linear_ticks + log_ticks)
  all_labels = [str(t) if t in linear_ticks else (f'$-10^{{{int(np.log10(abs(t)))}}}$' if t < 0 else f'$10^{{{int(np.log10(t))}}}$') for t in all_ticks]
  ax_vel.set_yticks(all_ticks)
  ax_vel.set_yticklabels(all_labels)

  # ===== BOTTOM HISTOGRAM: Velocity consistency distribution =====
  # Combine all velocity components for histogram
  chi_vel_all = chi_vel.flatten()
  vel_bins = _symlog_bins(np.max(np.abs(chi_vel_all)))
  ax_vel_hist.hist(chi_vel_all, bins=vel_bins, orientation='horizontal', density=True,
                   alpha=0.5, color='steelblue', edgecolor='black', linewidth=0.5)

  # Overlay theoretical normal distribution N(0,1)
  ax_vel_hist.plot(normal_pdf, y_range, 'b-', linewidth=2, label='N(0,1)')

  # Add statistics text
  mean_val = np.mean(chi_vel_all)
  std_val = np.std(chi_vel_all)
  ax_vel_hist.text(0.95, 0.95, f'μ={mean_val:.2f}\nσ={std_val:.2f}',
                   transform=ax_vel_hist.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

  ax_vel_hist.set_xlabel('Density', fontsize=10)
  ax_vel_hist.grid(True, alpha=0.3, axis='x')
  ax_vel_hist.legend(loc='lower right', fontsize=9)
  ax_vel_hist.tick_params(labelleft=False)

  # Overall title
  fig.suptitle(title_text, fontsize=14, fontweight='bold')
  fig.tight_layout(rect=[0, 0, 1, 0.97])

  return fig
