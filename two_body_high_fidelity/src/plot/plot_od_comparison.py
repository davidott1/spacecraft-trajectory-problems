"""
Orbit Determination Comparison Plotting
========================================

Functions for visualizing filter vs smoother performance in orbit determination.
"""
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from datetime import datetime
from typing import Optional

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
