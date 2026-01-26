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
