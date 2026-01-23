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
  title_text       : str = "Filter vs Smoother Error Comparison",
  use_ric          : bool = True,
) -> Figure:
  """
  Plot filter and smoother errors relative to truth.

  Creates a 2x3 grid showing:
    Top row: Position errors (RIC or XYZ) for filter and smoother
    Bottom row: Velocity errors (RIC or XYZ) for filter and smoother

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
    pos_error_filter_ric = np.zeros((3, n_points))
    vel_error_filter_ric = np.zeros((3, n_points))
    pos_error_smoother_ric = np.zeros((3, n_points))
    vel_error_smoother_ric = np.zeros((3, n_points))

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
      pos_error_filter_ric[:, i] = R_inertial_to_ric @ pos_error_filter_inertial
      vel_error_filter_ric[:, i] = R_inertial_to_ric @ vel_error_filter_inertial
      pos_error_smoother_ric[:, i] = R_inertial_to_ric @ pos_error_smoother_inertial
      vel_error_smoother_ric[:, i] = R_inertial_to_ric @ vel_error_smoother_inertial

    pos_error_filter = pos_error_filter_ric
    vel_error_filter = vel_error_filter_ric
    pos_error_smoother = pos_error_smoother_ric
    vel_error_smoother = vel_error_smoother_ric
    component_labels = ['Radial', 'In-track', 'Cross-track']
    frame_label = 'RIC'
  else:
    # Use XYZ inertial frame errors
    pos_error_filter = state_filter[0:3, :] - state_truth[0:3, :]
    vel_error_filter = state_filter[3:6, :] - state_truth[3:6, :]
    pos_error_smoother = state_smoother[0:3, :] - state_truth[0:3, :]
    vel_error_smoother = state_smoother[3:6, :] - state_truth[3:6, :]
    component_labels = ['X', 'Y', 'Z']
    frame_label = 'XYZ'

  # Compute error magnitudes
  pos_mag_filter = np.linalg.norm(pos_error_filter, axis=0)
  vel_mag_filter = np.linalg.norm(vel_error_filter, axis=0)
  pos_mag_smoother = np.linalg.norm(pos_error_smoother, axis=0)
  vel_mag_smoother = np.linalg.norm(vel_error_smoother, axis=0)

  # Create figure
  fig, axes = plt.subplots(2, 3, figsize=(20, 10))

  # Top row: Position errors
  colors = ['r', 'g', 'b']
  for i in range(3):
    ax = axes[0, i]
    ax.plot(time, pos_error_filter[i, :], color=colors[i], linestyle='-',
            linewidth=2, label='Filter (EKF)', alpha=0.7)
    ax.plot(time, pos_error_smoother[i, :], color=colors[i], linestyle='--',
            linewidth=2, label='Smoother (RTS)', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Time [min]', fontsize=11)
    ax.set_ylabel(f'{component_labels[i]} Error [m]', fontsize=11)
    ax.set_title(f'Position Error: {component_labels[i]} ({frame_label})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

  # Bottom row: Velocity errors
  for i in range(3):
    ax = axes[1, i]
    ax.plot(time, vel_error_filter[i, :], color=colors[i], linestyle='-',
            linewidth=2, label='Filter (EKF)', alpha=0.7)
    ax.plot(time, vel_error_smoother[i, :], color=colors[i], linestyle='--',
            linewidth=2, label='Smoother (RTS)', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Time [min]', fontsize=11)
    ax.set_ylabel(f'{component_labels[i]} Error [m/s]', fontsize=11)
    ax.set_title(f'Velocity Error: {component_labels[i]} ({frame_label})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

  # Overall title
  fig.suptitle(title_text, fontsize=14, fontweight='bold')
  fig.tight_layout(rect=[0, 0, 1, 0.97])

  return fig


def plot_filter_smoother_rss_comparison(
  truth_result    : PropagationResult,
  filter_result   : PropagationResult,
  smoother_result : PropagationResult,
  epoch           : datetime,
  title_text      : str = "Filter vs Smoother RSS Error",
  use_ric         : bool = True,
) -> Figure:
  """
  Plot RSS (root-sum-square) errors for filter and smoother.

  Creates a 1x2 grid showing:
    Left: Position RSS error vs time
    Right: Velocity RSS error vs time

  Input:
  ------
    truth_result : PropagationResult
      Truth trajectory.
    filter_result : PropagationResult
      Filtered estimates.
    smoother_result : PropagationResult
      Smoothed estimates.
    epoch : datetime
      Reference epoch.
    title_text : str
      Figure title.
    use_ric : bool
      Use RIC frame if True, XYZ if False.

  Output:
  -------
    fig : Figure
      Matplotlib figure.
  """
  # Extract time and states
  time_truth = truth_result.time_grid.deltas
  time = time_truth / 60.0  # Minutes

  state_truth = truth_result.state
  state_filter = filter_result.state
  state_smoother = smoother_result.state

  n_points = len(time_truth)

  if use_ric:
    # Compute RIC errors
    pos_error_filter_ric = np.zeros((3, n_points))
    vel_error_filter_ric = np.zeros((3, n_points))
    pos_error_smoother_ric = np.zeros((3, n_points))
    vel_error_smoother_ric = np.zeros((3, n_points))

    for i in range(n_points):
      ref_pos = state_truth[0:3, i]
      ref_vel = state_truth[3:6, i]
      R_inertial_to_ric = FrameConverter.xyz_to_ric(ref_pos, ref_vel)

      pos_error_filter_ric[:, i] = R_inertial_to_ric @ (state_filter[0:3, i] - ref_pos)
      vel_error_filter_ric[:, i] = R_inertial_to_ric @ (state_filter[3:6, i] - ref_vel)
      pos_error_smoother_ric[:, i] = R_inertial_to_ric @ (state_smoother[0:3, i] - ref_pos)
      vel_error_smoother_ric[:, i] = R_inertial_to_ric @ (state_smoother[3:6, i] - ref_vel)

    pos_mag_filter = np.linalg.norm(pos_error_filter_ric, axis=0)
    vel_mag_filter = np.linalg.norm(vel_error_filter_ric, axis=0)
    pos_mag_smoother = np.linalg.norm(pos_error_smoother_ric, axis=0)
    vel_mag_smoother = np.linalg.norm(vel_error_smoother_ric, axis=0)
  else:
    # XYZ errors
    pos_error_filter = state_filter[0:3, :] - state_truth[0:3, :]
    vel_error_filter = state_filter[3:6, :] - state_truth[3:6, :]
    pos_error_smoother = state_smoother[0:3, :] - state_truth[0:3, :]
    vel_error_smoother = state_smoother[3:6, :] - state_truth[3:6, :]

    pos_mag_filter = np.linalg.norm(pos_error_filter, axis=0)
    vel_mag_filter = np.linalg.norm(vel_error_filter, axis=0)
    pos_mag_smoother = np.linalg.norm(pos_error_smoother, axis=0)
    vel_mag_smoother = np.linalg.norm(vel_error_smoother, axis=0)

  # Create figure
  fig, axes = plt.subplots(1, 2, figsize=(16, 6))

  # Left: Position RSS error
  ax1 = axes[0]
  ax1.semilogy(time, pos_mag_filter, 'b-', linewidth=2.5, label='Filter (EKF)', alpha=0.7)
  ax1.semilogy(time, pos_mag_smoother, 'r--', linewidth=2.5, label='Smoother (RTS)', alpha=0.7)
  ax1.set_xlabel('Time [min]', fontsize=12)
  ax1.set_ylabel('Position RSS Error [m]', fontsize=12)
  ax1.set_title('Position RSS Error vs Time', fontsize=13, fontweight='bold')
  ax1.grid(True, which='both', linestyle=':', alpha=0.5)
  ax1.legend(loc='best', fontsize=11)

  # Right: Velocity RSS error
  ax2 = axes[1]
  ax2.semilogy(time, vel_mag_filter, 'b-', linewidth=2.5, label='Filter (EKF)', alpha=0.7)
  ax2.semilogy(time, vel_mag_smoother, 'r--', linewidth=2.5, label='Smoother (RTS)', alpha=0.7)
  ax2.set_xlabel('Time [min]', fontsize=12)
  ax2.set_ylabel('Velocity RSS Error [m/s]', fontsize=12)
  ax2.set_title('Velocity RSS Error vs Time', fontsize=13, fontweight='bold')
  ax2.grid(True, which='both', linestyle=':', alpha=0.5)
  ax2.legend(loc='best', fontsize=11)

  # Overall title
  fig.suptitle(title_text, fontsize=14, fontweight='bold')
  fig.tight_layout(rect=[0, 0, 1, 0.96])

  return fig
