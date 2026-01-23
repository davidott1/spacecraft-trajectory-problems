"""
Covariance plotting functions for orbit determination.

This module provides functions to visualize the state covariance evolution
from the Extended Kalman Filter.
"""
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from typing import Optional


def plot_covariance_timeseries(
  covariances       : np.ndarray,
  delta_time_epoch  : np.ndarray,
  title_text        : str = "State Covariance Evolution",
  measurement_times : Optional[np.ndarray] = None,
) -> Figure:
  """
  Plot state covariance evolution over time.

  Creates a 2x1 grid showing:
    - Position uncertainty (3-sigma bounds) vs time
    - Velocity uncertainty (3-sigma bounds) vs time

  Input:
  ------
    covariances : np.ndarray (6, 6, N)
      State covariance matrices at each time step.
      Shape: (6, 6, N) where N is number of time steps.
    delta_time_epoch : np.ndarray (N,)
      Time array relative to epoch.
    title_text : str
      Title for the figure.
    measurement_times : np.ndarray, optional
      Times when measurements were incorporated relative to epoch. If provided,
      vertical lines are drawn at these times to indicate state updates.

  Output:
  -------
    fig : Figure
      Matplotlib figure containing the covariance plots.
  """
  n_steps = covariances.shape[2]

  # Extract standard deviations (1-sigma) for each state component
  sigma_x  = np.sqrt(covariances[0, 0, :])  # Position x uncertainty [m]
  sigma_y  = np.sqrt(covariances[1, 1, :])  # Position y uncertainty [m]
  sigma_z  = np.sqrt(covariances[2, 2, :])  # Position z uncertainty [m]
  sigma_vx = np.sqrt(covariances[3, 3, :])  # Velocity x uncertainty [m/s]
  sigma_vy = np.sqrt(covariances[4, 4, :])  # Velocity y uncertainty [m/s]
  sigma_vz = np.sqrt(covariances[5, 5, :])  # Velocity z uncertainty [m/s]

  # Compute RSS (Root Sum Square) uncertainties
  sigma_pos = np.sqrt(sigma_x**2 + sigma_y**2 + sigma_z**2)  # Total position uncertainty [m]
  sigma_vel = np.sqrt(sigma_vx**2 + sigma_vy**2 + sigma_vz**2)  # Total velocity uncertainty [m/s]

  # Convert time to minutes
  time_min = delta_time_epoch / 60.0

  # Create figure with 2 subplots (position and velocity)
  fig, axes = plt.subplots(2, 1, figsize=(12, 8))

  # Plot 1: Position uncertainty (1-sigma and 3-sigma)
  ax1 = axes[0]
  ax1.semilogy(time_min, sigma_pos, 'b-', linewidth=2, label='1-sigma (RSS)')
  ax1.semilogy(time_min, 3 * sigma_pos, 'r--', linewidth=1.5, label='3-sigma (RSS)')
  ax1.set_xlabel('Time [min]', fontsize=12)
  ax1.set_ylabel('Position Uncertainty [m]', fontsize=12)
  ax1.set_title('Position Uncertainty vs Time', fontsize=13)
  ax1.grid(True, alpha=0.3, which='both')

  # Plot 2: Velocity uncertainty (1-sigma and 3-sigma)
  ax2 = axes[1]
  ax2.semilogy(time_min, sigma_vel, 'b-', linewidth=2, label='1-sigma (RSS)')
  ax2.semilogy(time_min, 3 * sigma_vel, 'r--', linewidth=1.5, label='3-sigma (RSS)')
  ax2.set_xlabel('Time [min]', fontsize=12)
  ax2.set_ylabel('Velocity Uncertainty [m/s]', fontsize=12)
  ax2.set_title('Velocity Uncertainty vs Time', fontsize=13)
  ax2.grid(True, alpha=0.3, which='both')

  # Add vertical lines at measurement update times
  if measurement_times is not None and len(measurement_times) > 0:
    meas_times_min = measurement_times / 60.0
    for i, t in enumerate(meas_times_min):
      label = 'Measurement Update' if i == 0 else None
      ax1.axvline(x=t, color='green', linestyle='-', linewidth=0.5, alpha=0.5, label=label)
      ax2.axvline(x=t, color='green', linestyle='-', linewidth=0.5, alpha=0.5, label=label)

  ax1.legend(loc='best', fontsize=10)
  ax2.legend(loc='best', fontsize=10)

  # Overall figure title
  fig.suptitle(title_text, fontsize=14, fontweight='bold')
  fig.tight_layout(rect=[0, 0, 1, 0.97])

  return fig


def plot_covariance_components(
  covariances       : np.ndarray,
  delta_time_epoch  : np.ndarray,
  title_text        : str = "State Uncertainty Components",
  measurement_times : Optional[np.ndarray] = None,
) -> Figure:
  """
  Plot individual state uncertainty components over time.

  Creates a 2x1 grid showing:
    - Position component uncertainties (x, y, z) vs time
    - Velocity component uncertainties (vx, vy, vz) vs time

  Input:
  ------
    covariances : np.ndarray (6, 6, N)
      State covariance matrices at each time step.
    delta_time_epoch : np.ndarray (N,)
      Time array relative to epoch.
    title_text : str
      Title for the figure.
    measurement_times : np.ndarray, optional
      Times when measurements were incorporated relative to epoch. If provided,
      vertical lines are drawn at these times to indicate state updates.

  Output:
  -------
    fig : Figure
      Matplotlib figure containing the component uncertainty plots.
  """
  # Extract standard deviations (1-sigma) for each state component
  sigma_x  = np.sqrt(covariances[0, 0, :])  # Position x uncertainty [m]
  sigma_y  = np.sqrt(covariances[1, 1, :])  # Position y uncertainty [m]
  sigma_z  = np.sqrt(covariances[2, 2, :])  # Position z uncertainty [m]
  sigma_vx = np.sqrt(covariances[3, 3, :])  # Velocity x uncertainty [m/s]
  sigma_vy = np.sqrt(covariances[4, 4, :])  # Velocity y uncertainty [m/s]
  sigma_vz = np.sqrt(covariances[5, 5, :])  # Velocity z uncertainty [m/s]

  # Convert time to minutes
  time_min = delta_time_epoch / 60.0

  # Create figure with 2 subplots
  fig, axes = plt.subplots(2, 1, figsize=(12, 8))

  # Plot 1: Position component uncertainties (1-sigma)
  ax1 = axes[0]
  ax1.semilogy(time_min, sigma_x, 'r-', linewidth=1.5, label='σx')
  ax1.semilogy(time_min, sigma_y, 'g-', linewidth=1.5, label='σy')
  ax1.semilogy(time_min, sigma_z, 'b-', linewidth=1.5, label='σz')
  ax1.set_xlabel('Time [min]', fontsize=12)
  ax1.set_ylabel('Position Uncertainty [m]', fontsize=12)
  ax1.set_title('Position Component Uncertainties (1-sigma)', fontsize=13)
  ax1.grid(True, alpha=0.3, which='both')

  # Plot 2: Velocity component uncertainties (1-sigma)
  ax2 = axes[1]
  ax2.semilogy(time_min, sigma_vx, 'r-', linewidth=1.5, label='σvx')
  ax2.semilogy(time_min, sigma_vy, 'g-', linewidth=1.5, label='σvy')
  ax2.semilogy(time_min, sigma_vz, 'b-', linewidth=1.5, label='σvz')
  ax2.set_xlabel('Time [min]', fontsize=12)
  ax2.set_ylabel('Velocity Uncertainty [m/s]', fontsize=12)
  ax2.set_title('Velocity Component Uncertainties (1-sigma)', fontsize=13)
  ax2.grid(True, alpha=0.3, which='both')

  # Add vertical lines at measurement update times
  if measurement_times is not None and len(measurement_times) > 0:
    meas_times_min = measurement_times / 60.0
    for i, t in enumerate(meas_times_min):
      label = 'Measurement Update' if i == 0 else None
      ax1.axvline(x=t, color='gray', linestyle='-', linewidth=0.5, alpha=0.5, label=label)
      ax2.axvline(x=t, color='gray', linestyle='-', linewidth=0.5, alpha=0.5, label=label)

  ax1.legend(loc='best', fontsize=10)
  ax2.legend(loc='best', fontsize=10)

  # Overall figure title
  fig.suptitle(title_text, fontsize=14, fontweight='bold')
  fig.tight_layout(rect=[0, 0, 1, 0.97])

  return fig


def plot_covariance_combined(
  covariances       : np.ndarray,
  delta_time_epoch  : np.ndarray,
  title_text        : str = "State Uncertainty Evolution",
  measurement_times : Optional[np.ndarray] = None,
) -> Figure:
  """
  Plot combined covariance visualization with RSS and component uncertainties.

  Creates a 1x2 grid showing:
    - Left: Position uncertainties (RSS and components) vs time
    - Right: Velocity uncertainties (RSS and components) vs time

  Input:
  ------
    covariances : np.ndarray (6, 6, N)
      State covariance matrices at each time step.
    delta_time_epoch : np.ndarray (N,)
      Time array relative to epoch.
    title_text : str
      Title for the figure.
    measurement_times : np.ndarray, optional
      Times when measurements were incorporated relative to epoch. If provided,
      vertical lines are drawn at these times to indicate state updates.

  Output:
  -------
    fig : Figure
      Matplotlib figure containing the combined covariance plots.
  """
  # Extract standard deviations (1-sigma) for each state component
  sigma_x  = np.sqrt(covariances[0, 0, :])  # Position x uncertainty [m]
  sigma_y  = np.sqrt(covariances[1, 1, :])  # Position y uncertainty [m]
  sigma_z  = np.sqrt(covariances[2, 2, :])  # Position z uncertainty [m]
  sigma_vx = np.sqrt(covariances[3, 3, :])  # Velocity x uncertainty [m/s]
  sigma_vy = np.sqrt(covariances[4, 4, :])  # Velocity y uncertainty [m/s]
  sigma_vz = np.sqrt(covariances[5, 5, :])  # Velocity z uncertainty [m/s]

  # Compute RSS (Root Sum Square) uncertainties
  sigma_pos_rss = np.sqrt(sigma_x**2 + sigma_y**2 + sigma_z**2)  # Position RSS [m]
  sigma_vel_rss = np.sqrt(sigma_vx**2 + sigma_vy**2 + sigma_vz**2)  # Velocity RSS [m/s]

  # Convert time to minutes
  time_min = delta_time_epoch / 60.0

  # Create figure with 1x2 subplots
  fig, axes = plt.subplots(1, 2, figsize=(16, 6))

  # Left: Position uncertainties (RSS + components)
  ax1 = axes[0]
  # Plot RSS
  ax1.semilogy(time_min, sigma_pos_rss, 'k-', linewidth=3.0, alpha=0.5, label='√(σrx² + σry² + σrz²)')
  # Plot components
  ax1.semilogy(time_min, sigma_x, 'r-', linewidth=1.2, label='σrx', alpha=0.6)
  ax1.semilogy(time_min, sigma_y, 'g-', linewidth=1.2, label='σry', alpha=0.6)
  ax1.semilogy(time_min, sigma_z, 'b-', linewidth=1.2, label='σrz', alpha=0.6)
  ax1.set_xlabel('Time [min]', fontsize=12)
  ax1.set_ylabel('Position Uncertainty [m]', fontsize=12)
  ax1.set_title('Position Uncertainty', fontsize=13, fontweight='bold')
  ax1.grid(True, which='both', linestyle=':', alpha=0.5)
  ax1.legend(loc='best', fontsize=10)

  # Add measurement markers if provided
  if measurement_times is not None:
    meas_time_min = measurement_times / 60.0
    for mt in meas_time_min:
      ax1.axvline(mt, color='magenta', alpha=0.1, linewidth=0.8, linestyle='-')

  # Right: Velocity uncertainties (RSS + components)
  ax2 = axes[1]
  # Plot RSS
  ax2.semilogy(time_min, sigma_vel_rss, 'k-', linewidth=3.0, alpha=0.5, label='√(σvx² + σvy² + σvz²)')
  # Plot components
  ax2.semilogy(time_min, sigma_vx, 'r-', linewidth=1.2, label='σvx', alpha=0.6)
  ax2.semilogy(time_min, sigma_vy, 'g-', linewidth=1.2, label='σvy', alpha=0.6)
  ax2.semilogy(time_min, sigma_vz, 'b-', linewidth=1.2, label='σvz', alpha=0.6)
  ax2.set_xlabel('Time [min]', fontsize=12)
  ax2.set_ylabel('Velocity Uncertainty [m/s]', fontsize=12)
  ax2.set_title('Velocity Uncertainty', fontsize=13, fontweight='bold')
  ax2.grid(True, which='both', linestyle=':', alpha=0.5)
  ax2.legend(loc='best', fontsize=10)

  # Add measurement markers if provided
  if measurement_times is not None:
    for mt in meas_time_min:
      ax2.axvline(mt, color='magenta', alpha=0.1, linewidth=0.8, linestyle='-')

  # Overall figure title
  fig.suptitle(title_text, fontsize=14, fontweight='bold')
  fig.tight_layout(rect=[0, 0, 1, 0.96])

  return fig


def plot_covariance_filter_vs_smoother(
  filter_covariances  : np.ndarray,
  smoother_covariances: np.ndarray,
  delta_time_epoch    : np.ndarray,
  title_text          : str = "Filter vs Smoother Uncertainty",
  measurement_times   : Optional[np.ndarray] = None,
) -> Figure:
  """
  Plot filter and smoother covariance comparison.

  Creates a 1x2 grid showing:
    - Left: Position uncertainties (filter RSS vs smoother RSS)
    - Right: Velocity uncertainties (filter RSS vs smoother RSS)

  Input:
  ------
    filter_covariances : np.ndarray (6, 6, N)
      Forward-filtered state covariance matrices.
    smoother_covariances : np.ndarray (6, 6, N)
      Backward-smoothed state covariance matrices.
    delta_time_epoch : np.ndarray (N,)
      Time array relative to epoch [s].
    title_text : str
      Title for the figure.
    measurement_times : np.ndarray, optional
      Times when measurements were incorporated [s].

  Output:
  -------
    fig : Figure
      Matplotlib figure containing the comparison plots.
  """
  # Extract filter standard deviations
  sigma_x_filt  = np.sqrt(filter_covariances[0, 0, :])
  sigma_y_filt  = np.sqrt(filter_covariances[1, 1, :])
  sigma_z_filt  = np.sqrt(filter_covariances[2, 2, :])
  sigma_vx_filt = np.sqrt(filter_covariances[3, 3, :])
  sigma_vy_filt = np.sqrt(filter_covariances[4, 4, :])
  sigma_vz_filt = np.sqrt(filter_covariances[5, 5, :])

  # Extract smoother standard deviations
  sigma_x_smooth  = np.sqrt(smoother_covariances[0, 0, :])
  sigma_y_smooth  = np.sqrt(smoother_covariances[1, 1, :])
  sigma_z_smooth  = np.sqrt(smoother_covariances[2, 2, :])
  sigma_vx_smooth = np.sqrt(smoother_covariances[3, 3, :])
  sigma_vy_smooth = np.sqrt(smoother_covariances[4, 4, :])
  sigma_vz_smooth = np.sqrt(smoother_covariances[5, 5, :])

  # Compute RSS uncertainties
  sigma_pos_filt   = np.sqrt(sigma_x_filt**2 + sigma_y_filt**2 + sigma_z_filt**2)
  sigma_vel_filt   = np.sqrt(sigma_vx_filt**2 + sigma_vy_filt**2 + sigma_vz_filt**2)
  sigma_pos_smooth = np.sqrt(sigma_x_smooth**2 + sigma_y_smooth**2 + sigma_z_smooth**2)
  sigma_vel_smooth = np.sqrt(sigma_vx_smooth**2 + sigma_vy_smooth**2 + sigma_vz_smooth**2)

  # Convert time to minutes
  time_min = delta_time_epoch / 60.0

  # Create figure with 1x2 subplots
  fig, axes = plt.subplots(1, 2, figsize=(16, 6))

  # Left: Position uncertainties
  ax1 = axes[0]
  ax1.semilogy(time_min, sigma_pos_filt, 'b-', linewidth=2.5, label='Filter (EKF)', alpha=0.7)
  ax1.semilogy(time_min, sigma_pos_smooth, 'r-', linewidth=2.5, label='Smoother (RTS)', alpha=0.7)
  ax1.set_xlabel('Time [min]', fontsize=12)
  ax1.set_ylabel('Position Uncertainty [m]', fontsize=12)
  ax1.set_title('Position Uncertainty (1-σ RSS)', fontsize=13, fontweight='bold')
  ax1.grid(True, which='both', linestyle=':', alpha=0.5)
  ax1.legend(loc='best', fontsize=11)

  # Add measurement markers
  if measurement_times is not None:
    meas_time_min = measurement_times / 60.0
    for mt in meas_time_min:
      ax1.axvline(mt, color='green', alpha=0.15, linewidth=0.8, linestyle='-')

  # Right: Velocity uncertainties
  ax2 = axes[1]
  ax2.semilogy(time_min, sigma_vel_filt, 'b-', linewidth=2.5, label='Filter (EKF)', alpha=0.7)
  ax2.semilogy(time_min, sigma_vel_smooth, 'r-', linewidth=2.5, label='Smoother (RTS)', alpha=0.7)
  ax2.set_xlabel('Time [min]', fontsize=12)
  ax2.set_ylabel('Velocity Uncertainty [m/s]', fontsize=12)
  ax2.set_title('Velocity Uncertainty (1-σ RSS)', fontsize=13, fontweight='bold')
  ax2.grid(True, which='both', linestyle=':', alpha=0.5)
  ax2.legend(loc='best', fontsize=11)

  # Add measurement markers
  if measurement_times is not None:
    for mt in meas_time_min:
      ax2.axvline(mt, color='green', alpha=0.15, linewidth=0.8, linestyle='-')

  # Overall figure title
  fig.suptitle(title_text, fontsize=14, fontweight='bold')
  fig.tight_layout(rect=[0, 0, 1, 0.96])

  return fig
