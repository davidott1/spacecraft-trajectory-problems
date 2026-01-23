"""
Measurement Residual Ratio Plotting Functions
==============================================

This module provides functions for plotting measurement residual ratios
from Extended Kalman Filter orbit determination.

Residual Ratio = residual / sqrt(H*P*H^T + R)

where:
  - residual: measurement innovation (y = z_measured - z_predicted)
  - H*P*H^T + R: innovation covariance (S)

The residual ratio should be roughly normally distributed with mean 0 and
standard deviation 1 if the filter is performing well.
"""
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Optional
from matplotlib.figure import Figure


def plot_measurement_residual_ratio(
  residuals              : List[np.ndarray],
  innovation_covariances : List[np.ndarray],
  measurement_times      : np.ndarray,
  title_text             : str = "Measurement Residual Ratio",
  measurement_types      : Optional[List[str]] = None,
) -> Figure:
  """
  Plot measurement residual ratio timeseries.

  The residual ratio is computed as:
    ratio[i] = residual[i] / sqrt(S[i,i])

  where S is the innovation covariance matrix.

  For a well-performing filter, the residual ratios should be normally
  distributed with mean ~0 and standard deviation ~1.

  Input:
  ------
    residuals : List[np.ndarray]
      List of measurement residuals (innovations) from EKF.
      Each element is a 1D array of size n_meas_types.
    innovation_covariances : List[np.ndarray]
      List of innovation covariance matrices from EKF.
      Each element is a 2D array of size (n_meas_types, n_meas_types).
    measurement_times : np.ndarray
      Times when measurements were processed [s].
    title_text : str
      Plot title.
    measurement_types : List[str], optional
      Names of measurement types (e.g., ['range', 'range_dot', 'azimuth', ...]).
      If None, defaults to standard 6-element measurement vector.

  Output:
  -------
    fig : matplotlib.figure.Figure
      Figure containing residual ratio plots.
  """
  # Default measurement types for 6-element measurement vector
  if measurement_types is None:
    measurement_types = [
      'Range',
      'Range Rate',
      'Azimuth',
      'Azimuth Rate',
      'Elevation',
      'Elevation Rate',
    ]

  n_measurements = len(residuals)
  if n_measurements == 0:
    # No measurements to plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.text(0.5, 0.5, 'No measurements available',
            ha='center', va='center', fontsize=14)
    ax.set_title(title_text)
    return fig

  # Get number of measurement types from first residual
  n_types = len(residuals[0])

  # Ensure we have the right number of labels
  if len(measurement_types) < n_types:
    measurement_types = [f'Meas {i+1}' for i in range(n_types)]

  # Compute residual ratios for each measurement type
  residual_ratios = np.zeros((n_types, n_measurements))

  for i in range(n_measurements):
    residual = residuals[i]
    S = innovation_covariances[i]

    # Compute residual ratio for each measurement type
    # ratio = residual / sqrt(diagonal of S)
    for j in range(n_types):
      sigma_j = np.sqrt(S[j, j])
      if sigma_j > 1e-12:  # Avoid division by zero
        residual_ratios[j, i] = residual[j] / sigma_j
      else:
        residual_ratios[j, i] = 0.0

  # Convert measurement times to minutes for plotting
  times_minutes = measurement_times / 60.0

  # Create figure with subplots for each measurement type
  fig, axes = plt.subplots(n_types, 1, figsize=(12, 2.5*n_types), sharex=True)

  # Handle single subplot case
  if n_types == 1:
    axes = [axes]

  # Plot residual ratio for each measurement type
  for i, (ax, meas_type) in enumerate(zip(axes, measurement_types)):
    # Plot residual ratio
    ax.plot(times_minutes, residual_ratios[i, :], 'o-',
            linewidth=1, markersize=3, label='Residual Ratio')

    # Add ±1σ, ±2σ, ±3σ bounds
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.axhline(y=1, color='g', linestyle='--', linewidth=0.8, alpha=0.5, label='±1σ')
    ax.axhline(y=-1, color='g', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=2, color='orange', linestyle='--', linewidth=0.8, alpha=0.5, label='±2σ')
    ax.axhline(y=-2, color='orange', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=3, color='r', linestyle='--', linewidth=0.8, alpha=0.5, label='±3σ')
    ax.axhline(y=-3, color='r', linestyle='--', linewidth=0.8, alpha=0.5)

    # Formatting
    ax.set_ylabel(f'{meas_type}\n(σ)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, ncol=4)

    # Set y-limits to show ±3.5σ range
    ax.set_ylim([-3.5, 3.5])

  # Set common x-label
  axes[-1].set_xlabel('Time Since Epoch (minutes)', fontsize=11)

  # Set title
  fig.suptitle(title_text, fontsize=14, y=0.995)

  # Adjust layout
  fig.tight_layout(rect=[0, 0, 1, 0.99])

  return fig


def compute_residual_statistics(
  residuals              : List[np.ndarray],
  innovation_covariances : List[np.ndarray],
) -> dict:
  """
  Compute statistics for measurement residual ratios.

  Input:
  ------
    residuals : List[np.ndarray]
      List of measurement residuals from EKF.
    innovation_covariances : List[np.ndarray]
      List of innovation covariance matrices from EKF.

  Output:
  -------
    stats : dict
      Dictionary containing:
        'mean': np.ndarray - Mean residual ratio for each measurement type
        'std': np.ndarray - Standard deviation of residual ratio
        'rms': np.ndarray - RMS residual ratio
        'n_measurements': int - Number of measurements
        'n_types': int - Number of measurement types
  """
  if len(residuals) == 0:
    return {
      'mean': np.array([]),
      'std': np.array([]),
      'rms': np.array([]),
      'n_measurements': 0,
      'n_types': 0,
    }

  n_measurements = len(residuals)
  n_types = len(residuals[0])

  # Compute residual ratios
  residual_ratios = np.zeros((n_types, n_measurements))

  for i in range(n_measurements):
    residual = residuals[i]
    S = innovation_covariances[i]

    for j in range(n_types):
      sigma_j = np.sqrt(S[j, j])
      if sigma_j > 1e-12:
        residual_ratios[j, i] = residual[j] / sigma_j
      else:
        residual_ratios[j, i] = 0.0

  # Compute statistics
  mean = np.mean(residual_ratios, axis=1)
  std = np.std(residual_ratios, axis=1)
  rms = np.sqrt(np.mean(residual_ratios**2, axis=1))

  return {
    'mean': mean,
    'std': std,
    'rms': rms,
    'n_measurements': n_measurements,
    'n_types': n_types,
  }
