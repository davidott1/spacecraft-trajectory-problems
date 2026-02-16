"""
Rauch-Tung-Striebel (RTS) Smoother for Orbit Determination
===========================================================

Implements the RTS smoother for post-processing EKF estimates to produce
smoothed state estimates over the full time span using both past and future
measurements.

The RTS smoother runs backward in time after the forward EKF pass, combining:
- Forward filtered estimate at time k
- Backward smoothed estimate at time k+1
- State transition matrix from k to k+1

This produces optimal estimates using all measurements (past and future).

References:
-----------
  Rauch, H. E., Tung, F., & Striebel, C. T. (1965). Maximum likelihood
  estimates of linear dynamic systems. AIAA Journal, 3(8), 1445-1450.
"""

import numpy as np

from typing import Tuple, Optional, Callable
from datetime import datetime, timedelta

from src.schemas.propagation import PropagationResult, Time
from src.schemas.state import ClassicalOrbitalElements, ModifiedEquinoctialElements
from src.model.orbit_converter import OrbitConverter
from src.model.constants import SOLARSYSTEMCONSTANTS


def rts_smoother(
  filtered_states      : np.ndarray,
  filtered_covariances : np.ndarray,
  estimation_times     : np.ndarray,
  propagator           : Callable[[np.ndarray, float, float], Tuple[np.ndarray, np.ndarray]],
  epoch_dt_utc         : datetime,
  process_noise        : Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
  """
  Apply Rauch-Tung-Striebel (RTS) smoother to forward-filtered EKF estimates.

  The smoother runs backward in time, combining:
  - Filtered estimate at k: x_k|k, P_k|k (from forward pass)
  - Smoothed estimate at k+1: x_{k+1|N}, P_{k+1|N} (from backward pass)
  - Predicted estimate at k+1: x_{k+1|k}, P_{k+1|k} (from forward prediction)
  - State transition matrix: Phi_k (propagates from k to k+1)

  Smoother equations:
    C_k = P_k|k @ Phi_k^T @ inv(P_{k+1|k})
    x_k|N = x_k|k + C_k @ (x_{k+1|N} - x_{k+1|k})
    P_k|N = P_k|k + C_k @ (P_{k+1|N} - P_{k+1|k}) @ C_k^T

  Input:
  ------
    filtered_states : np.ndarray (6, N)
      Forward-filtered state estimates from EKF.
    filtered_covariances : np.ndarray (6, 6, N)
      Forward-filtered covariance matrices from EKF.
    estimation_times : np.ndarray (N,)
      Time array [s] for filtered estimates.
    propagator : Callable
      Function to propagate state and STM: (x, t0, tf) -> (x_f, Phi).
      Same propagator used in forward EKF pass.
    epoch_dt_utc : datetime
      Reference epoch for time conversion.
    process_noise : np.ndarray (6, 6), optional
      Process noise covariance matrix Q used in forward EKF pass.
      If None, assumes zero process noise (only valid for perfect dynamics).

  Output:
  -------
    smoothed_states : np.ndarray (6, N)
      Backward-smoothed state estimates.
    smoothed_covariances : np.ndarray (6, 6, N)
      Backward-smoothed covariance matrices.

  Notes:
  ------
    - Smoother runs backward from N-1 to 0
    - At final time N-1: smoothed = filtered (no future measurements)
    - Requires storing all filtered states and covariances
    - Computationally O(N) in time, O(N) in memory
  """
  n_states, n_times = filtered_states.shape

  # Initialize smoothed arrays (copy filtered to preserve original)
  smoothed_states = filtered_states.copy()
  smoothed_covariances = filtered_covariances.copy()

  # At final time: smoothed = filtered (no future data)
  # smoothed_states[:, -1] = filtered_states[:, -1]  (already copied)
  # smoothed_covariances[:, :, -1] = filtered_covariances[:, :, -1]  (already copied)

  # Backward pass from k = N-2 down to 0
  for k in range(n_times - 2, -1, -1):
    # Times
    t_k = estimation_times[k]
    t_kp1 = estimation_times[k + 1]

    # Skip if times are identical (measurement update duplicates)
    # In this case, use same smoother result for both
    if np.abs(t_kp1 - t_k) < 1e-6:
      smoothed_states[:, k] = smoothed_states[:, k + 1]
      smoothed_covariances[:, :, k] = smoothed_covariances[:, :, k + 1]
      continue

    # Filtered estimates at k
    x_k_filt = filtered_states[:, k]
    P_k_filt = filtered_covariances[:, :, k]

    # Smoothed estimates at k+1 (already computed in previous iteration)
    x_kp1_smooth = smoothed_states[:, k + 1]
    P_kp1_smooth = smoothed_covariances[:, :, k + 1]

    # Predict from k to k+1 to get x_{k+1|k}, P_{k+1|k}, and Phi_k
    x_kp1_pred, Phi_k = propagator(x_k_filt, t_k, t_kp1)

    # Predicted covariance at k+1: P_{k+1|k} = Phi_k @ P_k|k @ Phi_k^T + Q
    # Must use same process noise Q as forward EKF for consistency
    # The EKF computes Q as: Q_pos = (sigma_pos * dt)^2, Q_vel = (sigma_vel * sqrt(dt))^2
    # where sigma_pos = sqrt(process_noise[0,0]), sigma_vel = sqrt(process_noise[3,3])
    # So we must reconstruct the same dt-dependent Q here.
    dt = t_kp1 - t_k
    P_kp1_pred = Phi_k @ P_k_filt @ Phi_k.T
    if process_noise is not None:
      # Reconstruct Q exactly as the EKF does (dt-dependent)
      sigma_pos = np.sqrt(process_noise[0, 0])
      sigma_vel = np.sqrt(process_noise[3, 3])
      Q_k = np.zeros_like(process_noise)
      q_pos = (sigma_pos * abs(dt))**2
      q_vel = (sigma_vel * np.sqrt(abs(dt)))**2
      Q_k[0, 0] = q_pos
      Q_k[1, 1] = q_pos
      Q_k[2, 2] = q_pos
      Q_k[3, 3] = q_vel
      Q_k[4, 4] = q_vel
      Q_k[5, 5] = q_vel
      P_kp1_pred += Q_k

    # Smoother gain: C_k = P_k|k @ Phi_k^T @ inv(P_{k+1|k})
    try:
      C_k = P_k_filt @ Phi_k.T @ np.linalg.inv(P_kp1_pred)
    except np.linalg.LinAlgError:
      # Predicted covariance is singular - use pseudo-inverse
      C_k = P_k_filt @ Phi_k.T @ np.linalg.pinv(P_kp1_pred)

    # Smoothed state: x_k|N = x_k|k + C_k @ (x_{k+1|N} - x_{k+1|k})
    x_k_smooth = x_k_filt + C_k @ (x_kp1_smooth - x_kp1_pred)

    # Smoothed covariance: P_k|N = P_k|k + C_k @ (P_{k+1|N} - P_{k+1|k}) @ C_k^T
    P_k_smooth = P_k_filt + C_k @ (P_kp1_smooth - P_kp1_pred) @ C_k.T

    # Ensure covariance is symmetric (numerical stability)
    P_k_smooth = 0.5 * (P_k_smooth + P_k_smooth.T)

    # Ensure positive-definite by adding small regularization if needed
    # Check if any diagonal elements are non-positive
    min_diag = np.min(np.diag(P_k_smooth))
    if min_diag <= 0:
      # Add small regularization to ensure positive-definiteness
      epsilon = 1e-12
      P_k_smooth += epsilon * np.eye(n_states)

    # Store smoothed estimates
    smoothed_states[:, k] = x_k_smooth
    smoothed_covariances[:, :, k] = P_k_smooth

  return smoothed_states, smoothed_covariances


def smooth_ekf_estimates(
  filter_result        : PropagationResult,
  filtered_covariances : np.ndarray,
  estimation_times     : np.ndarray,
  propagator           : Callable[[np.ndarray, float, float], Tuple[np.ndarray, np.ndarray]],
  epoch_dt_utc         : datetime,
  process_noise        : Optional[np.ndarray] = None,
) -> Tuple[PropagationResult, np.ndarray]:
  """
  Apply RTS smoother to EKF results and create smoothed PropagationResult.

  This is a convenience wrapper around rts_smoother() that:
  - Extracts filtered states from PropagationResult
  - Applies RTS smoother
  - Computes orbital elements for smoothed states
  - Packages results into PropagationResult

  Input:
  ------
    filter_result : PropagationResult
      Forward-filtered EKF result containing states and time grid.
    filtered_covariances : np.ndarray (6, 6, N)
      Forward-filtered covariances from EKF.
    estimation_times : np.ndarray (N,)
      Time array [s] for estimates.
    propagator : Callable
      Propagator function used in EKF: (x, t0, tf) -> (x_f, Phi).
    epoch_dt_utc : datetime
      Reference epoch.
    process_noise : np.ndarray (6, 6), optional
      Process noise covariance matrix Q used in forward EKF pass.

  Output:
  -------
    smoothed_result : PropagationResult
      Smoothed state estimates with orbital elements.
    smoothed_covariances : np.ndarray (6, 6, N)
      Smoothed covariance matrices.
  """
  # Extract filtered states
  filtered_states = filter_result.state

  # Apply RTS smoother
  smoothed_states, smoothed_covariances = rts_smoother(
    filtered_states      = filtered_states,
    filtered_covariances = filtered_covariances,
    estimation_times     = estimation_times,
    propagator           = propagator,
    epoch_dt_utc         = epoch_dt_utc,
    process_noise        = process_noise,
  )

  # Compute orbital elements for smoothed states
  n_times = smoothed_states.shape[1]
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
    # Compute COE
    coe = OrbitConverter.pv_to_coe(
      smoothed_states[0:3, i],
      smoothed_states[3:6, i],
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

    # Compute MEE
    mee = OrbitConverter.pv_to_mee(
      smoothed_states[0:3, i],
      smoothed_states[3:6, i],
      gp = SOLARSYSTEMCONSTANTS.EARTH.GP,
    )
    mee_p[i] = mee.p
    mee_f[i] = mee.f
    mee_g[i] = mee.g
    mee_h[i] = mee.h
    mee_k[i] = mee.k
    mee_L[i] = mee.L

  # Create orbital element objects
  coe_time_series = ClassicalOrbitalElements(
    sma  = coe_sma,
    ecc  = coe_ecc,
    inc  = coe_inc,
    raan = coe_raan,
    aop  = coe_aop,
    ma   = coe_ma,
    ta   = coe_ta,
    ea   = coe_ea,
  )
  mee_time_series = ModifiedEquinoctialElements(
    p = mee_p,
    f = mee_f,
    g = mee_g,
    h = mee_h,
    k = mee_k,
    L = mee_L,
  )

  # Create time grid (same as filter result)
  time_grid = filter_result.time_grid

  # Create smoothed PropagationResult
  smoothed_result = PropagationResult(
    state     = smoothed_states,
    time_grid = time_grid,
    coe       = coe_time_series,
    mee       = mee_time_series,
    success   = True,
    message   = 'RTS smoothed orbit determination',
  )

  # Create at_ephem_times from filter result structure if it exists
  if hasattr(filter_result, 'at_ephem_times') and filter_result.at_ephem_times is not None:
    # Get ephemeris time grid from filter
    ephem_time_grid = filter_result.at_ephem_times.time_grid
    n_ephem = len(ephem_time_grid.grid.relative_initial)

    # Find indices in smoothed states that correspond to ephemeris times
    # We need to match the time grids
    ephem_times_target = ephem_time_grid.grid.relative_initial
    estimation_times_all = time_grid.grid.relative_initial

    # Find matching indices (within tolerance for floating point)
    ephem_indices = []
    for t_target in ephem_times_target:
      idx = np.argmin(np.abs(estimation_times_all - t_target))
      if np.abs(estimation_times_all[idx] - t_target) < 1e-3:  # Within 1ms
        ephem_indices.append(idx)

    ephem_indices = np.array(ephem_indices)

    # Extract smoothed states and orbital elements at ephemeris times
    smoothed_result.at_ephem_times = PropagationResult(
      state     = smoothed_states[:, ephem_indices],
      time_grid = ephem_time_grid,
      coe       = ClassicalOrbitalElements(
        sma  = coe_sma[ephem_indices],
        ecc  = coe_ecc[ephem_indices],
        inc  = coe_inc[ephem_indices],
        raan = coe_raan[ephem_indices],
        aop  = coe_aop[ephem_indices],
        ma   = coe_ma[ephem_indices],
        ta   = coe_ta[ephem_indices],
        ea   = coe_ea[ephem_indices],
      ),
      mee       = ModifiedEquinoctialElements(
        p = mee_p[ephem_indices],
        f = mee_f[ephem_indices],
        g = mee_g[ephem_indices],
        h = mee_h[ephem_indices],
        k = mee_k[ephem_indices],
        L = mee_L[ephem_indices],
      ),
      success   = True,
      message   = 'RTS smoothed states at ephemeris_times',
    )

  return smoothed_result, smoothed_covariances
