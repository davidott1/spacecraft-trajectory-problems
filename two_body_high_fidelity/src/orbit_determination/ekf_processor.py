"""
EKF processing functions for orbit determination.

This module provides high-level functions for processing measurements
with the Extended Kalman Filter to produce state estimates.
"""
import numpy as np

from datetime         import datetime, timedelta
from scipy.integrate  import solve_ivp
from typing           import Optional, Tuple, Callable

from src.orbit_determination.extended_kalman_filter import ExtendedKalmanFilter, EKFMeasurement, EKFConfig
from src.orbit_determination.rts_smoother           import smooth_ekf_estimates
from src.schemas.measurement                        import SimulatedMeasurements, MultiTrackerMeasurements
from src.schemas.propagation                        import PropagationResult, TimeGrid
from src.schemas.state                              import TrackerStation, ClassicalOrbitalElements, ModifiedEquinoctialElements
from src.model.orbit_converter                      import OrbitConverter
from src.model.constants                            import SOLARSYSTEMCONSTANTS
from src.model.dynamics                             import AccelerationSTMDot, GeneralStateEquationsOfMotion
from src.model.time_converter                       import utc_to_et


def create_measurement_noise_covariance(tracker: TrackerStation) -> np.ndarray:
  """
  Create measurement noise covariance matrix R from tracker uncertainty.

  The measurement order is: [range, range_dot, azimuth, azimuth_dot, elevation, elevation_dot]

  Input:
  ------
    tracker : TrackerStation
      Tracker with performance.uncertainty defined.

  Output:
  -------
    R : np.ndarray (6x6)
      Measurement noise covariance matrix with variances on diagonal.
  """
  if tracker.performance is None or tracker.performance.uncertainty is None:
    # Default to zero noise if no uncertainty specified
    return np.zeros((6, 6))

  unc = tracker.performance.uncertainty

  # Measurement noise covariance (diagonal matrix with variances)
  R = np.diag([
    unc.range**2,          # range variance [m^2]
    unc.range_rate**2,     # range_dot variance [(m/s)^2]
    unc.azimuth**2,        # azimuth variance [rad^2]
    unc.azimuth_rate**2,   # azimuth_dot variance [(rad/s)^2]
    unc.elevation**2,      # elevation variance [rad^2]
    unc.elevation_rate**2, # elevation_dot variance [(rad/s)^2]
  ])

  return R


def create_default_process_noise(scale: float = 1.0) -> np.ndarray:
  """
  Create default process noise covariance matrix Q.

  For two-body dynamics, process noise should be small since the model is accurate.
  Larger values mean less trust in the dynamics model.

  Input:
  ------
    scale : float
      Scaling factor for process noise (default 1.0).

  Output:
  -------
    Q : np.ndarray (6x6)
      Process noise covariance matrix.
  """
  # Default process noise
  # We use a relatively high process noise to prevent the filter from becoming
  # "smug" (underestimating covariance) in the presence of unmodeled dynamics
  # (e.g., gravity truncation, unmodeled third bodies, etc.).
  
  # Position: 6e-5 (tuned for sigma ~1)
  # Velocity: 1e-8
  Q_default = np.diag([
    6e-5,  # x (m^2/s)
    6e-5,  # y
    6e-5,  # z
    1e-8,  # vx ((m/s)^2/s)
    1e-8,  # vy
    1e-8,  # vz
  ])

  return Q_default * scale


def create_default_initial_covariance(
  position_sigma: float = 100.0,
  velocity_sigma: float = 1.0,
) -> np.ndarray:
  """
  Create default initial state covariance matrix P0.

  Input:
  ------
    position_sigma : float
      Initial position uncertainty (1-sigma) in meters (default 100 m).
    velocity_sigma : float
      Initial velocity uncertainty (1-sigma) in m/s (default 1 m/s).

  Output:
  -------
    P0 : np.ndarray (6x6)
      Initial state covariance matrix.
  """
  P0 = np.diag([
    position_sigma**2,  # x position variance [m^2]
    position_sigma**2,  # y position variance [m^2]
    position_sigma**2,  # z position variance [m^2]
    velocity_sigma**2,  # vx velocity variance [(m/s)^2]
    velocity_sigma**2,  # vy velocity variance [(m/s)^2]
    velocity_sigma**2,  # vz velocity variance [(m/s)^2]
  ])

  return P0


def create_high_fidelity_propagator(
  dynamics     : GeneralStateEquationsOfMotion,
  epoch_dt_utc : datetime,
) -> Callable[[np.ndarray, float, float], Tuple[np.ndarray, np.ndarray]]:
  """
  Create a high-fidelity propagator function with STM for use with EKF.

  The returned function integrates the state and state transition matrix (STM)
  using the provided dynamics model. Uses the dynamics object's built-in
  state+STM integration method for efficiency.

  Input:
  ------
    dynamics : GeneralStateEquationsOfMotion
      High-fidelity dynamics model (gravity harmonics, drag, SRP, third-body).
    epoch_dt_utc : datetime
      Reference epoch (UTC) for converting delta times to ephemeris times.

  Output:
  -------
    propagator : Callable[[np.ndarray, float, float], Tuple[np.ndarray, np.ndarray]]
      Function with signature (x, t0, tf) -> (x_f, Phi) where:
        x    : Initial state [6]
        t0   : Initial time [s] relative to epoch
        tf   : Final time [s] relative to epoch
        x_f  : Final state [6]
        Phi  : State transition matrix [6x6]
  """

  # Get epoch in ephemeris time
  epoch_et = utc_to_et(epoch_dt_utc)

  def propagator(
      x  : np.ndarray,
      t0 : float,
      tf : float,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate state and STM using high-fidelity dynamics.

    Input:
    ------
      x  : State vector [r, v] at t0
      t0 : Initial time [s] relative to epoch
      tf : Final time [s] relative to epoch

    Returns:
    --------
      x_f : State vector at tf
      Phi : State transition matrix (6x6)
    """
    # Convert delta times to ephemeris times
    et0 = epoch_et + t0
    etf = epoch_et + tf

    # Initial conditions: state + identity STM
    stm_initial = np.eye(6)
    y0          = np.zeros(42)
    y0[0:6]     = x
    y0[6:42]    = stm_initial.flatten()

    # Integrate using dynamics object's state+STM derivative method
    sol = solve_ivp(
      fun    = dynamics.state_stm_time_derivative,
      t_span = [et0, etf],
      y0     = y0,
      method = 'DOP853',
      rtol   = 1e-12,
      atol   = 1e-12,
    )

    # Extract final state and STM
    y_final   = sol.y[:, -1]
    x_f       = y_final[0:6]
    stm_final = y_final[6:42].reshape((6, 6))

    return x_f, stm_final

  return propagator


def process_measurements_with_ekf(
  measurements        : SimulatedMeasurements,
  tracker             : TrackerStation,
  initial_state       : np.ndarray,
  epoch_dt_utc        : datetime,
  ephemeris_times     : np.ndarray,
  propagation_times   : Optional[np.ndarray] = None,
  initial_covariance  : Optional[np.ndarray] = None,
  process_noise       : Optional[np.ndarray] = None,
  process_noise_scale : float = 1.0,
  dynamics            : Optional[GeneralStateEquationsOfMotion] = None,
) -> Tuple[PropagationResult, np.ndarray, np.ndarray, dict]:
  """
  Process simulated measurements with EKF to produce state estimates.

  Time arrays:
  ------------
    ephemeris_times   : Truth data times from JPL Horizons (N points)
    propagation_times : Times to propagate at (defaults to ephemeris_times)
                        Could be finer grid merged with ephemeris_times
    estimation_times  : Output times for EKF results (propagation_times with
                        measurement times repeated for pre/post update)

  Input:
  ------
    measurements : SimulatedMeasurements
      Simulated measurements containing truth and measured data with rates.
    tracker : TrackerStation
      Ground tracking station used for measurements.
    initial_state : np.ndarray (6,)
      Initial state estimate [x, y, z, vx, vy, vz] in ECI frame [m, m/s].
    epoch_dt_utc : datetime
      Reference epoch (UTC) for time conversion.
    ephemeris_times : np.ndarray
      Truth ephemeris times [s]. Used for error comparison.
    propagation_times : np.ndarray, optional
      Times to propagate at [s]. If None, defaults to ephemeris_times.
      Measurements occur at a subset of these times.
    initial_covariance : np.ndarray (6x6), optional
      Initial state covariance. If None, uses default (1 km pos, 10 m/s vel).
    process_noise : np.ndarray (6x6), optional
      Process noise covariance Q. If None, uses default scaled by process_noise_scale.
    process_noise_scale : float
      Scaling factor for default process noise (default 1.0).
    dynamics : GeneralStateEquationsOfMotion, optional
      High-fidelity dynamics model. If None, uses simple two-body dynamics.

  Output:
  -------
    result : PropagationResult
      Propagation result containing estimated states at estimation_times.
      result.at_ephem_times contains states at ephemeris_times for error comparison.
    covariances : np.ndarray (6, 6, n_estimation)
      State covariance matrices at estimation_times.
    estimation_times : np.ndarray (n_estimation,)
      Output time array [s]. Measurement times are repeated (pre/post update).
    residual_data : dict
      Dictionary containing:
        'residuals': List[np.ndarray] - Measurement residuals (innovations)
        'innovation_covariances': List[np.ndarray] - Innovation covariances S = H*P*H^T + R
        'measurement_times': np.ndarray - Times when measurements were processed [s]
  """
  # Default propagation times to ephemeris times
  if propagation_times is None:
    propagation_times = ephemeris_times

  # Set defaults
  if initial_covariance is None:
    initial_covariance = create_default_initial_covariance()

  if process_noise is None:
    process_noise = create_default_process_noise(scale=process_noise_scale)

  # Create measurement noise covariance
  R = create_measurement_noise_covariance(tracker)

  # Create EKF configuration
  # Extract process noise standard deviations from Q matrix
  pos_noise_sigma = np.sqrt(process_noise[0, 0])  # Position process noise [m]
  vel_noise_sigma = np.sqrt(process_noise[3, 3])  # Velocity process noise [m/s]

  # Create propagator: high-fidelity if dynamics provided, otherwise two-body
  propagator = None
  if dynamics is not None:
    propagator = create_high_fidelity_propagator(dynamics, epoch_dt_utc)

  config = EKFConfig(
    process_noise_pos = pos_noise_sigma,
    process_noise_vel = vel_noise_sigma,
    epoch_dt_utc      = epoch_dt_utc,
    propagator        = propagator,  # High-fidelity if dynamics provided, else two-body
    use_joseph_form   = True,
  )

  # Initialize EKF
  ekf = ExtendedKalmanFilter(config)
  ekf.initialize(
    x0     = initial_state,
    P0     = initial_covariance,
    time_s = 0.0,
  )

  # Get measurement times (subset of propagation_times when tracker has visibility)
  measurement_times = measurements.measured.delta_time_epoch
  n_measurements = len(measurement_times)
  n_propagation = len(propagation_times)

  # Build measurement lookup: which propagation indices have measurements
  # Use dict lookup for O(n+m) instead of nested loop O(n*m)
  meas_time_to_idx = {round(mt, 2): j for j, mt in enumerate(measurement_times)}
  meas_data_idx = {}  # Maps propagation index to measurement data index
  for i, t in enumerate(propagation_times):
    rounded_t = round(t, 2)
    if rounded_t in meas_time_to_idx:
      meas_data_idx[i] = meas_time_to_idx[rounded_t]

  # Build estimation_times: propagation_times with measurement times repeated
  # At measurement times: [t, t] for pre-update and post-update
  # At non-measurement times: [t] single point
  n_estimation = n_propagation + n_measurements
  estimated_states = np.zeros((6, n_estimation))
  estimated_covariances = np.zeros((6, 6, n_estimation))
  estimation_times = np.zeros(n_estimation)

  # Process all propagation times
  out_idx = 0
  for i in range(n_propagation):
    t = propagation_times[i]

    if i in meas_data_idx:
      # This is a measurement time - store pre-update and post-update
      j = meas_data_idx[i]

      # Build measurement vector
      z = np.array([
        measurements.measured.range[j],
        measurements.measured.range_dot[j] if measurements.measured.range_dot is not None else 0.0,
        measurements.measured.azimuth[j],
        measurements.measured.azimuth_dot[j] if measurements.measured.azimuth_dot is not None else 0.0,
        measurements.measured.elevation[j],
        measurements.measured.elevation_dot[j] if measurements.measured.elevation_dot is not None else 0.0,
      ])

      # Create EKF measurement object
      meas = EKFMeasurement(
        z       = z,
        R       = R,
        time_s  = t,
        tracker = tracker,
        types   = ['range', 'range_dot', 'azimuth', 'azimuth_dot', 'elevation', 'elevation_dot'],
      )

      # PREDICT to measurement time
      predicted_state = ekf.predict(t)

      # Store PRE-UPDATE (before measurement)
      estimation_times[out_idx] = t
      estimated_states[:, out_idx] = predicted_state.x
      estimated_covariances[:, :, out_idx] = predicted_state.P
      out_idx += 1

      # UPDATE with measurement
      updated_state = ekf.update(meas)

      # Store POST-UPDATE (after measurement) - same time, repeated
      estimation_times[out_idx] = t
      estimated_states[:, out_idx] = updated_state.x
      estimated_covariances[:, :, out_idx] = updated_state.P
      out_idx += 1

    else:
      # No measurement at this time - just predict
      predicted_state = ekf.predict(t)

      # Store single point
      estimation_times[out_idx] = t
      estimated_states[:, out_idx] = predicted_state.x
      estimated_covariances[:, :, out_idx] = predicted_state.P
      out_idx += 1

  # Trim arrays to actual size (in case of rounding)
  estimated_states = estimated_states[:, :out_idx]
  estimated_covariances = estimated_covariances[:, :, :out_idx]
  estimation_times = estimation_times[:out_idx]
  n_estimation = out_idx

  # Compute orbital elements from estimated states
  coe_sma  = np.zeros(n_estimation)
  coe_ecc  = np.zeros(n_estimation)
  coe_inc  = np.zeros(n_estimation)
  coe_raan = np.zeros(n_estimation)
  coe_aop  = np.zeros(n_estimation)
  coe_ta   = np.zeros(n_estimation)
  coe_ea   = np.zeros(n_estimation)
  coe_ma   = np.zeros(n_estimation)
  mee_p    = np.zeros(n_estimation)
  mee_f    = np.zeros(n_estimation)
  mee_g    = np.zeros(n_estimation)
  mee_h    = np.zeros(n_estimation)
  mee_k    = np.zeros(n_estimation)
  mee_L    = np.zeros(n_estimation)

  for i in range(n_estimation):
    # Compute COE
    coe = OrbitConverter.pv_to_coe(
      estimated_states[0:3, i],
      estimated_states[3:6, i],
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
      estimated_states[0:3, i],
      estimated_states[3:6, i],
      gp = SOLARSYSTEMCONSTANTS.EARTH.GP,
    )
    mee_p[i] = mee.p
    mee_f[i] = mee.f
    mee_g[i] = mee.g
    mee_h[i] = mee.h
    mee_k[i] = mee.k
    mee_L[i] = mee.L

  # Construct orbital element objects
  coe_time_series = ClassicalOrbitalElements(sma=coe_sma, ecc=coe_ecc, inc=coe_inc, raan=coe_raan, aop=coe_aop, ma=coe_ma, ta=coe_ta, ea=coe_ea)
  mee_time_series = ModifiedEquinoctialElements(p=mee_p, f=mee_f, g=mee_g, h=mee_h, k=mee_k, L=mee_L)

  # Create time grid for estimation times
  estimation_time_grid = TimeGrid(
    initial = epoch_dt_utc,
    final   = epoch_dt_utc + timedelta(seconds=float(estimation_times[-1])),
    deltas  = estimation_times,
  )

  # Create PropagationResult at estimation_times
  result = PropagationResult(
    state     = estimated_states,
    time_grid = estimation_time_grid,
    coe       = coe_time_series,
    mee       = mee_time_series,
    success   = True,
    message   = 'EKF orbit determination successful',
  )

  # Create at_ephem_times with states at ephemeris_times only (post-update at meas times)
  # for error comparison against truth ephemeris
  ephem_indices = []
  out_idx = 0
  for i in range(n_propagation):
    if i in meas_data_idx:
      # Skip pre-update (out_idx), take post-update (out_idx + 1)
      ephem_indices.append(out_idx + 1)
      out_idx += 2
    else:
      ephem_indices.append(out_idx)
      out_idx += 1

  ephem_indices = np.array(ephem_indices)
  # Create time grid for ephemeris times
  ephem_time_grid = TimeGrid(
    initial = epoch_dt_utc,
    final   = epoch_dt_utc + timedelta(seconds=float(ephemeris_times[-1])),
    deltas  = ephemeris_times,
  )
  result.at_ephem_times = PropagationResult(
    state     = estimated_states[:, ephem_indices],
    time_grid = ephem_time_grid,
    coe       = ClassicalOrbitalElements(
      sma  = coe_sma [ephem_indices],
      ecc  = coe_ecc [ephem_indices],
      inc  = coe_inc [ephem_indices],
      raan = coe_raan[ephem_indices],
      aop  = coe_aop [ephem_indices],
      ma   = coe_ma  [ephem_indices],
      ta   = coe_ta  [ephem_indices],
      ea   = coe_ea  [ephem_indices],
    ),
    mee       = ModifiedEquinoctialElements(
      p = mee_p[ephem_indices],
      f = mee_f[ephem_indices],
      g = mee_g[ephem_indices],
      h = mee_h[ephem_indices],
      k = mee_k[ephem_indices],
      L = mee_L[ephem_indices],
    ),
    success     = True,
    message     = 'EKF states at ephemeris_times (post-update at measurement times)',
  )

  # Extract residuals and innovation covariances from EKF
  residuals, innovation_covariances, residual_times = ekf.get_residuals()

  # Package residual data
  residual_data = {
    'residuals': residuals,
    'innovation_covariances': innovation_covariances,
    'measurement_times': np.array(residual_times),
  }

  return result, estimated_covariances, estimation_times, residual_data


def process_multi_tracker_measurements_with_ekf(
  merged_measurements : MultiTrackerMeasurements,
  initial_state       : np.ndarray,
  epoch_dt_utc        : datetime,
  ephemeris_times     : np.ndarray,
  propagation_times   : Optional[np.ndarray] = None,
  initial_covariance  : Optional[np.ndarray] = None,
  process_noise       : Optional[np.ndarray] = None,
  process_noise_scale : float = 1.0,
  dynamics            : Optional[GeneralStateEquationsOfMotion] = None,
) -> Tuple[PropagationResult, np.ndarray, np.ndarray, dict]:
  """
  Process merged multi-tracker measurements with EKF to produce state estimates.

  This function processes measurements from multiple ground trackers, using the
  correct tracker-specific measurement model for each observation.

  Input:
  ------
    merged_measurements : MultiTrackerMeasurements
      Merged measurements from multiple trackers, sorted by time.
    initial_state : np.ndarray (6,)
      Initial state estimate [x, y, z, vx, vy, vz] in ECI frame [m, m/s].
    epoch_dt_utc : datetime
      Reference epoch (UTC) for time conversion.
    ephemeris_times : np.ndarray
      Truth ephemeris times [s]. Used for error comparison.
    propagation_times : np.ndarray, optional
      Times to propagate at [s]. If None, defaults to ephemeris_times.
    initial_covariance : np.ndarray (6x6), optional
      Initial state covariance. If None, uses default (1 km pos, 10 m/s vel).
    process_noise : np.ndarray (6x6), optional
      Process noise covariance Q. If None, uses default scaled by process_noise_scale.
    process_noise_scale : float
      Scaling factor for default process noise (default 1.0).
    dynamics : GeneralStateEquationsOfMotion, optional
      High-fidelity dynamics model. If None, uses simple two-body dynamics.

  Output:
  -------
    result : PropagationResult
      Propagation result containing estimated states at estimation_times.
      result.at_ephem_times contains states at ephemeris_times for error comparison.
    covariances : np.ndarray (6, 6, n_estimation)
      State covariance matrices at estimation_times.
    estimation_times : np.ndarray (n_estimation,)
      Output time array [s]. Measurement times are repeated (pre/post update).
    residual_data : dict
      Dictionary containing:
        'residuals': List[np.ndarray] - Measurement residuals (innovations)
        'innovation_covariances': List[np.ndarray] - Innovation covariances S = H*P*H^T + R
        'measurement_times': np.ndarray - Times when measurements were processed [s]
        'tracker_names': List[str] - Tracker name for each measurement
  """
  # Default propagation times to ephemeris times
  if propagation_times is None:
    propagation_times = ephemeris_times

  # Set defaults
  if initial_covariance is None:
    initial_covariance = create_default_initial_covariance()

  if process_noise is None:
    process_noise = create_default_process_noise(scale=process_noise_scale)

  # Precompute R matrices for each tracker
  R_matrices = {}
  for meas in merged_measurements.measurements:
    tracker_name = meas.tracker.name
    if tracker_name not in R_matrices:
      R_matrices[tracker_name] = create_measurement_noise_covariance(meas.tracker)

  # Create EKF configuration
  pos_noise_sigma = np.sqrt(process_noise[0, 0])
  vel_noise_sigma = np.sqrt(process_noise[3, 3])

  propagator = None
  if dynamics is not None:
    propagator = create_high_fidelity_propagator(dynamics, epoch_dt_utc)

  config = EKFConfig(
    process_noise_pos = pos_noise_sigma,
    process_noise_vel = vel_noise_sigma,
    epoch_dt_utc      = epoch_dt_utc,
    propagator        = propagator,
    use_joseph_form   = True,
  )

  # Initialize EKF
  ekf = ExtendedKalmanFilter(config)
  ekf.initialize(
    x0     = initial_state,
    P0     = initial_covariance,
    time_s = 0.0,
  )

  # Get measurement times from merged measurements
  measurement_times = merged_measurements.times
  n_measurements = len(measurement_times)
  n_propagation = len(propagation_times)

  # Build measurement lookup: which propagation indices have measurements
  # Multiple measurements can occur at the same time (from different trackers)
  meas_time_to_indices = {}  # Maps rounded time to list of (merged_idx,)
  for merged_idx, t in enumerate(measurement_times):
    rounded_t = round(t, 2)
    if rounded_t not in meas_time_to_indices:
      meas_time_to_indices[rounded_t] = []
    meas_time_to_indices[rounded_t].append(merged_idx)

  # Map propagation index to measurement indices at that time
  prop_idx_to_meas_indices = {}
  for i, t in enumerate(propagation_times):
    rounded_t = round(t, 2)
    if rounded_t in meas_time_to_indices:
      prop_idx_to_meas_indices[i] = meas_time_to_indices[rounded_t]

  # Build estimation_times: propagation_times with measurement times expanded
  # At measurement times: [t, t] for pre-update and post-update
  # At non-measurement times: [t] single point
  n_unique_meas_times = len(prop_idx_to_meas_indices)
  n_estimation = n_propagation + n_unique_meas_times
  estimated_states = np.zeros((6, n_estimation))
  estimated_covariances = np.zeros((6, 6, n_estimation))
  estimation_times = np.zeros(n_estimation)

  # Track which tracker each measurement came from (for residual_data)
  measurement_tracker_names = []

  # Process all propagation times
  out_idx = 0
  for i in range(n_propagation):
    t = propagation_times[i]

    if i in prop_idx_to_meas_indices:
      # This is a measurement time - process all measurements at this time
      meas_indices = prop_idx_to_meas_indices[i]

      # PREDICT to measurement time (once for all measurements at this time)
      predicted_state = ekf.predict(t)

      # Store PRE-UPDATE (before first measurement at this time)
      estimation_times[out_idx] = t
      estimated_states[:, out_idx] = predicted_state.x
      estimated_covariances[:, :, out_idx] = predicted_state.P
      out_idx += 1

      # Process each measurement at this time
      for merged_idx in meas_indices:
        tracker_idx = merged_measurements.tracker_indices[merged_idx]
        meas_idx = merged_measurements.meas_indices[merged_idx]
        tracker = merged_measurements.trackers[merged_idx]
        sim_meas = merged_measurements.measurements[tracker_idx]

        # Build measurement vector from this tracker's data
        z = np.array([
          sim_meas.measured.range[meas_idx],
          sim_meas.measured.range_dot[meas_idx] if sim_meas.measured.range_dot is not None else 0.0,
          sim_meas.measured.azimuth[meas_idx],
          sim_meas.measured.azimuth_dot[meas_idx] if sim_meas.measured.azimuth_dot is not None else 0.0,
          sim_meas.measured.elevation[meas_idx],
          sim_meas.measured.elevation_dot[meas_idx] if sim_meas.measured.elevation_dot is not None else 0.0,
        ])

        # Get R matrix for this tracker
        R = R_matrices[tracker.name]

        # Create EKF measurement object
        meas = EKFMeasurement(
          z       = z,
          R       = R,
          time_s  = t,
          tracker = tracker,
          types   = ['range', 'range_dot', 'azimuth', 'azimuth_dot', 'elevation', 'elevation_dot'],
        )

        # UPDATE with measurement
        updated_state = ekf.update(meas)
        measurement_tracker_names.append(tracker.name)

      # Store POST-UPDATE (after all measurements at this time)
      estimation_times[out_idx] = t
      estimated_states[:, out_idx] = updated_state.x
      estimated_covariances[:, :, out_idx] = updated_state.P
      out_idx += 1

    else:
      # No measurement at this time - just predict
      predicted_state = ekf.predict(t)

      # Store single point
      estimation_times[out_idx] = t
      estimated_states[:, out_idx] = predicted_state.x
      estimated_covariances[:, :, out_idx] = predicted_state.P
      out_idx += 1

  # Trim arrays to actual size
  estimated_states = estimated_states[:, :out_idx]
  estimated_covariances = estimated_covariances[:, :, :out_idx]
  estimation_times = estimation_times[:out_idx]
  n_estimation = out_idx

  # Compute orbital elements from estimated states
  coe_sma  = np.zeros(n_estimation)
  coe_ecc  = np.zeros(n_estimation)
  coe_inc  = np.zeros(n_estimation)
  coe_raan = np.zeros(n_estimation)
  coe_aop  = np.zeros(n_estimation)
  coe_ta   = np.zeros(n_estimation)
  coe_ea   = np.zeros(n_estimation)
  coe_ma   = np.zeros(n_estimation)
  mee_p    = np.zeros(n_estimation)
  mee_f    = np.zeros(n_estimation)
  mee_g    = np.zeros(n_estimation)
  mee_h    = np.zeros(n_estimation)
  mee_k    = np.zeros(n_estimation)
  mee_L    = np.zeros(n_estimation)

  for i in range(n_estimation):
    coe = OrbitConverter.pv_to_coe(
      estimated_states[0:3, i],
      estimated_states[3:6, i],
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
      estimated_states[0:3, i],
      estimated_states[3:6, i],
      gp = SOLARSYSTEMCONSTANTS.EARTH.GP,
    )
    mee_p[i] = mee.p
    mee_f[i] = mee.f
    mee_g[i] = mee.g
    mee_h[i] = mee.h
    mee_k[i] = mee.k
    mee_L[i] = mee.L

  # Construct orbital element objects
  coe_time_series = ClassicalOrbitalElements(sma=coe_sma, ecc=coe_ecc, inc=coe_inc, raan=coe_raan, aop=coe_aop, ma=coe_ma, ta=coe_ta, ea=coe_ea)
  mee_time_series = ModifiedEquinoctialElements(p=mee_p, f=mee_f, g=mee_g, h=mee_h, k=mee_k, L=mee_L)

  # Create time grid for estimation times
  estimation_time_grid = TimeGrid(
    initial = epoch_dt_utc,
    final   = epoch_dt_utc + timedelta(seconds=float(estimation_times[-1])),
    deltas  = estimation_times,
  )

  # Create PropagationResult at estimation_times
  result = PropagationResult(
    state     = estimated_states,
    time_grid = estimation_time_grid,
    coe       = coe_time_series,
    mee       = mee_time_series,
    success   = True,
    message   = 'Multi-tracker EKF orbit determination successful',
  )

  # Create at_ephem_times with states at ephemeris_times only (post-update at meas times)
  ephem_indices = []
  out_idx = 0
  for i in range(n_propagation):
    if i in prop_idx_to_meas_indices:
      # Skip pre-update (out_idx), take post-update (out_idx + 1)
      ephem_indices.append(out_idx + 1)
      out_idx += 2
    else:
      ephem_indices.append(out_idx)
      out_idx += 1

  ephem_indices = np.array(ephem_indices)
  ephem_time_grid = TimeGrid(
    initial = epoch_dt_utc,
    final   = epoch_dt_utc + timedelta(seconds=float(ephemeris_times[-1])),
    deltas  = ephemeris_times,
  )
  result.at_ephem_times = PropagationResult(
    state     = estimated_states[:, ephem_indices],
    time_grid = ephem_time_grid,
    coe       = ClassicalOrbitalElements(
      sma  = coe_sma [ephem_indices],
      ecc  = coe_ecc [ephem_indices],
      inc  = coe_inc [ephem_indices],
      raan = coe_raan[ephem_indices],
      aop  = coe_aop [ephem_indices],
      ma   = coe_ma  [ephem_indices],
      ta   = coe_ta  [ephem_indices],
      ea   = coe_ea  [ephem_indices],
    ),
    mee       = ModifiedEquinoctialElements(
      p = mee_p[ephem_indices],
      f = mee_f[ephem_indices],
      g = mee_g[ephem_indices],
      h = mee_h[ephem_indices],
      k = mee_k[ephem_indices],
      L = mee_L[ephem_indices],
    ),
    success     = True,
    message     = 'Multi-tracker EKF states at ephemeris_times (post-update at measurement times)',
  )

  # Extract residuals and innovation covariances from EKF
  residuals, innovation_covariances, residual_times = ekf.get_residuals()

  # Package residual data with tracker names
  residual_data = {
    'residuals': residuals,
    'innovation_covariances': innovation_covariances,
    'measurement_times': np.array(residual_times),
    'tracker_names': measurement_tracker_names,
  }

  return result, estimated_covariances, estimation_times, residual_data


def apply_rts_smoother(
  filter_result        : PropagationResult,
  filtered_covariances : np.ndarray,
  estimation_times     : np.ndarray,
  epoch_dt_utc         : datetime,
  dynamics             : Optional[GeneralStateEquationsOfMotion] = None,
  process_noise        : Optional[np.ndarray] = None,
) -> Tuple[PropagationResult, np.ndarray]:
  """
  Apply Rauch-Tung-Striebel (RTS) smoother to forward-filtered EKF estimates.

  The smoother produces optimal state estimates using all measurements (past and
  future) by running a backward pass through the filtered estimates.

  Input:
  ------
    filter_result : PropagationResult
      Forward-filtered EKF result.
    filtered_covariances : np.ndarray (6, 6, N)
      Forward-filtered covariances from EKF.
    estimation_times : np.ndarray (N,)
      Time array [s] for estimates.
    epoch_dt_utc : datetime
      Reference epoch.
    dynamics : GeneralStateEquationsOfMotion, optional
      High-fidelity dynamics model (same as used in EKF).
      If None, uses two-body dynamics.
    process_noise : np.ndarray (6, 6), optional
      Process noise covariance matrix Q (same as used in EKF).
      If None, assumes zero process noise.

  Output:
  -------
    smoothed_result : PropagationResult
      Smoothed state estimates.
    smoothed_covariances : np.ndarray (6, 6, N)
      Smoothed covariance matrices.
  """
  # Create propagator (same as EKF)
  propagator = None
  if dynamics is not None:
    propagator = create_high_fidelity_propagator(dynamics, epoch_dt_utc)
  else:
    # Use two-body propagator
    from src.model.constants import SOLARSYSTEMCONSTANTS
    gp = SOLARSYSTEMCONSTANTS.EARTH.GP

    def two_body_propagator(x, t0, tf):
      # Simple two-body propagation with STM
      # (Implementation placeholder - would use actual two-body propagator)
      raise NotImplementedError("Two-body smoother propagator not yet implemented")

    propagator = two_body_propagator

  # Apply smoother
  smoothed_result, smoothed_covariances = smooth_ekf_estimates(
    filter_result        = filter_result,
    filtered_covariances = filtered_covariances,
    estimation_times     = estimation_times,
    propagator           = propagator,
    epoch_dt_utc         = epoch_dt_utc,
    process_noise        = process_noise,
  )

  return smoothed_result, smoothed_covariances


def perturb_initial_state(
  true_state      : np.ndarray,
  position_error  : float = 1000.0,
  velocity_error  : float = 10.0,
  seed            : Optional[int] = None,
) -> np.ndarray:
  """
  Add random perturbation to initial state to simulate imperfect initialization.

  Input:
  ------
    true_state : np.ndarray (6,)
      True initial state [x, y, z, vx, vy, vz].
    position_error : float
      Position error magnitude (1-sigma) in meters (default 1 km).
    velocity_error : float
      Velocity error magnitude (1-sigma) in m/s (default 10 m/s).
    seed : int, optional
      Random seed for reproducibility.

  Output:
  -------
    perturbed_state : np.ndarray (6,)
      Initial state with added noise.
  """
  if seed is not None:
    np.random.seed(seed)

  # Generate random perturbations (Gaussian)
  position_perturbation = np.random.randn(3) * position_error
  velocity_perturbation = np.random.randn(3) * velocity_error

  # Apply perturbations
  perturbed_state = true_state.copy()
  perturbed_state[0:3] += position_perturbation
  perturbed_state[3:6] += velocity_perturbation

  return perturbed_state
