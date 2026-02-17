"""
Extended Kalman Filter for Orbit Determination
==============================================

Implements an Extended Kalman Filter (EKF) for spacecraft state estimation
using ground-based tracking measurements (range, azimuth, elevation, and rates).

The EKF linearizes the nonlinear dynamics and measurement models about the
current state estimate, enabling optimal state estimation for orbit determination.

State Vector:
  x = [r_x, r_y, r_z, v_x, v_y, v_z]^T  (6-element state)
  
  Optionally augmented with:
  x = [r, v, C_d, C_r]^T  (drag coefficient, SRP coefficient)

Measurements:
  z = [range, azimuth, elevation, range_dot, az_dot, el_dot]^T
  
  Subset can be used depending on tracker capabilities.
"""

import numpy as np

from dataclasses     import dataclass, field
from datetime        import datetime
from typing          import Optional, Callable, List, Tuple
from scipy.integrate import solve_ivp

from src.model.orbit_converter import TopocentricConverter
from src.model.frame_and_vector_converter import VectorConverter
from src.model.time_converter  import utc_to_et
from src.schemas.state         import TrackerStation


@dataclass
class EKFState:
  """
  Extended Kalman Filter state container.

  Attributes:
    x      : State vector [n_states]
    P      : State covariance matrix [n_states x n_states]
    time_s : Time since epoch [s]
  """
  x      : np.ndarray
  P      : np.ndarray
  time_s : float

  def __post_init__(self):
    self.x = np.asarray(self.x).flatten()
    self.P = np.asarray(self.P)

  @property
  def position(self) -> np.ndarray:
    """Position vector [m]."""
    return self.x[0:3]

  @property
  def velocity(self) -> np.ndarray:
    """Velocity vector [m/s]."""
    return self.x[3:6]

  @property
  def n_states(self) -> int:
    """Number of states."""
    return len(self.x)

  def copy(self) -> 'EKFState':
    """Return a deep copy of the state."""
    return EKFState(
      x      = self.x.copy(),
      P      = self.P.copy(),
      time_s = self.time_s,
    )


@dataclass
class EKFMeasurement:
  """
  Measurement container for EKF update.

  Attributes:
    z       : measurement vector
    R       : measurement noise covariance matrix
    time_s  : measurement time since epoch [s]
    tracker : tracker station that made the measurement
    types   : list of measurement types in z (e.g., ['range', 'azimuth', 'elevation'])
  """
  z       : np.ndarray
  R       : np.ndarray
  time_s  : float
  tracker : TrackerStation
  types   : List[str] = field(default_factory=lambda: ['range', 'azimuth', 'elevation'])

  def __post_init__(self):
    self.z = np.asarray(self.z).flatten()
    self.R = np.asarray(self.R)


@dataclass
class EKFConfig:
  """
  Configuration for Extended Kalman Filter.

  Attributes:
    process_noise_pos    : Position process noise spectral density [m/s]
                          Q_pos = (process_noise_pos * dt)^2
    process_noise_vel    : Velocity process noise spectral density [m/s^2]
                          Q_vel = (process_noise_vel * sqrt(dt))^2
    epoch_dt_utc         : Reference epoch for time conversion
    propagator           : Function to propagate state: (x, t0, tf) -> (x_f, Phi)
    use_joseph_form      : Use Joseph form for covariance update (more numerically stable)
  """
  process_noise_pos : float = 1.0        # m/s (spectral density)
  process_noise_vel : float = 1e-3       # m/s^2 (spectral density)
  epoch_dt_utc      : Optional[datetime] = None
  propagator        : Optional[Callable] = None
  use_joseph_form   : bool = True


class ExtendedKalmanFilter:
  """
  Extended Kalman Filter for spacecraft orbit determination.

  The EKF estimates the spacecraft state (position and velocity) using
  nonlinear dynamics and measurement models that are linearized about
  the current state estimate.

  Example:
  --------
    # Initialize filter
    config = EKFConfig(
      process_noise_pos = 10.0,
      process_noise_vel = 0.01,
      epoch_dt_utc      = datetime(2024, 1, 1),
      propagator        = my_propagator,
    )
    ekf = ExtendedKalmanFilter(config)

    # Set initial state
    x0 = np.array([r_x, r_y, r_z, v_x, v_y, v_z])
    P0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
    ekf.initialize(x0, P0, t0=0.0)

    # Process measurements
    for meas in measurements:
      ekf.predict(meas.time_s)
      ekf.update(meas)

    # Get final state estimate
    state = ekf.get_state()
  """

  def __init__(self, config: EKFConfig):
    """
    Initialize the Extended Kalman Filter.

    Input:
    ------
      config : EKFConfig
        Filter configuration parameters.
    """
    self.config = config
    self._state   : Optional[EKFState] = None
    self._history : List[EKFState]     = []

    # Storage for measurement residuals and innovation covariances
    self._residuals               : List[np.ndarray] = []  # Innovation (y = z - h(x))
    self._innovation_covariances  : List[np.ndarray] = []  # S = H*P*H^T + R
    self._measurement_times       : List[float]      = []  # Times when measurements were processed

    # Convert epoch to ET if provided
    if config.epoch_dt_utc is not None:
      self._epoch_et = utc_to_et(config.epoch_dt_utc)
    else:
      self._epoch_et = 0.0

  def initialize(
    self,
    x0     : np.ndarray,
    P0     : np.ndarray,
    time_s : float = 0.0,
  ) -> None:
    """
    Initialize the filter with an initial state estimate.

    Input:
    ------
      x0     : initial state vector [n_states]
      P0     : initial state covariance [n_states x n_states]
      time_s : initial time since epoch [s]
    """
    self._state   = EKFState(x=x0, P=P0, time_s=time_s)
    self._history = [self._state.copy()]

  def get_state(self) -> EKFState:
    """
    Get the current state estimate.

    Returns:
    --------
      state : EKFState
        Current state estimate with covariance.
    """
    if self._state is None:
      raise RuntimeError("Filter not initialized. Call initialize() first.")
    return self._state.copy()

  def get_history(self) -> List[EKFState]:
    """
    Get the history of state estimates.

    Returns:
    --------
      history : List[EKFState]
        List of state estimates at each update.
    """
    return [s.copy() for s in self._history]

  def get_residuals(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Get measurement residuals and innovation covariances from filter updates.

    Returns:
    --------
      residuals : List[np.ndarray]
        List of measurement residuals (innovations): y = z - h(x_pred)
      innovation_covariances : List[np.ndarray]
        List of innovation covariances: S = H*P*H^T + R
      measurement_times : List[float]
        Times [s] when measurements were processed
    """
    return (
      [r.copy() for r in self._residuals],
      [S.copy() for S in self._innovation_covariances],
      self._measurement_times.copy()
    )

  def predict(self, time_s: float) -> EKFState:
    """
    Predict (propagate) the state to a new time.

    Uses the configured propagator to propagate the state and state
    transition matrix (STM) from the current time to the target time.

    Input:
    ------
      time_s : target time since epoch [s]

    Returns:
    --------
      state : EKFState
        Predicted state at target time.
    """
    if self._state is None:
      raise RuntimeError("Filter not initialized. Call initialize() first.")

    # Predict delta-time. Skip if already at target time
    dt = time_s - self._state.time_s
    if abs(dt) < 1e-9:
      return self._state.copy()

    # Propagate state and get state transition matrix
    x_pred, Phi = self._propagate_state(
      x  = self._state.x,
      t0 = self._state.time_s,
      tf = time_s,
    )

    # Propagate covariance: P_pred = Phi @ P @ Phi^T + Q
    Q = self._compute_process_noise(dt)
    P_pred = Phi @ self._state.P @ Phi.T + Q

    # Ensure symmetry
    P_pred = 0.5 * (P_pred + P_pred.T)

    # Update state
    self._state = EKFState(x=x_pred, P=P_pred, time_s=time_s)
    return self._state.copy()

  def update(
      self,
      measurement: EKFMeasurement,
    ) -> EKFState:
    """
    Update the state estimate with a measurement.

    Computes the Kalman gain and updates the state estimate and
    covariance using the measurement.

    Input:
    ------
      measurement : EKFMeasurement
        Measurement data including observation, covariance, and tracker info.

    Returns:
    --------
      state : EKFState
        Updated state estimate.
    """
    if self._state is None:
      raise RuntimeError("Filter not initialized. Call initialize() first.")

    # Predict to measurement time if needed
    if abs(measurement.time_s - self._state.time_s) > 1e-9:
      self.predict(measurement.time_s)

    # Compute predicted measurement and measurement Jacobian
    z_pred, H = self._compute_measurement(
      x       = self._state.x,
      tracker = measurement.tracker,
      types   = measurement.types,
    )

    # Innovation (measurement residual)
    y = measurement.z - z_pred

    # Handle angle wrapping for azimuth
    if 'azimuth' in measurement.types:
      az_idx = measurement.types.index('azimuth')
      y[az_idx] = self._wrap_angle(y[az_idx])

    # Innovation covariance: S = H @ P @ H^T + R
    S = H @ self._state.P @ H.T + measurement.R

    # Store residual and innovation covariance for later analysis
    self._residuals.append(y.copy())
    self._innovation_covariances.append(S.copy())
    self._measurement_times.append(measurement.time_s)

    # Kalman gain: K = P @ H^T @ S^(-1)
    K = self._state.P @ H.T @ np.linalg.inv(S)

    # State update: x = x + K @ y
    x_upd = self._state.x + K @ y

    # Covariance update
    if self.config.use_joseph_form:
      # Joseph form (more numerically stable):
      # P = (I - K @ H) @ P @ (I - K @ H)^T + K @ R @ K^T
      I_KH  = np.eye(self._state.n_states) - K @ H
      P_upd = I_KH @ self._state.P @ I_KH.T + K @ measurement.R @ K.T
    else:
      # Standard form: P = (I - K @ H) @ P
      P_upd = (np.eye(self._state.n_states) - K @ H) @ self._state.P

    # Ensure symmetry
    P_upd = 0.5 * (P_upd + P_upd.T)

    # Update state
    self._state = EKFState(x=x_upd, P=P_upd, time_s=measurement.time_s)
    self._history.append(self._state.copy())

    return self._state.copy()

  def process_measurements(
    self,
    measurements: List[EKFMeasurement],
  ) -> List[EKFState]:
    """
    Process a sequence of measurements.

    Input:
    ------
      measurements : List[EKFMeasurement]
        List of measurements sorted by time.

    Returns:
    --------
      states : List[EKFState]
        State estimates after each measurement update.
    """
    states = []
    for meas in sorted(measurements, key=lambda m: m.time_s):
      self.predict(meas.time_s)
      state = self.update(meas)
      states.append(state)
    return states

  # ==================== Private Methods ====================

  def _propagate_state(
    self,
    x  : np.ndarray,
    t0 : float,
    tf : float,
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate state and compute state transition matrix.

    If a custom propagator is configured, uses that. Otherwise,
    uses a simple two-body propagation.

    Input:
    ------
      x  : State vector at t0
      t0 : Initial time [s]
      tf : Final time [s]

    Returns:
    --------
      x_f : State vector at tf
      Phi : State transition matrix (6x6)
    """
    if self.config.propagator is not None:
      return self.config.propagator(x, t0, tf)
    else:
      # Default: two-body propagation with STM
      return self._two_body_propagate(x, t0, tf)

  def _two_body_propagate(
    self,
    x  : np.ndarray,
    t0 : float,
    tf : float,
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple two-body propagation with STM computation.

    Uses numerical integration of the equations of motion and
    variational equations.

    Input:
    ------
      x  : State vector [r, v] at t0
      t0 : Initial time [s]
      tf : Final time [s]

    Returns:
    --------
      x_f : State vector at tf
      Phi : State transition matrix (6x6)
    """

    # Earth GM
    GM = 3.986004415e14  # m³/s²

    def eom_with_stm(t, y):
      """Equations of motion with STM variational equations."""
      # State
      r_vec = y[0:3]
      v_vec = y[3:6]
      Phi_flat = y[6:42]
      Phi = Phi_flat.reshape((6, 6))

      # Position magnitude
      r = np.linalg.norm(r_vec)
      r3 = r**3
      r5 = r**5

      # Two-body acceleration
      a_vec = -GM * r_vec / r3

      # State derivative
      dx = np.zeros(6)
      dx[0:3] = v_vec
      dx[3:6] = a_vec

      # A matrix (Jacobian of f w.r.t. x)
      A = np.zeros((6, 6))
      A[0:3, 3:6] = np.eye(3)

      # Gravity gradient
      A[3:6, 0:3] = -GM / r3 * np.eye(3) + 3 * GM / r5 * np.outer(r_vec, r_vec)

      # STM derivative: Phi_dot = A @ Phi
      dPhi = A @ Phi

      # Combined derivative
      dy = np.zeros(42)
      dy[0:6] = dx
      dy[6:42] = dPhi.flatten()

      return dy

    # Initial conditions
    Phi0 = np.eye(6)
    y0 = np.zeros(42)
    y0[0:6] = x
    y0[6:42] = Phi0.flatten()

    # Integrate
    sol = solve_ivp(
      eom_with_stm,
      [t0, tf],
      y0,
      method = 'DOP853',
      rtol   = 1e-12,
      atol   = 1e-12,
    )

    # Extract final state and STM
    y_f = sol.y[:, -1]
    x_f = y_f[0:6]
    Phi = y_f[6:42].reshape((6, 6))

    return x_f, Phi

  def _compute_process_noise(self, dt: float) -> np.ndarray:
    """
    Compute process noise covariance matrix Q.

    Uses a simple model where position and velocity noise
    are uncorrelated.

    Input:
    ------
      dt : Time step [s]

    Returns:
    --------
      Q : Process noise covariance (6x6)
    """
    assert self._state is not None, "Filter not initialized"
    n = self._state.n_states
    Q = np.zeros((n, n))

    # Position noise (grows with dt)
    sigma_pos = self.config.process_noise_pos * abs(dt)
    Q[0, 0] = sigma_pos**2
    Q[1, 1] = sigma_pos**2
    Q[2, 2] = sigma_pos**2

    # Velocity noise (grows with sqrt(dt))
    sigma_vel = self.config.process_noise_vel * np.sqrt(abs(dt))
    Q[3, 3] = sigma_vel**2
    Q[4, 4] = sigma_vel**2
    Q[5, 5] = sigma_vel**2

    return Q

  def _compute_measurement(
    self,
    x       : np.ndarray,
    tracker : TrackerStation,
    types   : List[str],
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute predicted measurement and measurement Jacobian.

    Input:
    ------
      x       : State vector
      tracker : Tracker station
      types   : List of measurement types

    Returns:
    --------
      z_pred : Predicted measurement vector
      H      : Measurement Jacobian (dz/dx)
    """
    assert self._state is not None, "Filter not initialized"

    # Get current time ET
    current_et = self._epoch_et + self._state.time_s

    # Transform state to body-fixed frame
    pos_j2000 = x[0:3]
    vel_j2000 = x[3:6]

    pos_bf, vel_bf = VectorConverter.j2000_to_iau_earth(
      j2000_pos_vec = pos_j2000,
      j2000_vel_vec = vel_j2000,
      time_et       = current_et,
    )

    # Compute topocentric coordinates
    az, el, rng, az_dot, el_dot, rng_dot = TopocentricConverter.posvel_to_topocentric(
      sat_pos_vec = pos_bf,
      sat_vel_vec = vel_bf,
      tracker_lat = tracker.position.latitude,
      tracker_lon = tracker.position.longitude,
      tracker_alt = tracker.position.altitude,
    )

    # Build measurement vector based on types
    z_pred = []
    for t in types:
      if t == 'range':
        z_pred.append(rng)
      elif t == 'azimuth':
        z_pred.append(az)
      elif t == 'elevation':
        z_pred.append(el)
      elif t == 'range_dot':
        z_pred.append(rng_dot)
      elif t == 'azimuth_dot':
        z_pred.append(az_dot)
      elif t == 'elevation_dot':
        z_pred.append(el_dot)
      else:
        raise ValueError(f"Unknown measurement type: {t}")

    z_pred = np.array(z_pred)

    # Compute Jacobian numerically
    H = self._compute_measurement_jacobian(x, tracker, types)

    return z_pred, H

  def _compute_measurement_jacobian(
    self,
    x       : np.ndarray,
    tracker : TrackerStation,
    types   : List[str],
    eps     : float = 1.0,
  ) -> np.ndarray:
    """
    Compute measurement Jacobian numerically using finite differences.

    Input:
    ------
      x       : State vector
      tracker : Tracker station
      types   : List of measurement types
      eps     : Perturbation size [m for position, m/s for velocity]

    Returns:
    --------
      H : Measurement Jacobian [n_meas x n_states]
    """
    n_meas = len(types)
    n_states = len(x)
    H = np.zeros((n_meas, n_states))

    # Baseline measurement
    z0, _ = self._compute_measurement_no_jacobian(x, tracker, types)

    # Perturb each state element
    for i in range(n_states):
      # Use larger perturbation for position, smaller for velocity
      if i < 3:
        delta = eps  # Position: 1 m
      else:
        delta = eps * 1e-3  # Velocity: 1 mm/s

      x_pert = x.copy()
      x_pert[i] += delta

      z_pert, _ = self._compute_measurement_no_jacobian(x_pert, tracker, types)

      # Handle angle wrapping for azimuth
      dz = z_pert - z0
      if 'azimuth' in types:
        az_idx = types.index('azimuth')
        dz[az_idx] = self._wrap_angle(dz[az_idx])

      H[:, i] = dz / delta

    return H

  def _compute_measurement_no_jacobian(
    self,
    x       : np.ndarray,
    tracker : TrackerStation,
    types   : List[str],
  ) -> Tuple[np.ndarray, None]:
    """
    Compute predicted measurement without Jacobian (for finite differencing).

    Input:
    ------
      x       : State vector
      tracker : Tracker station
      types   : List of measurement types

    Returns:
    --------
      z_pred : Predicted measurement vector
      None   : Placeholder for Jacobian (not computed)
    """
    assert self._state is not None, "Filter not initialized"

    # Get current time ET
    current_et = self._epoch_et + self._state.time_s

    # Transform state to body-fixed frame
    pos_j2000 = x[0:3]
    vel_j2000 = x[3:6]

    pos_bf, vel_bf = VectorConverter.j2000_to_iau_earth(
      j2000_pos_vec = pos_j2000,
      j2000_vel_vec = vel_j2000,
      time_et       = current_et,
    )

    # Compute topocentric coordinates
    az, el, rng, az_dot, el_dot, rng_dot = TopocentricConverter.posvel_to_topocentric(
      sat_pos_vec = pos_bf,
      sat_vel_vec = vel_bf,
      tracker_lat = tracker.position.latitude,
      tracker_lon = tracker.position.longitude,
      tracker_alt = tracker.position.altitude,
    )

    # Build measurement vector based on types
    z_pred = []
    for t in types:
      if t == 'range':
        z_pred.append(rng)
      elif t == 'azimuth':
        z_pred.append(az)
      elif t == 'elevation':
        z_pred.append(el)
      elif t == 'range_dot':
        z_pred.append(rng_dot)
      elif t == 'azimuth_dot':
        z_pred.append(az_dot)
      elif t == 'elevation_dot':
        z_pred.append(el_dot)
      else:
        raise ValueError(f"Unknown measurement type: {t}")

    return np.array(z_pred), None

  @staticmethod
  def _wrap_angle(angle: float) -> float:
    """
    Wrap angle to [-pi, pi] range.

    Input:
    ------
      angle : Angle [rad]

    Returns:
    --------
      wrapped : Angle wrapped to [-pi, pi] [rad]
    """
    while angle > np.pi:
      angle -= 2 * np.pi
    while angle < -np.pi:
      angle += 2 * np.pi
    return angle
