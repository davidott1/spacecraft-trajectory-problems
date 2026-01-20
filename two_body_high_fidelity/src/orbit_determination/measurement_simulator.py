"""
Measurement Simulator for Orbit Determination
==============================================

Simulates ground-based tracking measurements (azimuth, elevation, range)
with configurable measurement noise.
"""

import datetime
import numpy as np

from typing import Optional

from src.model.constants       import CONVERTER
from src.model.frame_converter import VectorConverter
from src.model.orbit_converter import TopocentricConverter
from src.model.time_converter  import utc_to_et
from src.schemas.measurement   import MeasurementNoise, TopocentricState, SimulatedMeasurements
from src.schemas.propagation   import PropagationResult
from src.schemas.state         import TrackerStation


class MeasurementSimulator:
  """
  Simulate ground-based tracking measurements from a propagation result.

  Computes true topocentric coordinates, then optionally adds
  Gaussian measurement noise to simulate realistic tracker observations.

  Example:
  --------
    simulator    = MeasurementSimulator(result, tracker, epoch_dt)
    noise        = MeasurementNoise.from_degrees(azimuth_deg=0.01, elevation_deg=0.01, range_m=5.0)
    measurements = simulator.simulate(noise)
  """

  def __init__(
    self,
    result       : PropagationResult,
    tracker      : TrackerStation,
    epoch_dt_utc : Optional[datetime.datetime] = None,
  ):
    """
    Initialize the measurement simulator.

    Input:
    ------
      result : PropagationResult
        Propagation result containing state (6xN) and plot_delta_time.
      tracker : TrackerStation
        Ground tracking station with position and performance limits.
      epoch_dt_utc : datetime, optional
        Reference epoch for time conversion to ET.
    """
    self.result       = result
    self.tracker      = tracker
    self.epoch_dt_utc = epoch_dt_utc

    # Convert epoch to ET
    if epoch_dt_utc is not None:
      self.epoch_et = utc_to_et(epoch_dt_utc)
    else:
      self.epoch_et = 0.0

    # Cache for computed truth values
    self._truth        : Optional[TopocentricState] = None
    self._visible_mask : Optional[np.ndarray]       = None

  def compute_truth(
      self, 
      include_rates: bool = True,
    ) -> TopocentricState:
    """
    Compute true topocentric coordinates and optionally rates.

    Input:
    ------
      include_rates : bool
        If True, also compute azimuth_dot, elevation_dot, range_dot.

    Returns:
    --------
      truth : TopocentricState
        True azimuth, elevation, range (and rates if requested).
    """
    if self._truth is not None:
      return self._truth

    # Extract state vectors
    j2000_state   = self.result.state
    j2000_pos_vec = j2000_state[0:3, :]
    j2000_vel_vec = j2000_state[3:6, :]
    time_s        = self.result.time_grid.deltas
    n_points      = j2000_state.shape[1]

    # Transform positions and velocities from J2000 to body-fixed frame
    sat_pos_iau_earth_array = np.zeros((3, n_points))
    sat_vel_iau_earth_array = np.zeros((3, n_points))

    for i in range(n_points):
      et_i = self.epoch_et + time_s[i]
      sat_pos_iau_earth_array[:, i], sat_vel_iau_earth_array[:, i] = VectorConverter.j2000_to_iau_earth(
        j2000_pos_vec = j2000_pos_vec[:, i],
        j2000_vel_vec = j2000_vel_vec[:, i],
        time_et       = et_i,
      )

    if include_rates:
      # Use TopocentricConverter for az, el, range AND rates
      azimuth, elevation, range_arr, az_dot, el_dot, rng_dot = TopocentricConverter.posvel_to_topocentric_array(
        sat_pos_array = sat_pos_iau_earth_array,
        sat_vel_array = sat_vel_iau_earth_array,
        tracker_lat   = self.tracker.position.latitude,
        tracker_lon   = self.tracker.position.longitude,
        tracker_alt   = self.tracker.position.altitude,
      )

      # Store and return
      self._truth = TopocentricState(
        delta_time_epoch = time_s,
        azimuth          = azimuth,
        elevation        = elevation,
        range            = range_arr,
        azimuth_dot      = az_dot,
        elevation_dot    = el_dot,
        range_dot        = rng_dot,
      )
    else:
      # Use TopocentricConverter for az, el, range only
      azimuth, elevation, range_arr = TopocentricConverter.pos_to_topocentric_array(
        sat_pos_array = sat_pos_iau_earth_array,
        tracker_lat   = self.tracker.position.latitude,
        tracker_lon   = self.tracker.position.longitude,
        tracker_alt   = self.tracker.position.altitude,
      )

      # Store and return
      self._truth = TopocentricState(
        delta_time_epoch = time_s,
        azimuth          = azimuth,
        elevation        = elevation,
        range            = range_arr,
      )

    return self._truth

  def compute_visibility_mask(self) -> np.ndarray:
    """
    Compute visibility mask based on tracker performance limits.

    Returns:
    --------
      visible_mask : np.ndarray
        Boolean mask where True indicates visible points.
    """
    if self._visible_mask is not None:
      return self._visible_mask

    truth = self.compute_truth()
    n_points = truth.n_points

    # Start with all visible
    visible_mask = np.ones(n_points, dtype=bool)

    # Apply tracker performance constraints if available
    if self.tracker.performance is not None and self.tracker.performance.constraints is not None:
      constraints = self.tracker.performance.constraints

      # Elevation limits
      if constraints.elevation is not None:
        visible_mask &= (truth.elevation >= constraints.elevation.min)
        if constraints.elevation.max is not None:
          visible_mask &= (truth.elevation <= constraints.elevation.max)

      # Range limits
      if constraints.range is not None:
        if constraints.range.min is not None:
          visible_mask &= (truth.range >= constraints.range.min)
        if constraints.range.max is not None:
          visible_mask &= (truth.range <= constraints.range.max)

      # Azimuth limits (handle wraparound and full range)
      if constraints.azimuth is not None:
        az_min = constraints.azimuth.min
        az_max = constraints.azimuth.max
        # Check if full azimuth range (all angles valid)
        az_min_deg = az_min * CONVERTER.DEG_PER_RAD
        az_max_deg = az_max * CONVERTER.DEG_PER_RAD
        full_range = (az_max_deg - az_min_deg) >= 360.0 or (az_min_deg == -180.0 and az_max_deg == 180.0)
        if not full_range:
          if az_min <= az_max:
            visible_mask &= (truth.azimuth >= az_min) & (truth.azimuth <= az_max)
          else:
            # Wraparound case (e.g., 350° to 10°)
            visible_mask &= (truth.azimuth >= az_min) | (truth.azimuth <= az_max)

    # Store and return
    self._visible_mask = visible_mask
    return visible_mask

  def get_tracker_noise_config(self) -> Optional[MeasurementNoise]:
    """
    Create MeasurementNoise from tracker uncertainty configuration.

    Returns:
    --------
      noise_config : MeasurementNoise | None
        MeasurementNoise object with tracker uncertainty values,
        or None if tracker has no uncertainty configured.

    Example:
    --------
      simulator = MeasurementSimulator(result, tracker, epoch_dt)

      # For JPL Horizons truth - apply tracker uncertainty
      noise = simulator.get_tracker_noise_config()
      measurements = simulator.simulate(noise_config=noise)

      # For TLE/high-fidelity - no uncertainty
      measurements = simulator.simulate()  # noise_config=None by default
    """
    if self.tracker.performance is not None and self.tracker.performance.uncertainty is not None:
      unc = self.tracker.performance.uncertainty
      return MeasurementNoise(
        azimuth       = unc.azimuth,
        elevation     = unc.elevation,
        range         = unc.range,
        azimuth_dot   = unc.azimuth_rate,
        elevation_dot = unc.elevation_rate,
        range_dot     = unc.range_rate,
      )
    return None

  def simulate(
    self,
    noise_config  : Optional[MeasurementNoise] = None,
    seed          : Optional[int] = None,
    include_rates : bool = True,
  ) -> SimulatedMeasurements:
    """
    Simulate measurements with optional Gaussian noise.

    By default (noise_config=None), no noise is applied - measurements are pure
    geometric calculations from the state. This is appropriate for TLE and
    high-fidelity propagated solutions.

    For JPL Horizons truth comparison, explicitly pass noise_config with
    tracker uncertainty to generate simulated measurements.

    Input:
    ------
      noise_config : MeasurementNoise, optional
        Noise standard deviations. If None (default), no noise is added.
        To apply tracker uncertainty, create MeasurementNoise from
        tracker.performance.uncertainty and pass explicitly.
      seed : int, optional
        Random seed for reproducibility.
      include_rates : bool
        If True, include rate measurements (az_dot, el_dot, range_dot).

    Returns:
    --------
      measurements : SimulatedMeasurements
        Object containing:
        - truth: Pure geometric calculations (no noise)
        - measured: Same as truth if noise_config=None, otherwise truth + noise
        - noise_config: The noise configuration used
        - visible_mask: Boolean mask for visibility constraints
        - tracker: The tracker station
    """
    # Set random seed for reproducibility
    if seed is not None:
      np.random.seed(seed)

    # Compute true topocentric coordinates (and optionally rates) and visibility
    truth        = self.compute_truth(include_rates=include_rates)
    visible_mask = self.compute_visibility_mask()

    # Default to zero noise if no noise config provided
    # Uncertainty should only be applied when explicitly requested
    # (e.g., for JPL Horizons truth measurements)
    if noise_config is None:
      noise_config = MeasurementNoise()

    # Compute number of points
    n_points = truth.n_points

    # Generate noisy measurements by adding Gaussian noise to truth
    if include_rates and truth.has_rates:
      measured = TopocentricState(
        delta_time_epoch = truth.delta_time_epoch.copy(),
        azimuth          = truth.azimuth       + np.random.randn(n_points) * noise_config.azimuth,
        elevation        = truth.elevation     + np.random.randn(n_points) * noise_config.elevation,
        range            = truth.range         + np.random.randn(n_points) * noise_config.range,
        azimuth_dot      = truth.azimuth_dot   + np.random.randn(n_points) * noise_config.azimuth_dot,
        elevation_dot    = truth.elevation_dot + np.random.randn(n_points) * noise_config.elevation_dot,
        range_dot        = truth.range_dot     + np.random.randn(n_points) * noise_config.range_dot,
      )
    else:
      measured = TopocentricState(
        delta_time_epoch = truth.delta_time_epoch.copy(),
        azimuth          = truth.azimuth   + np.random.randn(n_points) * noise_config.azimuth,
        elevation        = truth.elevation + np.random.randn(n_points) * noise_config.elevation,
        range            = truth.range     + np.random.randn(n_points) * noise_config.range,
      )

    # Combine truth, measurements, config, visibility, tracker
    return SimulatedMeasurements(
      truth        = truth,
      measured     = measured,
      noise_config = noise_config,
      visible_mask = visible_mask,
      tracker      = self.tracker,
    )
