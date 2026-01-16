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
from src.model.frame_converter import FrameConverter
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
        Propagation result containing state (6xN) and plot_time_s.
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

  def compute_truth(self) -> TopocentricState:
    """
    Compute true topocentric coordinates.

    Returns:
    --------
      truth : TopocentricState
        True azimuth, elevation, range.
    """
    if self._truth is not None:
      return self._truth

    # Extract state vectors
    j2000_state   = self.result.state
    j2000_pos_vec = j2000_state[0:3, :]
    time_s        = self.result.plot_time_s
    n_points      = j2000_state.shape[1]

    # Transform positions from J2000 to body-fixed frame
    sat_pos_bf_array = np.zeros((3, n_points))

    for i in range(n_points):
      et_i = self.epoch_et + time_s[i]
      rot_mat_j2000_to_iau_earth = FrameConverter.j2000_to_iau_earth(et_i)
      sat_pos_bf_array[:, i] = rot_mat_j2000_to_iau_earth @ j2000_pos_vec[:, i]

    # Use TopocentricConverter for az, el, range
    azimuth, elevation, range_arr = TopocentricConverter.pos_to_topocentric_array(
      sat_pos_array = sat_pos_bf_array,
      tracker_lat   = self.tracker.position.latitude,
      tracker_lon   = self.tracker.position.longitude,
      tracker_alt   = self.tracker.position.altitude,
    )

    # Store and return
    self._truth = TopocentricState(
      time_s    = time_s,
      azimuth   = azimuth,
      elevation = elevation,
      range     = range_arr,
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

    # Apply tracker performance limits if available
    if self.tracker.performance is not None:
      perf = self.tracker.performance

      # Elevation limits
      if perf.elevation is not None:
        visible_mask &= (truth.elevation >= perf.elevation.min)
        if perf.elevation.max is not None:
          visible_mask &= (truth.elevation <= perf.elevation.max)

      # Range limits
      if perf.range is not None:
        if perf.range.min is not None:
          visible_mask &= (truth.range >= perf.range.min)
        if perf.range.max is not None:
          visible_mask &= (truth.range <= perf.range.max)

      # Azimuth limits (handle wraparound and full range)
      if perf.azimuth is not None:
        az_min = perf.azimuth.min
        az_max = perf.azimuth.max
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

  def simulate(
    self,
    noise_config : Optional[MeasurementNoise] = None,
    seed         : Optional[int] = None,
  ) -> SimulatedMeasurements:
    """
    Simulate measurements with optional Gaussian noise.

    Input:
    ------
      noise_config : MeasurementNoise, optional
        Noise standard deviations. If None, no noise is added.
      seed : int, optional
        Random seed for reproducibility.

    Returns:
    --------
      measurements : SimulatedMeasurements
        Simulated measurements with truth and noisy values.
    """
    # Set random seed for reproducibility
    if seed is not None:
      np.random.seed(seed)

    # Compute true topocentric coordinates and visibility
    truth        = self.compute_truth()
    visible_mask = self.compute_visibility_mask()

    # Use zero noise if no noise config provided
    if noise_config is None:
      noise_config = MeasurementNoise()

    # Compute number of points
    n_points = truth.n_points

    # Generate noisy measurements by adding Gaussian noise to truth
    measured = TopocentricState(
      time_s    = truth.time_s.copy(),
      azimuth   = truth.azimuth   + np.random.randn(n_points) * noise_config.azimuth,
      elevation = truth.elevation + np.random.randn(n_points) * noise_config.elevation,
      range     = truth.range     + np.random.randn(n_points) * noise_config.range,
    )

    # Combine truth, measurements, config, visibility, tracker
    return SimulatedMeasurements(
      truth        = truth,
      measured     = measured,
      noise_config = noise_config,
      visible_mask = visible_mask,
      tracker      = self.tracker,
    )
