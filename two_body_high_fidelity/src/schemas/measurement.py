"""
Measurement Schemas
===================

Dataclasses for simulated and observed measurements from ground trackers.
"""
import numpy as np

from dataclasses import dataclass
from typing      import Optional

from src.model.constants import CONVERTER
from src.schemas.state   import TrackerStation


@dataclass
class MeasurementNoise:
  """
  Measurement noise standard deviations (1-sigma).

  Attributes:
    azimuth       : azimuth noise [rad]
    elevation     : elevation noise [rad]
    range         : range noise [m]
    azimuth_dot   : azimuth rate noise [rad/s]
    elevation_dot : elevation rate noise [rad/s]
    range_dot     : range rate noise [m/s]
  """
  azimuth       : float = 0.0
  elevation     : float = 0.0
  range         : float = 0.0
  azimuth_dot   : float = 0.0
  elevation_dot : float = 0.0
  range_dot     : float = 0.0

  @classmethod
  def from_degrees(
    cls,
    azimuth_deg       : float = 0.0,
    elevation_deg     : float = 0.0,
    range_m           : float = 0.0,
    azimuth_dot_dps   : float = 0.0,
    elevation_dot_dps : float = 0.0,
    range_dot_mps     : float = 0.0,
  ) -> 'MeasurementNoise':
    """
    Create MeasurementNoise from angle noise in degrees.
    
    Input:
    ------
      azimuth_deg : float
        Azimuth noise [deg]
      elevation_deg : float
        Elevation noise [deg]
      range_m : float
        Range noise [m]
      azimuth_dot_dps : float
        Azimuth rate noise [deg/s]
      elevation_dot_dps : float
        Elevation rate noise [deg/s]
      range_dot_mps : float
        Range rate noise [m/s]
    """
    return cls(
      azimuth       = azimuth_deg       * CONVERTER.RAD_PER_DEG,
      elevation     = elevation_deg     * CONVERTER.RAD_PER_DEG,
      range         = range_m,
      azimuth_dot   = azimuth_dot_dps   * CONVERTER.RAD_PER_DEG,
      elevation_dot = elevation_dot_dps * CONVERTER.RAD_PER_DEG,
      range_dot     = range_dot_mps,
    )
  
  @classmethod
  def from_radians(
    cls,
    azimuth_rad       : float = 0.0,
    elevation_rad     : float = 0.0,
    range_m           : float = 0.0,
    azimuth_dot_rad_per_s   : float = 0.0,
    elevation_dot_rad_per_s : float = 0.0,
    range_dot_m_per_s     : float = 0.0,
  ) -> 'MeasurementNoise':
    """
    Create MeasurementNoise from angle noise in radians.
    """
    return cls(
      azimuth       = azimuth_rad,
      elevation     = elevation_rad,
      range         = range_m,
      azimuth_dot   = azimuth_dot_rps,
      elevation_dot = elevation_dot_rps,
      range_dot     = range_dot_mps,
    )


@dataclass
class TopocentricState:
  """
  Topocentric coordinates and rates from a ground station.

  Attributes:
    time_s        : time array relative to epoch [s], shape (N,)
    azimuth       : azimuth angle [rad], shape (N,)
    elevation     : elevation angle [rad], shape (N,)
    range         : range [m], shape (N,)
    azimuth_dot   : azimuth rate [rad/s], shape (N,), optional
    elevation_dot : elevation rate [rad/s], shape (N,), optional
    range_dot     : range rate [m/s], shape (N,), optional
  """
  time_s        : np.ndarray
  azimuth       : np.ndarray
  elevation     : np.ndarray
  range         : np.ndarray
  azimuth_dot   : Optional[np.ndarray] = None
  elevation_dot : Optional[np.ndarray] = None
  range_dot     : Optional[np.ndarray] = None

  def __post_init__(self):
    self.time_s    = np.asarray(self.time_s)
    self.azimuth   = np.asarray(self.azimuth)
    self.elevation = np.asarray(self.elevation)
    self.range     = np.asarray(self.range)
    if self.azimuth_dot is not None:
      self.azimuth_dot = np.asarray(self.azimuth_dot)
    if self.elevation_dot is not None:
      self.elevation_dot = np.asarray(self.elevation_dot)
    if self.range_dot is not None:
      self.range_dot = np.asarray(self.range_dot)

  @property
  def n_points(self) -> int:
    return len(self.time_s)

  @property
  def has_rates(self) -> bool:
    """
    Check if rate data is available.
    """
    return self.azimuth_dot is not None and self.elevation_dot is not None and self.range_dot is not None


@dataclass
class SimulatedMeasurements:
  """
  Simulated measurements with truth and noisy values.

  Attributes:
    truth        : true topocentric state (no noise)
    measured     : noisy measurements
    noise_config : noise configuration used
    visible_mask : boolean mask for visible points, shape (N,)
    tracker      : tracker station
  """
  truth        : TopocentricState
  measured     : TopocentricState
  noise_config : MeasurementNoise
  visible_mask : np.ndarray
  tracker      : TrackerStation

  def __post_init__(self):
    self.visible_mask = np.asarray(self.visible_mask)

  @property
  def n_visible(self) -> int:
    return int(np.sum(self.visible_mask))

  def get_visible_truth(self) -> TopocentricState:
    """
    Return truth values only for visible points.
    """
    return TopocentricState(
      time_s        = self.truth.time_s[self.visible_mask],
      azimuth       = self.truth.azimuth[self.visible_mask],
      elevation     = self.truth.elevation[self.visible_mask],
      range         = self.truth.range[self.visible_mask],
      azimuth_dot   = self.truth.azimuth_dot[self.visible_mask]   if self.truth.azimuth_dot   is not None else None,
      elevation_dot = self.truth.elevation_dot[self.visible_mask] if self.truth.elevation_dot is not None else None,
      range_dot     = self.truth.range_dot[self.visible_mask]     if self.truth.range_dot     is not None else None,
    )

  def get_visible_measured(self) -> TopocentricState:
    """
    Return measured (noisy) values only for visible points.
    """
    return TopocentricState(
      time_s        = self.measured.time_s[self.visible_mask],
      azimuth       = self.measured.azimuth[self.visible_mask],
      elevation     = self.measured.elevation[self.visible_mask],
      range         = self.measured.range[self.visible_mask],
      azimuth_dot   = self.measured.azimuth_dot[self.visible_mask]   if self.measured.azimuth_dot   is not None else None,
      elevation_dot = self.measured.elevation_dot[self.visible_mask] if self.measured.elevation_dot is not None else None,
      range_dot     = self.measured.range_dot[self.visible_mask]     if self.measured.range_dot     is not None else None,
    )
