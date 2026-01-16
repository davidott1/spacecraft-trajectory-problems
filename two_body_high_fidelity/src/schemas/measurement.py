"""
Measurement Schemas
===================

Dataclasses for simulated and observed measurements from ground trackers.
"""
import numpy as np

from dataclasses import dataclass

from src.model.constants import CONVERTER
from src.schemas.state   import TrackerStation


@dataclass
class MeasurementNoise:
  """
  Measurement noise standard deviations (1-sigma).

  Attributes:
    azimuth   : azimuth noise [rad]
    elevation : elevation noise [rad]
    range     : range noise [m]
  """
  azimuth   : float = 0.0
  elevation : float = 0.0
  range     : float = 0.0

  @classmethod
  def from_degrees(
    cls,
    azimuth_deg   : float = 0.0,
    elevation_deg : float = 0.0,
    range_m       : float = 0.0,
  ) -> 'MeasurementNoise':
    """
    Create MeasurementNoise from angle noise in degrees.
    """
    return cls(
      azimuth   = azimuth_deg   * CONVERTER.RAD_PER_DEG,
      elevation = elevation_deg * CONVERTER.RAD_PER_DEG,
      range     = range_m,
    )
  
  @classmethod
  def from_radians(
    cls,
    azimuth_rad   : float = 0.0,
    elevation_rad : float = 0.0,
    range_m       : float = 0.0,
  ) -> 'MeasurementNoise':
    """
    Create MeasurementNoise from angle noise in radians.
    """
    return cls(
      azimuth   = azimuth_rad,
      elevation = elevation_rad,
      range     = range_m,
    )


@dataclass
class TopocentricState:
  """
  Topocentric coordinates from a ground station.

  Attributes:
    time_s    : time array relative to epoch [s], shape (N,)
    azimuth   : azimuth angle [rad], shape (N,)
    elevation : elevation angle [rad], shape (N,)
    range     : range [m], shape (N,)
  """
  time_s    : np.ndarray
  azimuth   : np.ndarray
  elevation : np.ndarray
  range     : np.ndarray

  def __post_init__(self):
    self.time_s    = np.asarray(self.time_s)
    self.azimuth   = np.asarray(self.azimuth)
    self.elevation = np.asarray(self.elevation)
    self.range     = np.asarray(self.range)

  @property
  def n_points(self) -> int:
    return len(self.time_s)


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
      time_s    = self.truth.time_s[self.visible_mask],
      azimuth   = self.truth.azimuth[self.visible_mask],
      elevation = self.truth.elevation[self.visible_mask],
      range     = self.truth.range[self.visible_mask],
    )

  def get_visible_measured(self) -> TopocentricState:
    """
    Return measured (noisy) values only for visible points.
    """
    return TopocentricState(
      time_s    = self.measured.time_s[self.visible_mask],
      azimuth   = self.measured.azimuth[self.visible_mask],
      elevation = self.measured.elevation[self.visible_mask],
      range     = self.measured.range[self.visible_mask],
    )
