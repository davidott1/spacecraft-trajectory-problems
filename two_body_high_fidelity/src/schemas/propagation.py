"""
Propagation Schemas
===================

Dataclasses for propagation inputs, outputs, and configuration.
"""

from dataclasses import dataclass
from datetime    import datetime, timedelta
from typing      import Optional, cast
import numpy as np

from src.schemas.state      import ClassicalOrbitalElements, ModifiedEquinoctialElements
from src.model.time_converter import utc_to_et


@dataclass
class TimeGrid:
  """
  Time grid for propagation.

  Attributes:
    initial    : Initial time as datetime (UTC), size (1,)
    final      : Final time as datetime (UTC), size (1,)
    deltas     : Time deltas relative to initial [s], shape (N,)
    times      : Absolute times as datetime (UTC), shape (N,), computed as initial + deltas
    times_et   : Absolute ephemeris times [s past J2000], shape (N,), computed from initial + deltas
    steps      : Time steps between consecutive points [s], shape (N-1,), computed as diff(deltas)
    duration   : Total duration as timedelta, computed as final - initial

  Invariants:
    - initial   <= final
    - initial   <= times <= final
    - times[0]  == initial
    - times[-1] == final (approximately, within numerical precision)
    - times     == initial + deltas
  """
  initial : datetime
  final   : datetime
  deltas  : np.ndarray
  _times_et_cached : Optional[np.ndarray] = None
  _steps_cached    : Optional[np.ndarray] = None
  _duration_cached : Optional[timedelta]  = None

  def __post_init__(self):
    self.deltas = np.asarray(self.deltas)

  @property
  def times(self) -> np.ndarray:
    """
    Absolute times as datetime objects, computed from initial + deltas.
    Shape: (N,)
    """
    return np.array([self.initial + timedelta(seconds=float(dt)) for dt in self.deltas])

  @property
  def times_et(self) -> np.ndarray:
    """
    Absolute ephemeris times [s past J2000], computed from initial + deltas.
    Cached after first computation.
    Shape: (N,)
    """
    if self._times_et_cached is None:
      initial_et = utc_to_et(self.initial)
      self._times_et_cached = initial_et + self.deltas
    return cast(np.ndarray, self._times_et_cached)

  @property
  def steps(self) -> np.ndarray:
    """
    Time steps between consecutive points [s].
    Cached after first computation.
    Shape: (N-1,)
    """
    if self._steps_cached is None:
      self._steps_cached = np.diff(self.deltas)
    return cast(np.ndarray, self._steps_cached)

  @property
  def duration(self) -> timedelta:
    """
    Total duration as timedelta.
    Cached after first computation.
    """
    if self._duration_cached is None:
      self._duration_cached = self.final - self.initial
    return cast(timedelta, self._duration_cached)


@dataclass
class PropagationResult:
  """
  Result of orbit propagation.

  Attributes:
    success        : Whether propagation completed successfully
    message        : Status or error message
    time_grid      : Time grid used for propagation (contains deltas, times, times_et, etc.)
    state          : Cartesian state array, shape (6, N)
    coe            : Classical orbital elements at each time
    mee            : Modified equinoctial elements at each time
    at_ephem_times : Results interpolated to ephemeris times (optional, PropagationResult)

  Usage:
    - Use time_grid.deltas for plotting times (seconds relative to epoch)
    - Use time_grid.times_et for absolute ephemeris times (seconds past J2000)
    - Use time_grid.times for absolute datetime objects
  """
  success        : bool
  message        : str                  = ""
  time_grid      : Optional[TimeGrid]   = None
  state          : Optional[np.ndarray] = None
  coe            : Optional[ClassicalOrbitalElements]    = None
  mee            : Optional[ModifiedEquinoctialElements] = None
  at_ephem_times : Optional['PropagationResult'] = None


@dataclass
class PropagationConfig:
  """
  Configuration for orbit propagation.
  
  Attributes:
    time_o_dt : Start time (UTC datetime)
    time_f_dt : End time (UTC datetime)
    dt_output : Output time step [s]
    rtol      : Relative tolerance for integrator
    atol      : Absolute tolerance for integrator
    method    : Integration method (e.g., 'DOP853', 'RK45')
  """
  time_o_dt : datetime
  time_f_dt : datetime
  dt_output : float = 60.0
  rtol      : float = 1e-12
  atol      : float = 1e-12
  method    : str   = 'DOP853'
  
  @property
  def duration_s(self) -> float:
    """
    Total propagation duration in seconds.
    """
    return (self.time_f_dt - self.time_o_dt).total_seconds()
