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
from src.model.time_converter import utc_to_et, et_to_utc


# ---------------------------------------------------------------------------
# Helper classes
# ---------------------------------------------------------------------------

class TimePoint:
  """
  Single absolute time with .utc (datetime) and .et (float) accessors.
  Bidirectional: construct with either utc or et, the other is computed lazily.
  """
  def __init__(self, utc: Optional[datetime] = None, et: Optional[float] = None):
    self._utc = utc
    self._et  = et

  @property
  def utc(self) -> datetime:
    if self._utc is None:
      self._utc = et_to_utc(self._et)
    return self._utc

  @property
  def et(self) -> float:
    if self._et is None:
      self._et = utc_to_et(self._utc)
    return self._et


class TimeGrid:
  """
  Array of absolute times with .utc, .et, and .relative_initial accessors.

  Attributes:
    relative_initial : Time offsets relative to initial [s], shape (N,)
    utc              : Absolute times as datetime (UTC), shape (N,), cached
    et               : Absolute ephemeris times [s past J2000], shape (N,), cached
  """
  def __init__(self, initial_utc: datetime, relative_initial: np.ndarray):
    self.relative_initial = np.asarray(relative_initial)
    self._initial_utc = initial_utc
    self._utc: Optional[np.ndarray] = None
    self._et:  Optional[np.ndarray] = None

  @property
  def utc(self) -> np.ndarray:
    if self._utc is None:
      self._utc = np.array([
        self._initial_utc + timedelta(seconds=float(dt))
        for dt in self.relative_initial
      ])
    return self._utc

  @property
  def et(self) -> np.ndarray:
    if self._et is None:
      self._et = utc_to_et(self._initial_utc) + self.relative_initial
    return self._et


# ---------------------------------------------------------------------------
# Time — top-level container (replaces old TimeGrid)
# ---------------------------------------------------------------------------

class Time:
  """
  Time container for propagation.

  Constructor:
    Time(initial=datetime, grid_relative_initial=float_array)

  Nested accessors:
    time.initial.utc              # datetime
    time.initial.et               # float [s past J2000]
    time.final.utc                # datetime
    time.final.et                 # float [s past J2000]
    time.grid.relative_initial    # np.ndarray [s], offsets from initial, shape (N,)
    time.grid.utc                 # np.ndarray of datetime, shape (N,)
    time.grid.et                  # np.ndarray of ET floats, shape (N,)
    time.deltas                   # np.ndarray [s], step sizes between consecutive points, shape (N-1,)
    time.duration                 # float [s]

  Invariants:
    - initial.utc <= final.utc
    - grid.utc[0] == initial.utc
    - grid.relative_initial[0] == 0.0
  """
  def __init__(self, initial: datetime, grid_relative_initial: np.ndarray):
    grid_relative_initial = np.asarray(grid_relative_initial)
    self._initial = TimePoint(utc=initial)
    self._final   = TimePoint(utc=initial + timedelta(seconds=float(grid_relative_initial[-1])))
    self._grid    = TimeGrid(initial_utc=initial, relative_initial=grid_relative_initial)
    self._deltas: Optional[np.ndarray] = None
    self._duration: Optional[float]    = None

  @property
  def initial(self) -> TimePoint:
    return self._initial

  @property
  def final(self) -> TimePoint:
    return self._final

  @property
  def grid(self) -> TimeGrid:
    return self._grid

  @property
  def deltas(self) -> np.ndarray:
    """Step sizes between consecutive grid points [s], shape (N-1,)."""
    if self._deltas is None:
      self._deltas = np.diff(self._grid.relative_initial)
    return cast(np.ndarray, self._deltas)

  @property
  def duration(self) -> float:
    """Total duration [s]."""
    if self._duration is None:
      self._duration = float(self._grid.relative_initial[-1])
    return cast(float, self._duration)


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
    - Use time_grid.grid.relative_initial for plotting times (seconds relative to epoch)
    - Use time_grid.grid.et for absolute ephemeris times (seconds past J2000)
    - Use time_grid.grid.utc for absolute datetime objects
  """
  success        : bool
  message        : str                  = ""
  time_grid      : Optional[Time]       = None
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
