"""
Propagation Schemas
===================

Dataclasses for propagation inputs, outputs, and configuration.
"""

from dataclasses import dataclass, field
from datetime    import datetime
from typing      import Optional, TYPE_CHECKING
import numpy as np

from src.schemas.state import ClassicalOrbitalElements, ModifiedEquinoctialElements


@dataclass
class TimeGrid:
  """
  Time grid for propagation.
  
  Attributes:
    epoch_dt : Reference epoch as datetime (UTC)
    epoch_et : Reference epoch as ephemeris time [s past J2000]
    time_s   : Time values relative to epoch [s], shape (N,)
    time_et  : Absolute ephemeris times [s past J2000], shape (N,)
  """
  epoch_dt : datetime
  epoch_et : float
  time_s   : np.ndarray
  time_et  : Optional[np.ndarray] = None
  
  def __post_init__(self):
    self.time_s = np.asarray(self.time_s)
    if self.time_et is None:
      self.time_et = self.epoch_et + self.time_s
  
  @property
  def n_points(self) -> int:
    return len(self.time_s)
  
  @property
  def duration_s(self) -> float:
    return self.time_s[-1] - self.time_s[0]


@dataclass
class PropagationResult:
  """
  Result of orbit propagation.
  
  Attributes:
    success        : Whether propagation completed successfully
    message        : Status or error message
    time_grid      : Time grid used for propagation
    time           : Time array [s] (raw output from integrator/SGP4)
    state          : Cartesian state array, shape (6, N)
    coe            : Classical orbital elements at each time
    mee            : Modified equinoctial elements at each time
    at_ephem_times : Results interpolated to ephemeris times (optional)
    plot_time_s    : Time array for plotting [s from epoch] (optional)
    integ_time_et  : Integration time in ET (optional)
    integ_time_s   : Integration time in seconds from TLE epoch (optional)
  """
  success        : bool
  message        : str                  = ""
  time_grid      : Optional[TimeGrid]   = None
  time           : Optional[np.ndarray] = None
  state          : Optional[np.ndarray] = None
  coe            : Optional[ClassicalOrbitalElements]    = None
  mee            : Optional[ModifiedEquinoctialElements] = None
  at_ephem_times : Optional[dict]       = None
  plot_time_s    : Optional[np.ndarray] = None
  integ_time_et  : Optional[np.ndarray] = None
  integ_time_s   : Optional[np.ndarray] = None


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
