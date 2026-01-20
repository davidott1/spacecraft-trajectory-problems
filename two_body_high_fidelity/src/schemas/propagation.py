"""
Propagation Schemas
===================

Dataclasses for propagation inputs, outputs, and configuration.
"""

from dataclasses import dataclass
from datetime    import datetime
from typing      import Optional
import numpy as np

from src.schemas.state import ClassicalOrbitalElements, ModifiedEquinoctialElements


@dataclass
class TimeGrid:
  """
  Time grid for propagation.

  Attributes:
    epoch_dt         : reference epoch as datetime (UTC)
    epoch_et         : reference epoch as ephemeris time [s past J2000]
    delta_time_epoch : time values relative to epoch, shape (N,)
    time_et          : absolute ephemeris times [s past J2000], shape (N,)
  """
  epoch_dt         : datetime
  epoch_et         : float
  delta_time_epoch : np.ndarray
  time_et          : Optional[np.ndarray] = None

  def __post_init__(self):
    self.delta_time_epoch = np.asarray(self.delta_time_epoch)
    if self.time_et is None:
      self.time_et = self.epoch_et + self.delta_time_epoch

  @property
  def n_points(self) -> int:
    return len(self.delta_time_epoch)

  @property
  def duration_s(self) -> float:
    return self.delta_time_epoch[-1] - self.delta_time_epoch[0]


@dataclass
class PropagationResult:
  """
  Result of orbit propagation.

  Attributes:
    success              : Whether propagation completed successfully
    message              : Status or error message
    time_grid            : Time grid used for propagation
    time                 : Time array (raw output from integrator/SGP4)
    state                : Cartesian state array, shape (6, N)
    coe                  : Classical orbital elements at each time
    mee                  : Modified equinoctial elements at each time
    at_ephem_times       : Results interpolated to ephemeris times (optional, PropagationResult)
    plot_delta_time      : Time array for plotting relative to epoch (optional)
    integ_time_et        : Integration time in ET (optional)
    integ_delta_time_tle : Integration time relative to TLE epoch (optional)
  """
  success              : bool
  message              : str                  = ""
  time_grid            : Optional[TimeGrid]   = None
  time                 : Optional[np.ndarray] = None
  state                : Optional[np.ndarray] = None
  coe                  : Optional[ClassicalOrbitalElements]    = None
  mee                  : Optional[ModifiedEquinoctialElements] = None
  at_ephem_times       : Optional['PropagationResult'] = None
  plot_delta_time      : Optional[np.ndarray] = None
  integ_time_et        : Optional[np.ndarray] = None
  integ_delta_time_tle : Optional[np.ndarray] = None


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
