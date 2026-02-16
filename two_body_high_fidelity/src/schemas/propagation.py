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
from src.schemas.time  import TimeStructure


@dataclass
class PropagationResult:
  """
  Result of orbit propagation.

  Attributes:
    success        : Whether propagation completed successfully
    message        : Status or error message
    time           : TimeStructure with initial, final, grid, deltas, duration
    state          : Cartesian state array, shape (6, N)
    coe            : Classical orbital elements at each time
    mee            : Modified equinoctial elements at each time
    at_ephem_times : Results interpolated to ephemeris times (optional, PropagationResult)

  Usage:
    - Use time.grid.relative_initial for plotting times (seconds relative to epoch)
    - Use time.grid.et for absolute ephemeris times (seconds past J2000)
    - Use time.grid.utc for absolute datetime objects
  """
  success        : bool
  message        : str                            = ""
  time           : Optional[TimeStructure]        = None
  state          : Optional[np.ndarray]           = None
  coe            : Optional[ClassicalOrbitalElements]    = None
  mee            : Optional[ModifiedEquinoctialElements] = None
  at_ephem_times : Optional['PropagationResult']  = None


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
