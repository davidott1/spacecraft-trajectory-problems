"""
Ephemeris Schemas
=================

Dataclasses for ephemeris data from JPL Horizons and TLE.
"""

from dataclasses import dataclass
from datetime    import datetime
from typing      import Optional, Union

import numpy as np


@dataclass
class TLEData:
  """
  Two-Line Element set data.
  
  Attributes:
    line_1              : TLE line 1
    line_2              : TLE line 2
    epoch_dt            : TLE epoch as datetime (UTC)
    norad_id            : NORAD catalog ID
    object_name         : Satellite name
    bstar               : B* drag term [1/Earth radii]
    inclination         : Inclination [rad]
    raan                : Right ascension of ascending node [rad]
    eccentricity        : Eccentricity [-]
    argument_of_perigee : Argument of perigee [rad]
    mean_anomaly        : Mean anomaly [rad]
    mean_motion         : Mean motion [rev/day]
  """
  line_1              : str
  line_2              : str
  epoch_dt            : datetime
  norad_id            : Optional[str]   = None
  object_name         : Optional[str]   = None
  bstar               : Optional[float] = None
  inclination         : Optional[float] = None
  raan                : Optional[float] = None
  eccentricity        : Optional[float] = None
  argument_of_perigee : Optional[float] = None
  mean_anomaly        : Optional[float] = None
  mean_motion         : Optional[float] = None
  
  @property
  def tle_lines(self) -> tuple:
    """
    Return both TLE lines as tuple.
    """
    return (self.line_1, self.line_2)


@dataclass
class HorizonsEphemeris:
  """
  Ephemeris data from JPL Horizons.
  
  Attributes:
    success     : Whether query succeeded
    message     : Status or error message
    norad_id    : NORAD catalog ID (for satellites)
    object_name : Object name from Horizons
    epoch_dt    : Reference epoch
    time_s      : Time array relative to epoch [s]
    state       : State array (6, N) in J2000 frame [m, m/s]
    coe         : Classical orbital elements
    mee         : Modified equinoctial elements
  """
  success     : bool
  message     : str                  = ""
  norad_id    : Optional[str]        = None
  object_name : Optional[str]        = None
  epoch_dt    : Optional[datetime]   = None
  time_s      : Optional[np.ndarray] = None
  state       : Optional[np.ndarray] = None
  coe         : Optional[dict]       = None
  mee         : Optional[dict]       = None


@dataclass
class EphemerisResult:
  """
  Generic ephemeris result container.
  
  Attributes:
    success : Whether operation succeeded
    message : Status or error message
    source  : Data source ('horizons', 'tle', 'spice', etc.)
    data    : Source-specific data object
  """
  success : bool
  message : str                                         = ""
  source  : str                                         = ""
  data    : Optional[Union[HorizonsEphemeris, TLEData]] = None
