"""
State Representation Schemas
============================

Dataclasses for orbital state representations including Cartesian,
classical orbital elements, modified equinoctial elements, and TLE data.
"""

from dataclasses import dataclass, field
from datetime    import datetime
from typing      import Optional, Union
import numpy as np


@dataclass
class CartesianState:
  """
  Cartesian state vector in an inertial frame.
  
  Attributes:
    position : Position vector [m], shape (3,) or (3, N)
    velocity : Velocity vector [m/s], shape (3,) or (3, N)
    frame    : Reference frame name (default "J2000")
  """
  position : np.ndarray
  velocity : np.ndarray
  frame    : str = "J2000"
  
  def __post_init__(self):
    self.position = np.asarray(self.position)
    self.velocity = np.asarray(self.velocity)
  
  @property
  def state_vector(self) -> np.ndarray:
    """
    Combined state vector [pos, vel], shape (6,) or (6, N).
    """
    if self.position.ndim == 1:
      return np.concatenate([self.position, self.velocity])
    return np.vstack([self.position, self.velocity])
  
  @classmethod
  def from_state_vector(cls, state: np.ndarray, frame: str = "J2000") -> 'CartesianState':
    """
    Create from combined state vector.
    """
    state = np.asarray(state)
    if state.ndim == 1:
      return cls(position=state[0:3], velocity=state[3:6], frame=frame)
    return cls(position=state[0:3, :], velocity=state[3:6, :], frame=frame)


@dataclass
class ClassicalOrbitalElements:
  """
  Classical (Keplerian) orbital elements.
  
  Attributes:
    sma  : Semi-major axis [m] (np.inf for parabolic)
    ecc  : Eccentricity [-]
    inc  : Inclination [rad]
    raan : Right ascension of ascending node [rad]
    aop  : Argument of periapsis [rad]
    ta   : True anomaly [rad]
    ea   : Eccentric anomaly [rad] (None for hyperbolic/parabolic)
    ma   : Mean anomaly [rad] (None for rectilinear/parabolic)
    ha   : Hyperbolic anomaly [rad] (None for elliptic/parabolic)
    pa   : Parabolic anomaly [rad] (None for elliptic/hyperbolic)
  """
  sma  : Union[float, np.ndarray]
  ecc  : Union[float, np.ndarray]
  inc  : Union[float, np.ndarray]
  raan : Union[float, np.ndarray]
  aop  : Union[float, np.ndarray]
  ta   : Union[float, np.ndarray]
  ea   : Optional[Union[float, np.ndarray]] = None
  ma   : Optional[Union[float, np.ndarray]] = None
  ha   : Optional[Union[float, np.ndarray]] = None
  pa   : Optional[Union[float, np.ndarray]] = None


@dataclass
class ModifiedEquinoctialElements:
  """
  Modified Equinoctial Elements (MEE).
  
  Avoids singularities at zero eccentricity and zero/180° inclination.
  
  Attributes:
    p : Semi-latus rectum [m]
    f : e*cos(ω + I*Ω) [-]
    g : e*sin(ω + I*Ω) [-]
    h : tan(i/2)*cos(Ω) or cot(i/2)*cos(Ω) [-]
    k : tan(i/2)*sin(Ω) or cot(i/2)*sin(Ω) [-]
    L : True longitude = ω + I*Ω + ν [rad]
    I : Retrograde factor (+1 prograde, -1 retrograde)
  """
  p : Union[float, np.ndarray]
  f : Union[float, np.ndarray]
  g : Union[float, np.ndarray]
  h : Union[float, np.ndarray]
  k : Union[float, np.ndarray]
  L : Union[float, np.ndarray]
  I : int = 1


@dataclass
class GeodeticCoordinates:
  """
  Geodetic (ellipsoidal) coordinates using WGS84.
  
  Attributes:
    latitude  : Geodetic latitude [rad]
    longitude : Longitude [rad]
    altitude  : Height above ellipsoid [m]
  """
  latitude  : Union[float, np.ndarray]
  longitude : Union[float, np.ndarray]
  altitude  : Union[float, np.ndarray]


@dataclass
class GeocentricCoordinates:
  """
  Geocentric (spherical) coordinates.
  
  Attributes:
    latitude  : Geocentric latitude [rad]
    longitude : Longitude [rad]
    altitude  : Altitude above spherical Earth [m]
  """
  latitude  : Union[float, np.ndarray]
  longitude : Union[float, np.ndarray]
  altitude  : Union[float, np.ndarray]


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
