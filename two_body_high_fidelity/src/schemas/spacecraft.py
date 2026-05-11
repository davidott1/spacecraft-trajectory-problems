"""
Spacecraft Property Schemas
===========================

Dataclasses for spacecraft physical properties and force model configuration.
"""

import numpy as np

from dataclasses import dataclass
from datetime    import datetime
from typing      import Optional, List


@dataclass
class DragConfig:
  """
  Atmospheric drag configuration.
  
  Attributes:
    enabled : Whether drag is enabled
    cd      : Drag coefficient [-]
    area    : Cross-sectional area [m²]
  """
  enabled : bool  = False
  cd      : float = 2.2
  area    : float = 0.0
  
  @property
  def is_valid(self) -> bool:
    """
    Check if drag parameters are valid.
    """
    return self.enabled and self.cd > 0 and self.area > 0


@dataclass
class SRPConfig:
  """
  Solar radiation pressure configuration.
  
  Attributes:
    enabled : Whether SRP is enabled
    cr      : Reflectivity coefficient [-] (1.0=absorbing, 2.0=reflecting)
    area    : Cross-sectional area [m²]
  """
  enabled : bool  = False
  cr      : float = 1.3
  area    : float = 0.0
  
  @property
  def is_valid(self) -> bool:
    """
    Check if SRP parameters are valid.
    """
    return self.enabled and self.cr > 0 and self.area > 0


@dataclass
class ManeuversConfig:
  """
  Maneuvers configuration with list-like behavior.

  This class acts like a list of maneuvers but also stores metadata like the filename.
  You can iterate over it, index it, and check its length just like a list.

  Attributes:
    filename : Maneuver configuration filename (for reference)
    _items   : Internal list of impulsive maneuvers

  Example:
    maneuvers = ManeuversConfig(filename='burns.yaml')
    maneuvers.append(ImpulsiveManeuver(...))

    # Use like a list
    for maneuver in maneuvers:
      print(f"ΔV = {maneuver.mag():.1f} m/s")

    # Access metadata
    print(maneuvers.filename)
  """
  filename : Optional[str]                       = None
  _items   : Optional[List['ImpulsiveManeuver']] = None

  def __post_init__(self):
    if self._items is None:
      self._items = []
    else:
      self._items.sort(key=lambda m: m.time_dt)

  def __len__(self) -> int:
    """Return number of maneuvers."""
    return len(self._items)

  def __iter__(self):
    """Make iterable: for m in maneuvers."""
    return iter(self._items)

  def __getitem__(self, index):
    """Support indexing: maneuvers[0]."""
    return self._items[index]

  def __bool__(self) -> bool:
    """Return True if maneuvers exist."""
    return len(self._items) > 0

  def append(self, maneuver: 'ImpulsiveManeuver'):
    """Add a maneuver to the list, maintaining time order."""
    self._items.append(maneuver)
    self._items.sort(key=lambda m: m.time_dt)

  @property
  def is_valid(self) -> bool:
    """Check if maneuvers configuration is valid."""
    return len(self._items) > 0


@dataclass
class ImpulsiveManeuver:
  """
  Impulsive Delta-V maneuver specification.

  An impulsive maneuver represents an instantaneous change in velocity
  at a specific time. The Delta-V can be specified in different reference
  frames for convenience.

  Attributes:
    time_dt      : Maneuver execution time (UTC datetime)
    delta_vel_vec: Delta-V vector [m/s] (3-element array)
    frame        : Reference frame for delta_vel_vec ('J2000', 'RIC', 'RTN')

  Reference Frames:
    - J2000: Inertial J2000 frame (default for propagation)
    - RIC: Radial-In-track-Cross-track (orbit-aligned)
    - RTN: Radial-Tangential-Normal (orbit-aligned)

  Example:
    # Prograde burn of 10 m/s in RIC frame
    maneuver = ImpulsiveManeuver(
      time_dt       = datetime(2025, 10, 1, 12, 0, 0),
      delta_vel_vec = np.array([0.0, 10.0, 0.0]),
      frame         = 'RIC'
    )
  """
  time_dt       : datetime
  delta_vel_vec : np.ndarray  # [3] in m/s
  frame         : str = 'J2000'  # 'J2000', 'RIC', 'RTN'
  name          : Optional[str] = None
  description   : Optional[str] = None

  def __post_init__(self):
    """Validate maneuver parameters."""
    if len(self.delta_vel_vec) != 3:
      raise ValueError(f"delta_vel_vec must be 3-element array, got {len(self.delta_vel_vec)}")

    valid_frames = ['J2000', 'RIC', 'RTN']
    if self.frame not in valid_frames:
      raise ValueError(f"frame must be one of {valid_frames}, got '{self.frame}'")

    # Ensure delta_vel_vec is numpy array
    if not isinstance(self.delta_vel_vec, np.ndarray):
      self.delta_vel_vec = np.array(self.delta_vel_vec)

  def mag(self) -> float:
    """
    Compute Delta-V magnitude.

    Output:
    -------
      magnitude : float
        Delta-V magnitude [m/s]
    """
    return float(np.linalg.norm(self.delta_vel_vec))


@dataclass
class SpacecraftProperties:
  """
  Complete spacecraft physical properties.

  Attributes:
    mass      : Spacecraft mass [kg]
    drag      : Drag configuration
    srp       : Solar radiation pressure configuration
    maneuvers : Maneuvers configuration (acts like a list with metadata)
    norad_id  : NORAD catalog ID (if applicable)
    name      : Spacecraft name
  """
  mass      : float                   = 1000.0
  drag      : Optional[DragConfig]    = None
  srp       : Optional[SRPConfig]     = None
  maneuvers : Optional[ManeuversConfig] = None
  norad_id  : Optional[str]           = None
  name      : Optional[str]           = None

  def __post_init__(self):
    if self.drag is None:
      self.drag = DragConfig()
    if self.srp is None:
      self.srp = SRPConfig()
    if self.maneuvers is None:
      self.maneuvers = ManeuversConfig()
