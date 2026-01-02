"""
Spacecraft Property Schemas
===========================

Dataclasses for spacecraft physical properties and force model configuration.
"""

from dataclasses import dataclass
from typing      import Optional


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
    """Check if drag parameters are valid."""
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
    """Check if SRP parameters are valid."""
    return self.enabled and self.cr > 0 and self.area > 0


@dataclass
class SpacecraftProperties:
  """
  Complete spacecraft physical properties.
  
  Attributes:
    mass     : Spacecraft mass [kg]
    drag     : Drag configuration
    srp      : Solar radiation pressure configuration
    norad_id : NORAD catalog ID (if applicable)
    name     : Spacecraft name
  """
  mass     : float         = 1000.0
  drag     : DragConfig    = None
  srp      : SRPConfig     = None
  norad_id : Optional[str] = None
  name     : Optional[str] = None
  def __post_init__(self):
    if self.drag is None:
      self.drag = DragConfig()
    if self.srp is None:
      self.srp = SRPConfig()
