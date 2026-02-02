"""
Gravity Model Schemas
=====================

Dataclasses for gravity model configuration.
"""

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
  from src.model.gravity_field import SphericalHarmonicsGravity


@dataclass
class SphericalHarmonicsConfig:
  """
  Configuration for spherical harmonics gravity model.
  
  Attributes:
    degree       : Maximum degree of expansion
    order        : Maximum order of expansion
    gp           : Gravitational parameter [m³/s²]
    radius       : Reference radius [m]
    coefficients : List of coefficient names to include (e.g., ['J2', 'J3', 'C22'])
    model        : Loaded spherical harmonics model instance
  """
  degree       : int                                   = 0
  order        : int                                   = 0
  gp           : Optional[float]                       = None
  radius       : Optional[float]                       = None
  coefficients : List[str]                             = field(default_factory=list)
  model        : Optional['SphericalHarmonicsGravity'] = None
  
  @property
  def enabled(self) -> bool:
    """
    Check if spherical harmonics are enabled.
    """
    return self.degree > 0 or len(self.coefficients) > 0


@dataclass
class ThirdBodyConfig:
  """
  Configuration for third-body gravitational perturbations.

  Attributes:
    enabled : Whether third-body gravity is enabled
    bodies  : List of body names (e.g., ['SUN', 'MOON', 'JUPITER'])
  """
  enabled : bool      = False
  bodies  : List[str] = field(default_factory=list)

  def __post_init__(self):
    self.bodies = [b.upper() for b in self.bodies]


@dataclass
class RelativityConfig:
  """
  Configuration for general relativistic corrections.

  Attributes:
    enabled             : Whether relativistic corrections are enabled
    include_schwarzschild : Include Schwarzschild (point-mass) correction
    include_lense_thirring : Include Lense-Thirring (frame-dragging) correction (future)
  """
  enabled               : bool = False
  include_schwarzschild : bool = True
  include_lense_thirring : bool = False


@dataclass
class SolidEarthTidesConfig:
  """
  Configuration for solid Earth tide corrections.

  Attributes:
    enabled : Whether solid Earth tides are enabled
  """
  enabled : bool = False


@dataclass
class OceanTidesConfig:
  """
  Configuration for ocean tide corrections.

  Attributes:
    enabled : Whether ocean tides are enabled
  """
  enabled : bool = False


@dataclass
class GravityModelConfig:
  """
  Complete gravity model configuration.

  Attributes:
    gp                    : Central body gravitational parameter [m³/s²]
    central_body          : Central body name (default 'EARTH')
    spherical_harmonics   : Spherical harmonics configuration
    third_body            : Third-body configuration
    relativity            : General relativity configuration
    solid_tides           : Solid Earth tides configuration
    ocean_tides           : Ocean tides configuration
    folderpath            : Path to gravity model files
    filename              : Gravity model filename (e.g., 'EGM2008.gfc')
    use_approx_jacobian   : Use numerical Jacobian for gravity (default handled in config)
    use_analytic_jacobian : Use analytic Jacobian (currently J2-only)
    jacobian_approx_eps   : Relative step size for numerical Jacobian
  """
  gp                    : float
  central_body          : str                      = "EARTH"
  spherical_harmonics   : SphericalHarmonicsConfig = field(default_factory=SphericalHarmonicsConfig)
  third_body            : ThirdBodyConfig          = field(default_factory=ThirdBodyConfig)
  relativity            : RelativityConfig         = field(default_factory=RelativityConfig)
  solid_tides           : SolidEarthTidesConfig    = field(default_factory=SolidEarthTidesConfig)
  ocean_tides           : OceanTidesConfig         = field(default_factory=OceanTidesConfig)
  folderpath            : Optional[Path]           = None
  filename              : Optional[str]            = None
  use_approx_jacobian   : Optional[bool]           = None
  use_analytic_jacobian : Optional[bool]           = None
  jacobian_approx_eps   : Optional[float]          = None
  
  @property
  def filepath(self) -> Optional[Path]:
    """
    Full path to gravity model file.
    """
    if self.folderpath and self.filename:
      return self.folderpath / self.filename
    return None
