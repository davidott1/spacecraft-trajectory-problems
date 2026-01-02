"""
Configuration Schemas
=====================

Dataclasses for simulation configuration and output paths.
"""

from dataclasses import dataclass, field
from datetime    import datetime
from pathlib     import Path
from typing      import Optional, List

from src.schemas.gravity    import GravityModelConfig
from src.schemas.spacecraft import SpacecraftProperties, DragConfig, SRPConfig
from src.schemas.state      import CartesianState


@dataclass
class OutputPaths:
  """
  Output file and directory paths.

  Attributes:
    base_dir     : Base output directory
    figures_dir  : Directory for figure output
    logs_dir     : Directory for log files
    data_dir     : Directory for data output
    log_filepath : Path to log file
  """
  base_dir     : Path
  figures_dir  : Optional[Path] = None
  logs_dir     : Optional[Path] = None
  data_dir     : Optional[Path] = None
  log_filepath : Optional[Path] = None
  def __post_init__(self):
    if self.figures_dir is None:
      self.figures_dir = self.base_dir / 'figures'
    if self.logs_dir is None:
      self.logs_dir = self.base_dir / 'logs'
    if self.data_dir is None:
      self.data_dir = self.base_dir / 'data'
    def ensure_directories(self) -> None:
      """
      Create output directories if they don't exist.
      """
      self.figures_dir.mkdir(parents=True, exist_ok=True)
      self.logs_dir.mkdir(parents=True, exist_ok=True)
      self.data_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class InitialStateConfig:
  """
  Initial state configuration.
  
  Attributes:
    source   : str = 'jpl_horizons'
    norad_id : Optional[str] = None
    filename : Optional[str] = None
    state    : Optional[CartesianState] = None
    epoch_dt : Optional[datetime] = None
  """
  source   : str = 'jpl_horizons'
  norad_id : Optional[str] = None
  filename : Optional[str] = None
  state    : Optional[CartesianState] = None
  epoch_dt : Optional[datetime] = None


@dataclass
class ComparisonConfig:
    """
    Comparison/validation configuration.
    
    Attributes:
        compare_jpl_horizons: Compare with JPL Horizons ephemeris
        compare_tle: Compare with TLE/SGP4 propagation
    """
    compare_jpl_horizons : bool = False
    compare_tle          : bool = False


@dataclass
class SimulationConfig:
    """
    Complete simulation configuration.
    
    Attributes:
        initial_state: Initial state configuration
        time_o_dt: Start time (UTC)
        time_f_dt: End time (UTC)
        spacecraft: Spacecraft properties
        gravity: Gravity model configuration
        comparison: Comparison configuration
        output_paths: Output file paths
        object_name: Sanitized object name for filenames
        object_name_display: Display name for plots
        auto_download: Auto-download missing data
    """
    initial_state       : InitialStateConfig
    time_o_dt           : datetime
    time_f_dt           : datetime
    spacecraft          : SpacecraftProperties = field(default_factory=SpacecraftProperties)
    gravity             : GravityModelConfig = None
    comparison          : ComparisonConfig = field(default_factory=ComparisonConfig)
    output_paths        : Optional[OutputPaths] = None
    object_name         : str = "object"
    object_name_display : str = "Object"
    auto_download       : bool = False
    
    def __post_init__(self):
        if self.gravity is None:
            from src.model.constants import SOLARSYSTEMCONSTANTS
            self.gravity = GravityModelConfig(gp=SOLARSYSTEMCONSTANTS.EARTH.GP)
    
    @property
    def duration_s(self) -> float:
        """Total simulation duration in seconds."""
        return (self.time_f_dt - self.time_o_dt).total_seconds()
    
    @property
    def include_drag(self) -> bool:
        return self.spacecraft.drag.enabled
    
    @property
    def include_srp(self) -> bool:
        return self.spacecraft.srp.enabled
    
    @property
    def include_third_body(self) -> bool:
        return self.gravity.third_body.enabled
    
    @property
    def include_gravity_harmonics(self) -> bool:
        return self.gravity.spherical_harmonics.is_enabled
