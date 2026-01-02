"""
Data Schemas
============

Dataclass definitions for inputs, outputs, and intermediate data structures.
Using dataclasses for type safety, validation, and self-documentation.
"""

from src.schemas.state import (
  CartesianState,
  ClassicalOrbitalElements,
  ModifiedEquinoctialElements,
  GeodeticCoordinates,
  GeocentricCoordinates,
)
from src.schemas.propagation import (
  PropagationResult,
  PropagationConfig,
  TimeGrid,
)
from src.schemas.ephemeris import (
  EphemerisResult,
  TLEData,
  HorizonsEphemeris,
)
from src.schemas.gravity import (
  GravityModelConfig,
  SphericalHarmonicsConfig,
  ThirdBodyConfig,
)
from src.schemas.spacecraft import (
  SpacecraftProperties,
  DragConfig,
  SRPConfig,
)
from src.schemas.config import (
  SimulationConfig,
  OutputPaths,
)

__all__ = [
  # State representations
  'CartesianState',
  'ClassicalOrbitalElements',
  'ModifiedEquinoctialElements',
  'GeodeticCoordinates',
  'GeocentricCoordinates',
  # Propagation
  'PropagationResult',
  'PropagationConfig',
  'TimeGrid',
  # Ephemeris
  'EphemerisResult',
  'TLEData',
  'HorizonsEphemeris',
  # Gravity
  'GravityModelConfig',
  'SphericalHarmonicsConfig',
  'ThirdBodyConfig',
  # Spacecraft
  'SpacecraftProperties',
  'DragConfig',
  'SRPConfig',
  # Configuration
  'SimulationConfig',
  'OutputPaths',
]
