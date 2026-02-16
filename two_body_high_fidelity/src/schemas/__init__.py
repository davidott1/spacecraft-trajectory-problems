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
  TLEData,
)
from src.schemas.propagation import (
  PropagationResult,
  PropagationConfig,
)
from src.schemas.time import (
  TimeStructure,
  TimePoint,
  TimeGrid,
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
from src.schemas.measurement import (
  MeasurementNoise,
  TopocentricState,
  SimulatedMeasurements,
)
from src.schemas.optimization import (
  LunarTransferConfig,
  LunarTransferResult,
  TransferLeg,
)

__all__ = [
  # State representations
  'CartesianState',
  'ClassicalOrbitalElements',
  'ModifiedEquinoctialElements',
  'GeodeticCoordinates',
  'GeocentricCoordinates',
  'TLEData',
  # Propagation
  'PropagationResult',
  'PropagationConfig',
  'TimeStructure',
  'TimePoint',
  'TimeGrid',
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
  # Measurements
  'MeasurementNoise',
  'TopocentricState',
  'SimulatedMeasurements',
  # Optimization
  'LunarTransferConfig',
  'LunarTransferResult',
  'TransferLeg',
]
