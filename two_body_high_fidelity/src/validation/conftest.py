"""
Pytest Configuration and Fixtures
=================================

Shared fixtures for all validation tests.
"""
import pytest
import numpy as np
from pathlib import Path


@pytest.fixture(scope="session")
def project_root():
  """Return the project root directory."""
  return Path(__file__).parent.parent.parent


@pytest.fixture(scope="session")
def spice_kernels_path(project_root):
  """Return path to SPICE kernels."""
  return project_root / "data" / "spice_kernels"


@pytest.fixture(scope="session")
def load_spice_kernels(spice_kernels_path):
  """Load SPICE kernels for tests that need them."""
  import spiceypy as spice
  
  # Load leap seconds kernel
  lsk_path = spice_kernels_path / "naif0012.tls"
  if lsk_path.exists():
    spice.furnsh(str(lsk_path))
  
  # Load planetary ephemeris
  spk_path = spice_kernels_path / "de440.bsp"
  if spk_path.exists():
    spice.furnsh(str(spk_path))
  
  # Load planetary constants
  pck_path = spice_kernels_path / "pck00010.tpc"
  if pck_path.exists():
    spice.furnsh(str(pck_path))
  
  yield  # Tests run here
  
  # Cleanup
  spice.kclear()


@pytest.fixture
def leo_initial_state():
  """Typical LEO initial state for testing."""
  return np.array([
    7000.0e3,    # x [m]
    0.0,         # y [m]
    0.0,         # z [m]
    0.0,         # vx [m/s]
    7.5e3,       # vy [m/s]
    0.0,         # vz [m/s]
  ])


@pytest.fixture
def meo_initial_state():
  """Typical MEO initial state (GPS-like) for testing."""
  return np.array([
    26560.0e3,   # x [m]
    0.0,         # y [m]
    0.0,         # z [m]
    0.0,         # vx [m/s]
    3.87e3,      # vy [m/s]
    0.0,         # vz [m/s]
  ])


@pytest.fixture
def geo_initial_state():
  """Typical GEO initial state for testing."""
  return np.array([
    42164.0e3,   # x [m]
    0.0,         # y [m]
    0.0,         # z [m]
    0.0,         # vx [m/s]
    3.075e3,     # vy [m/s]
    0.0,         # vz [m/s]
  ])


@pytest.fixture(scope="session", autouse=True)
def test_data_paths(monkeypatch_session):
  """
  Override data paths to use test fixtures instead of user data.
  This fixture runs automatically for all tests.
  """
  # Test fixtures directory (adjacent to this conftest.py)
  fixtures_path = Path(__file__).parent / "fixtures"
  
  # Only override if fixtures exist (allows graceful fallback)
  if fixtures_path.exists():
    # Monkeypatch the setup_paths_and_files function or
    # set environment variables that configuration.py reads
    import os
    os.environ['ORBIT_PROPAGATOR_TEST_DATA'] = str(fixtures_path)


@pytest.fixture(scope="session")
def monkeypatch_session():
  """Session-scoped monkeypatch fixture."""
  from _pytest.monkeypatch import MonkeyPatch
  mp = MonkeyPatch()
  yield mp
  mp.undo()


@pytest.fixture(scope="session")
def fixtures_path():
  """Return path to test fixtures directory."""
  return Path(__file__).parent / "fixtures"
