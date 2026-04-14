"""
Integration Tests for src.main Propagation Pipeline
=====================================================

End-to-end tests that invoke main() directly — equivalent to running the
CLI command. These verify the full pipeline: SPICE loading, JPL Horizons
initial state, numerical integration, and output structure.

Uses the pre-downloaded ISS ephemeris for 2025-10-01 so no network access
is required.

Tests:
------
TestMainPropagation
  - test_keplerian_propagation   : point-mass only, verifies SMA conservation
  - test_high_fidelity_propagation : J2/J3/J4 + third-body + drag + SRP,
                                     verifies SMA changes relative to Keplerian

Usage:
------
  python -m pytest src/verification/integration/test_main_propagation.py -v

Equivalent CLI commands:
  # Keplerian
  python -m src.main \\
    --initial-state-source jpl_horizons \\
    --initial-state-norad-id 25544 \\
    --timespan 2025-10-01T00:00:00 2025-10-02T00:00:00

  # High-fidelity
  python -m src.main \\
    --initial-state-source jpl_horizons \\
    --initial-state-norad-id 25544 \\
    --timespan 2025-10-01T00:00:00 2025-10-02T00:00:00 \\
    --gravity-harmonics-coefficients J2 J3 J4 \\
    --third-bodies sun moon \\
    --include-srp \\
    --include-drag
"""
import pytest
import numpy as np

from datetime import datetime

from src.main                  import main
from src.model.orbit_converter import OrbitConverter
from src.model.constants       import SOLARSYSTEMCONSTANTS


NORAD_ID  = '25544'
TIMESPAN  = [datetime(2025, 10, 1), datetime(2025, 10, 2)]


class TestMainPropagation:
  """
  End-to-end propagation tests via main().
  Requires SPICE kernels and pre-downloaded ISS ephemeris.
  """

  @pytest.fixture(autouse=True)
  def _load_spice(self, load_spice_kernels):
    pass

  def test_keplerian_propagation(self):
    """
    Point-mass only propagation of ISS over 1 day.

    Verifies:
      - Pipeline completes successfully
      - Output state array has correct shape
      - SMA is conserved to within 0.1% (no perturbations)
    """
    result = main(
      initial_state_norad_id = NORAD_ID,
      initial_state_filename = None,
      timespan               = TIMESPAN,
      initial_state_source   = 'jpl_horizons',
      auto_download          = False,
    )

    assert result.success, f"Propagation failed: {result.message}"
    assert result.state is not None
    assert result.state.shape[0] == 6
    assert result.state.shape[1] > 1

    gp      = SOLARSYSTEMCONSTANTS.EARTH.GP
    state_o = result.state[:, 0]
    state_f = result.state[:, -1]
    sma_o   = OrbitConverter.pv_to_coe(state_o[0:3], state_o[3:6], gp).sma
    sma_f   = OrbitConverter.pv_to_coe(state_f[0:3], state_f[3:6], gp).sma

    assert np.isclose(sma_o, sma_f, rtol=1e-3), (
      f"SMA changed in Keplerian propagation: "
      f"{sma_o/1e3:.3f} km → {sma_f/1e3:.3f} km"
    )


  def test_high_fidelity_propagation(self):
    """
    High-fidelity propagation of ISS over 1 day with J2/J3/J4,
    third-body (Sun + Moon), atmospheric drag, and SRP.

    Verifies:
      - Pipeline completes successfully
      - Output state array has correct shape
      - SMA drifts relative to Keplerian (perturbations are active)
      - Final altitude is physically reasonable for LEO (200–450 km)
    """
    result = main(
      initial_state_norad_id = NORAD_ID,
      initial_state_filename = None,
      timespan               = TIMESPAN,
      initial_state_source   = 'jpl_horizons',
      gravity_harmonics      = ['J2', 'J3', 'J4'],
      third_bodies           = ['SUN', 'MOON'],
      include_drag           = True,
      include_srp            = True,
      auto_download          = False,
    )

    assert result.success, f"Propagation failed: {result.message}"
    assert result.state is not None
    assert result.state.shape[0] == 6
    assert result.state.shape[1] > 1

    gp      = SOLARSYSTEMCONSTANTS.EARTH.GP
    state_o = result.state[:, 0]
    state_f = result.state[:, -1]
    sma_o   = OrbitConverter.pv_to_coe(state_o[0:3], state_o[3:6], gp).sma
    sma_f   = OrbitConverter.pv_to_coe(state_f[0:3], state_f[3:6], gp).sma

    # Perturbations must cause measurable SMA change
    assert not np.isclose(sma_o, sma_f, rtol=1e-6), (
      "SMA did not change under high-fidelity perturbations — "
      "check force models are active"
    )

    # Final altitude must be physically reasonable for ISS
    earth_radius   = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR
    altitude_f__m  = np.linalg.norm(state_f[0:3]) - earth_radius
    altitude_f__km = altitude_f__m / 1000.0
    assert 200.0 < altitude_f__km < 450.0, (
      f"Final altitude = {altitude_f__km:.1f} km, expected 200–450 km for ISS"
    )
