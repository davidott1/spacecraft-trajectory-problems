"""
Integration Tests for src.main Orbit Determination Pipeline
=============================================================

End-to-end tests that invoke main() directly with EKF orbit determination
enabled — equivalent to running the CLI command. These verify the full
pipeline: SPICE loading, JPL Horizons initial state, numerical integration,
measurement simulation, EKF filtering, and output structure.

Uses the pre-downloaded ISS ephemeris for 2025-10-01 and the ISS tracker
station configuration so no network access is required.

Tests:
------
TestMainOrbitDetermination
  - test_od_keplerian_jpl_truth   : Keplerian propagation + EKF with JPL Horizons truth;
                                    verifies EKF state shape and ISS altitude bounds
  - test_od_high_fidelity_model_truth : J2/J3/J4 + third-body propagation + EKF with
                                        closed-loop model truth; verifies same bounds

Usage:
------
  python -m pytest src/verification/integration/test_main_od.py -v

Equivalent CLI commands:
  # Keplerian + EKF (JPL Horizons truth)
  python -m src.main \\
    --initial-state-source jpl_horizons \\
    --initial-state-norad-id 25544 \\
    --timespan 2025-10-01T00:00:00 2025-10-01T00:10:00 \\
    --include-tracker-skyplots \\
    --tracker-filepath input/trackers/trackers_iss.yaml \\
    --include-orbit-determination \\
    --make-meas-from jpl_horizons

  # High-fidelity + EKF (closed-loop model truth)
  python -m src.main \\
    --initial-state-source jpl_horizons \\
    --initial-state-norad-id 25544 \\
    --timespan 2025-10-01T00:00:00 2025-10-01T00:10:00 \\
    --gravity-harmonics-coefficients J2 J3 J4 \\
    --third-bodies sun moon \\
    --include-tracker-skyplots \\
    --tracker-filepath input/trackers/trackers_iss.yaml \\
    --include-orbit-determination \\
    --make-meas-from model
"""
import pytest
import numpy as np

from datetime import datetime
from pathlib  import Path

from src.main            import main
from src.model.constants import SOLARSYSTEMCONSTANTS


NORAD_ID         = '25544'
TIMESPAN         = [datetime(2025, 10, 1), datetime(2025, 10, 1, 0, 10)]
PROJECT_ROOT     = Path(__file__).parent.parent.parent.parent
TRACKER_FILEPATH = str(PROJECT_ROOT / 'input' / 'trackers' / 'trackers_iss.yaml')


class TestMainOrbitDetermination:
  """
  End-to-end EKF orbit determination tests via main().
  Requires SPICE kernels and pre-downloaded ISS ephemeris.
  """

  @pytest.fixture(autouse=True)
  def _load_spice(self, load_spice_kernels):
    pass

  def test_od_keplerian_jpl_truth(self):
    """
    Keplerian propagation of ISS over 6 hours with EKF using
    JPL Horizons ephemeris as measurement truth.

    Verifies:
      - Pipeline completes successfully
      - Output state array has correct shape (EKF filter estimates)
      - Final altitude is physically reasonable for ISS (200–450 km)
    """
    result = main(
      initial_state_norad_id      = NORAD_ID,
      initial_state_filename      = None,
      timespan                    = TIMESPAN,
      initial_state_source        = 'jpl_horizons',
      include_tracker_skyplots    = True,
      tracker_filepath            = TRACKER_FILEPATH,
      include_orbit_determination = True,
      make_meas_from              = 'jpl_horizons',
      auto_download               = False,
    )

    assert result.success, f"OD pipeline failed: {result.message}"
    assert result.state is not None
    assert result.state.shape[0] == 6
    assert result.state.shape[1] > 1

    earth_radius   = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR
    state_f        = result.state[:, -1]
    altitude_f__m  = np.linalg.norm(state_f[0:3]) - earth_radius
    altitude_f__km = altitude_f__m / 1000.0
    assert 200.0 < altitude_f__km < 450.0, (
      f"Final altitude = {altitude_f__km:.1f} km, expected 200–450 km for ISS"
    )

  def test_od_high_fidelity_model_truth(self):
    """
    High-fidelity propagation of ISS over 6 hours with J2/J3/J4 and
    third-body perturbations, EKF using the closed-loop model as truth.

    Verifies:
      - Pipeline completes successfully
      - Output state array has correct shape (EKF filter estimates)
      - Final altitude is physically reasonable for ISS (200–450 km)
    """
    result = main(
      initial_state_norad_id      = NORAD_ID,
      initial_state_filename      = None,
      timespan                    = TIMESPAN,
      initial_state_source        = 'jpl_horizons',
      gravity_harmonics           = ['J2', 'J3', 'J4'],
      third_bodies                = ['SUN', 'MOON'],
      include_tracker_skyplots    = True,
      tracker_filepath            = TRACKER_FILEPATH,
      include_orbit_determination = True,
      make_meas_from              = 'model',
      auto_download               = False,
    )

    assert result.success, f"OD pipeline failed: {result.message}"
    assert result.state is not None
    assert result.state.shape[0] == 6
    assert result.state.shape[1] > 1

    earth_radius   = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR
    state_f        = result.state[:, -1]
    altitude_f__m  = np.linalg.norm(state_f[0:3]) - earth_radius
    altitude_f__km = altitude_f__m / 1000.0
    assert 200.0 < altitude_f__km < 450.0, (
      f"Final altitude = {altitude_f__km:.1f} km, expected 200–450 km for ISS"
    )
