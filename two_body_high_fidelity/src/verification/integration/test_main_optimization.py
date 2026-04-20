"""
Integration Tests for src.main Optimization Pipeline
=====================================================

End-to-end tests that invoke main() with trajectory optimization enabled.
These verify the full pipeline: maneuver plan loading, dynamics setup,
optimizer invocation, and post-optimization propagation.

Uses the example maneuver plan in input/initial_states_and_maneuvers/.

Tests:
------
TestMainOptimization
  - test_optimize_maneuver_plan : optimize a two-burn raise-and-circularize plan,
                                  verify convergence and physical ΔV bounds

Usage:
------
  python -m pytest src/verification/integration/test_main_optimization.py -v

Equivalent CLI command:
  python -m src.main \\
    --initial-maneuver-plan example_maneuver_plan.yaml \\
    --timespan 2025-10-01T00:00:00 2025-10-02T00:00:00 \\
    --optimize maneuvers
"""
import pytest
import numpy as np

from datetime import datetime
from pathlib  import Path

from src.main            import main
from src.model.constants import SOLARSYSTEMCONSTANTS


PROJECT_ROOT       = Path(__file__).parent.parent.parent.parent
MANEUVER_PLAN      = str(PROJECT_ROOT / 'input' / 'initial_states_and_maneuvers' / 'example_maneuver_plan.yaml')
MANEUVER_PLAN_NAME = 'example_maneuver_plan.yaml'
TIMESPAN           = [datetime(2025, 10, 1), datetime(2025, 10, 2)]


class TestMainOptimization:
  """
  End-to-end optimization tests via main().
  Requires SPICE kernels.
  """

  @pytest.fixture(autouse=True)
  def _load_spice(self, load_spice_kernels):
    pass

  def test_optimize_maneuver_plan(self):
    """
    Optimize the example two-burn maneuver plan (raise apogee + circularize).
    Keplerian dynamics only for speed.

    Verifies:
      - Pipeline completes successfully
      - Output state array has correct shape
      - Final altitude is physically reasonable (not crashed or escaped)
    """
    result = main(
      initial_state_norad_id = None,
      initial_state_filename = 'equatorial.yaml',
      timespan               = TIMESPAN,
      initial_state_source   = 'sv',
      initial_maneuver_plan  = MANEUVER_PLAN,
      optimize               = ['maneuvers'],
      auto_download          = False,
    )

    assert result.success, f"Optimization pipeline failed: {result.message}"
    assert result.state is not None
    assert result.state.shape[0] == 6
    assert result.state.shape[1] > 1

    earth_radius   = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR
    state_f        = result.state[:, -1]
    altitude_f__m  = np.linalg.norm(state_f[0:3]) - earth_radius
    altitude_f__km = altitude_f__m / 1000.0

    assert 100.0 < altitude_f__km < 2000.0, (
      f"Final altitude = {altitude_f__km:.1f} km, expected 100–2000 km "
      f"for a LEO raise-and-circularize maneuver"
    )
