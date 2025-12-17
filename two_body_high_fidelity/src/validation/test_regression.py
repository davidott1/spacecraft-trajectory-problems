"""
Regression Tests for Orbit Propagator
=====================================

End-to-end regression tests that run example propagations and verify they complete successfully.

Tests:
------
TestRegressionEndToEnd
  - test_regression_two_body_propagation_completes         : verify basic two-body propagation completes successfully
  - test_regression_propagation_with_j2_completes          : verify propagation with J2 perturbation completes successfully
  - test_regression_propagation_with_full_forces_completes : verify propagation with all force models enabled completes
  - test_regression_full_forces_with_comparisons           : verify GPS satellite with all forces and comparisons

TestCLIIntegration
  - test_regression_cli_basic_propagation : verify CLI invocation works for a basic propagation

Usage:
------
  python -m pytest src/validation/test_regression.py -v

Note:
-----
  These tests require SPICE kernels and JPL Horizons ephemeris data to be available.
"""
import subprocess
import sys

from datetime import datetime

from src.main import main


class TestRegressionEndToEnd:
  """
  End-to-end regression tests.
  """
  
  def test_regression_two_body_propagation_completes(self):
    """
    Test that a basic two-body propagation completes successfully.

    CLI Equivalent:
    ---------------
    python -m src.main \
      --initial-state-norad-id 25544 \
      --timespan 2025-10-01T00:00:00 2025-10-01T01:00:00
    """
    result = main(
      initial_state_norad_id = '25544',  # ISS
      initial_state_filename = None,
      timespan               = [datetime(2025, 10, 1, 0, 0, 0), datetime(2025, 10, 1, 1, 0, 0)],  # 1 hour
      include_drag           = False,
      compare_tle            = False,
      compare_jpl_horizons   = False,
      third_bodies           = None,
      gravity_harmonics      = None,
      include_srp            = False,
      initial_state_source   = 'jpl_horizons',
    )
    
    assert result.get('success', False), f"Propagation failed: {result.get('message', 'Unknown error')}"
  
  def test_regression_propagation_with_j2_completes(self):
    """
    Test that propagation with J2 perturbation completes successfully.

    CLI Equivalent:
    ---------------
    python -m src.main \
      --initial-state-norad-id 25544 \
      --timespan 2025-10-01T00:00:00 2025-10-01T01:00:00 \
      --gravity-harmonics J2
    """
    result = main(
      initial_state_norad_id = '25544',
      initial_state_filename = None,
      timespan               = [datetime(2025, 10, 1, 0, 0, 0), datetime(2025, 10, 1, 1, 0, 0)],
      include_drag           = False,
      compare_tle            = False,
      compare_jpl_horizons   = False,
      third_bodies           = None,
      gravity_harmonics      = ['J2'],
      include_srp            = False,
      initial_state_source   = 'jpl_horizons',
    )
    
    assert result.get('success', False)
  
  def test_regression_propagation_with_full_forces_completes(self):
    """
    Test propagation with all force models enabled.

    CLI Equivalent:
    ---------------
    python -m src.main \
      --initial-state-norad-id 25544 \
      --timespan 2025-10-01T00:00:00 2025-10-01T01:00:00 \
      --gravity-harmonics J2 J3 J4 \
      --third-bodies SUN MOON \
      --drag \
      --srp
    """
    result = main(
      initial_state_norad_id = '25544',
      initial_state_filename = None,
      timespan               = [datetime(2025, 10, 1, 0, 0, 0), datetime(2025, 10, 1, 1, 0, 0)],
      include_drag           = True,
      compare_tle            = False,
      compare_jpl_horizons   = False,
      third_bodies           = ['SUN', 'MOON'],
      gravity_harmonics      = ['J2', 'J3', 'J4'],
      include_srp            = True,
      initial_state_source   = 'jpl_horizons',
    )
    
    assert result.get('success', False)

  def test_regression_full_forces_with_comparisons(self):
    """
    Test GPS satellite (39166) with all force models and comparisons enabled.
    
    This is a comprehensive regression test that exercises:
    - All gravity harmonics (J2, J3, J4, C22, S22)
    - All third-body perturbations (Sun, Moon, planets)
    - Atmospheric drag
    - Solar radiation pressure
    - JPL Horizons comparison
    - TLE/SGP4 comparison

    CLI Equivalent:
    ---------------
    python -m src.main \
      --initial-state-norad-id 39166 \
      --timespan 2025-10-01T00:00:00 2025-10-02T00:00:00 \
      --gravity-harmonics J2 J3 J4 C22 S22 \
      --third-bodies SUN MOON MERCURY VENUS MARS JUPITER SATURN URANUS NEPTUNE PLUTO \
      --drag \
      --srp \
      --compare-tle \
      --compare-jpl-horizons
    """
    result = main(
      initial_state_norad_id = '39166',  # GPS IIF-3
      initial_state_filename = None,
      timespan               = [datetime(2025, 10, 1, 0, 0, 0), datetime(2025, 10, 2, 0, 0, 0)],  # 1 day
      include_drag           = True,
      compare_tle            = True,
      compare_jpl_horizons   = True,
      third_bodies           = ['SUN', 'MOON', 'MERCURY', 'VENUS', 'MARS', 'JUPITER', 'SATURN', 'URANUS', 'NEPTUNE', 'PLUTO'],
      gravity_harmonics      = ['J2', 'J3', 'J4', 'C22', 'S22'],
      include_srp            = True,
      initial_state_source   = 'jpl_horizons',
    )
    
    assert result.get('success', False), f"Propagation failed: {result.get('message', 'Unknown error')}"


class TestCLIIntegration:
  """
  Integration tests that verify CLI argument parsing works correctly.
  """
  
  def test_regression_cli_basic_propagation(self):
    """
    Test that CLI invocation works for a basic propagation.

    CLI Equivalent:
    ---------------
    python -m src.main \
      --initial-state-norad-id 25544 \
      --timespan 2025-10-01T00:00:00 2025-10-01T00:30:00
    """
    result = subprocess.run(
      [
        sys.executable, '-m', 'src.main',
        '--initial-state-norad-id', '25544',
        '--timespan', '2025-10-01T00:00:00', '2025-10-01T00:30:00',
      ],
      capture_output = True,
      text           = True,
      cwd            = str(__file__).rsplit('/src/', 1)[0],  # Project root
    )
    
    # Check process completed (exit code 0 means success)
    assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"
    
    # Verify expected output in stdout
    assert "High-Fidelity Model" in result.stdout
    assert "Complete" in result.stdout
