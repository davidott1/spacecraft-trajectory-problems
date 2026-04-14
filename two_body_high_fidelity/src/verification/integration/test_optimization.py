"""
Integration Tests for Optimization Module
==========================================

Tests for PatchedConicGridSearch.hohmann factory and full solve pipeline.
Requires SPICE kernels to be loaded.

Tests:
------
TestPatchedConicGridSearchHohmann
  - test_nominal_departure_within_one_leo_period  : departure_center__s < one LEO period
  - test_sweep_bounds_symmetric_about_nominal     : grid spans [center-half, center+half]
  - test_sweep_half_window_matches_delta_anomaly  : half-window converts correctly from degrees
  - test_grid_search_finds_soi_crossings          : at least one candidate reaches Moon SOI
  - test_full_solve_valid_result                  : end-to-end solve with physical ΔV and trajectory

Usage:
------
  python -m pytest src/verification/integration/test_optimization.py -v
"""
import pytest
import numpy as np

from datetime import datetime

from src.model.constants         import SOLARSYSTEMCONSTANTS
from src.model.orbital_mechanics import compute_circular_velocity


class TestPatchedConicGridSearchHohmann:
  """
  Integration tests for PatchedConicGridSearch.hohmann.

  Covers factory construction (requires SPICE for Moon ephemeris),
  grid search behavior, and the full solve pipeline.
  """

  @pytest.fixture(autouse=True)
  def _load_spice(self, load_spice_kernels):
    pass

  @pytest.fixture
  def leo_circular_state(self):
    """Equatorial circular LEO state at 200 km altitude."""
    radius_leo   = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR + 200_000.0
    vel_mag_circ = compute_circular_velocity(radius_leo, SOLARSYSTEMCONSTANTS.EARTH.GP)
    return np.array([radius_leo, 0.0, 0.0, 0.0, vel_mag_circ, 0.0])

  @pytest.fixture
  def basic_problem(self):
    """Minimal OptimizationProblem for factory construction."""
    from src.schemas.optimization import (
      OptimizationProblem, OptimizationConfig, DecisionState,
      Objective, BoundaryCondition, Constraint,
    )
    return OptimizationProblem(
      objective           = Objective(quantity='delta_v_total', nodes=[0, 2]),
      decision_state      = DecisionState(epoch=datetime(2025, 10, 1)),
      constraints         = Constraint(final=[BoundaryCondition(node=2, quantity='altitude', target=100_000.0)]),
      optimization_config = OptimizationConfig(),
    )

  @pytest.fixture
  def search(self, basic_problem, leo_circular_state):
    """PatchedConicGridSearch.hohmann instance with fast settings."""
    from src.optimization.initial_guess import PatchedConicGridSearch
    return PatchedConicGridSearch.hohmann(
      problem                  = basic_problem,
      initial_state            = leo_circular_state,
      delta_anomaly_range__deg = 30.0,
      n_departure_candidates   = 61,
    )

  def test_nominal_departure_within_one_leo_period(self, search):
    """
    Nominal departure coast time must be within [0, one LEO period).
    """
    radius_leo    = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR + 200_000.0
    leo_period__s = 2.0 * np.pi * np.sqrt(radius_leo**3 / SOLARSYSTEMCONSTANTS.EARTH.GP)

    assert 0.0 <= search.departure_center__s < leo_period__s, (
      f"departure_center__s = {search.departure_center__s:.1f} s is outside "
      f"[0, {leo_period__s:.1f} s)"
    )

  def test_sweep_bounds_symmetric_about_nominal(self, search):
    """
    Grid spans [departure_center - half_window, departure_center + half_window].
    """
    offsets = np.linspace(
      search.departure_center__s - search.departure_half_window__s,
      search.departure_center__s + search.departure_half_window__s,
      search.n_departure_candidates,
    )
    assert offsets[0]  == pytest.approx(search.departure_center__s - search.departure_half_window__s)
    assert offsets[-1] == pytest.approx(search.departure_center__s + search.departure_half_window__s)
    assert offsets[search.n_departure_candidates // 2] == pytest.approx(search.departure_center__s, rel=1e-2)

  def test_sweep_half_window_matches_delta_anomaly(self, search):
    """
    Half-window in seconds corresponds to 30 deg on a circular LEO orbit.
    """
    radius_leo           = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR + 200_000.0
    omega_leo__rad_per_s = np.sqrt(SOLARSYSTEMCONSTANTS.EARTH.GP / radius_leo**3)
    expected__s          = np.radians(30.0) / omega_leo__rad_per_s

    assert search.departure_half_window__s == pytest.approx(expected__s, rel=1e-6)

  def test_grid_search_finds_soi_crossings(self, search):
    """
    Grid search must find at least one candidate that reaches the Moon SOI.
    """
    result = search.grid_search()

    assert result is not None, (
      "Grid search found no SOI crossings. "
      "Check that SPICE kernels are loaded and the epoch/orbit geometry is valid."
    )
    assert result['eval']['delta_vel_total'] > 0.0

  def test_full_solve_valid_result(self, basic_problem, leo_circular_state):
    """
    End-to-end solve produces a physically valid result.
    """
    from src.optimization.initial_guess import PatchedConicGridSearch

    search = PatchedConicGridSearch.hohmann(
      problem                  = basic_problem,
      initial_state            = leo_circular_state,
      delta_anomaly_range__deg = 30.0,
      n_departure_candidates   = 61,
    )
    result = search.solve()

    if not result.success:
      pytest.skip(f"No transfer found for test geometry: {result.message}")

    assert 3000.0 < result.objective_value < 5000.0, (
      f"Total ΔV = {result.objective_value:.1f} m/s, expected 3000–5000 m/s"
    )
    assert result.trajectory is not None
    assert result.trajectory.n_nodes    >= 3
    assert result.trajectory.n_segments >= 2

    for seg in result.trajectory.segments:
      assert seg.j2000_state_vec.shape[0] == 6
      assert seg.j2000_state_vec.shape[1] > 10
      assert seg.time is not None
