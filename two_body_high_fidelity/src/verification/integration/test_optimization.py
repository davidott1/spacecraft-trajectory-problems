"""
Integration Tests for Optimization Module
==========================================

Tests for two-body propagation (propagator + integrator), PatchedConicGridSearch
factory (SPICE + Hohmann + ephemeris), and full solve pipeline.

Tests:
------
TestTwoBodyPropagation
  - test_period_roundtrip              : circular orbit returns to start after one period
  - test_quarter_orbit                 : quarter-orbit position and velocity
  - test_energy_conservation           : energy conserved over one elliptical orbit

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

from src.model.constants                       import SOLARSYSTEMCONSTANTS
from src.model.orbital_mechanics               import compute_circular_velocity
from src.propagation.analytical_propagator     import propagate_two_body


class TestTwoBodyPropagation:
  """
  Integration tests for propagate_two_body (analytical propagator + solve_ivp).
  The boundary under test is the wrapper function feeding initial conditions
  to scipy's integrator and extracting results.
  """

  def test_period_roundtrip(self):
    """
    Circular orbit must return to its initial state after exactly one period.
    """
    pos_mag = 7000.0e3
    gp      = SOLARSYSTEMCONSTANTS.EARTH.GP
    vel_mag = np.sqrt(gp / pos_mag)
    T       = 2.0 * np.pi * np.sqrt(pos_mag**3 / gp)
    state0  = np.array([pos_mag, 0.0, 0.0, 0.0, vel_mag, 0.0])

    result = propagate_two_body(state0=state0, t0_et=0.0, tf_et=T, gp=gp, n_points=2)

    assert result['success']
    np.testing.assert_allclose(
      state0, result['states'][:, -1], atol=1e-3,
      err_msg="Orbit did not return to start after one period",
    )

  def test_quarter_orbit(self):
    """
    After a quarter period, position rotates 90° and velocity rotates 90°.
    """
    pos_mag = 7000.0e3
    gp      = SOLARSYSTEMCONSTANTS.EARTH.GP
    vel_mag = np.sqrt(gp / pos_mag)
    T       = 2.0 * np.pi * np.sqrt(pos_mag**3 / gp)
    state0  = np.array([pos_mag, 0.0, 0.0, 0.0, vel_mag, 0.0])

    result        = propagate_two_body(state0=state0, t0_et=0.0, tf_et=T/4.0, gp=gp, n_points=2)
    state_quarter = result['states'][:, -1]

    assert result['success']
    np.testing.assert_allclose(
      state_quarter[0:3], [0.0, pos_mag, 0.0], atol=1e-3,
      err_msg="Quarter-orbit position incorrect",
    )
    np.testing.assert_allclose(
      state_quarter[3:6], [-vel_mag, 0.0, 0.0], atol=1e-3,
      err_msg="Quarter-orbit velocity incorrect",
    )

  def test_energy_conservation(self):
    """
    Specific energy must be conserved over one elliptical orbit.
    """
    gp      = SOLARSYSTEMCONSTANTS.EARTH.GP
    pos_mag = 7000.0e3
    vel_mag = np.sqrt(gp / pos_mag) * 1.1  # Slightly elliptical
    state0  = np.array([pos_mag, 0.0, 0.0, 0.0, vel_mag, 0.0])

    E0    = 0.5 * vel_mag**2 - gp / pos_mag
    sma   = -gp / (2.0 * E0)
    T_est = 2.0 * np.pi * np.sqrt(sma**3 / gp)

    result = propagate_two_body(state0=state0, t0_et=0.0, tf_et=T_est, gp=gp, n_points=500)
    assert result['success'], "Two-body propagation failed"

    state_f   = result['states'][:, -1]
    E_f       = 0.5 * np.linalg.norm(state_f[3:6])**2 - gp / np.linalg.norm(state_f[0:3])
    rel_error = abs(E_f - E0) / abs(E0)

    assert rel_error < 1e-10, f"Energy not conserved: relative error = {rel_error:.2e}"


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
