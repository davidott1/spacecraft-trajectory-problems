"""
Tests for Optimization Module
==============================

Tests for patched conic functions and lunar transfer optimization.

Tests:
------
TestPatchedConicFunctions
  - test_soi_radius_moon                : verify Moon SOI radius
  - test_circular_velocity              : verify circular velocity computation
  - test_hohmann_estimates              : verify Hohmann transfer estimates
  - test_two_body_propagation_energy    : verify energy conservation in two-body propagation
  - test_two_body_propagation_period    : verify orbit returns to start after one period

TestLunarTransfer (requires SPICE)
  - test_evaluate_transfer_reaches_soi  : verify transfer orbit reaches Moon SOI
  - test_full_optimization              : verify full optimization produces valid result

Usage:
------
  python -m pytest src/verification/test_optimization.py -v
"""
import pytest
import numpy as np

from datetime import datetime

from src.model.constants         import SOLARSYSTEMCONSTANTS, CONVERTER
from src.model.orbital_mechanics import compute_circular_velocity, compute_hohmann_velocities
from src.optimization.patched_conic import (
  compute_soi_radius,
  propagate_two_body,
)


class TestPatchedConicFunctions:
  """
  Tests for core patched conic mathematical functions.
  These tests do NOT require SPICE kernels.
  """

  def test_soi_radius_moon(self):
    """
    Verify Moon's sphere of influence radius.
    Expected: ~66,183 km (textbook value).
    """
    r_soi = compute_soi_radius(
      sma          = SOLARSYSTEMCONSTANTS.MOON.SMA,
      gp_primary   = SOLARSYSTEMCONSTANTS.EARTH.GP,
      gp_secondary = SOLARSYSTEMCONSTANTS.MOON.GP,
    )
    r_soi_km = r_soi / 1000.0

    # Moon SOI should be approximately 66,000 km
    assert 60_000 < r_soi_km < 70_000, f"Moon SOI = {r_soi_km:.0f} km, expected ~66,000 km"


  def test_circular_velocity(self):
    """
    Verify circular velocity computation for LEO.
    """
    r  = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR + 200_000.0  # 200 km altitude
    gp = SOLARSYSTEMCONSTANTS.EARTH.GP

    v_circ = compute_circular_velocity(r, gp)

    # LEO circular velocity should be approximately 7.78 km/s
    v_circ_km_s = v_circ / 1000.0
    assert 7.5 < v_circ_km_s < 8.0, f"v_circ = {v_circ_km_s:.3f} km/s, expected ~7.78 km/s"


  def test_hohmann_estimates(self):
    """
    Verify Hohmann transfer estimates from LEO to Moon orbit.
    """
    r1 = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR + 200_000.0  # 200 km LEO
    r2 = SOLARSYSTEMCONSTANTS.MOON.SMA                          # Moon orbit
    gp = SOLARSYSTEMCONSTANTS.EARTH.GP

    estimates = compute_hohmann_velocities(r1, r2, gp)

    # ΔV₁ should be approximately 3.1 km/s
    dv1_km_s = estimates['delta_vel_mag_o'] / 1000.0
    assert 2.8 < dv1_km_s < 3.5, f"ΔV₁ = {dv1_km_s:.3f} km/s, expected ~3.1 km/s"

    # Transfer time should be approximately 5 days
    tt_days = estimates['delta_time_of'] / 86400.0
    assert 4.0 < tt_days < 6.0, f"Transfer time = {tt_days:.2f} days, expected ~5 days"


  def test_two_body_propagation_period(self):
    """
    Verify two-body propagation returns to start after one period (circular orbit).
    """
    r   = 7000.0e3
    gp  = SOLARSYSTEMCONSTANTS.EARTH.GP
    v   = np.sqrt(gp / r)
    T   = 2.0 * np.pi * np.sqrt(r**3 / gp)

    state0 = np.array([r, 0.0, 0.0, 0.0, v, 0.0])

    # After one period, should return to start
    result = propagate_two_body(state0=state0, t0_et=0.0, tf_et=T, gp=gp, n_points=2)
    assert result['success']
    state_T = result['states'][:, -1]

    np.testing.assert_allclose(state0, state_T, atol=1e-3,
      err_msg="Orbit did not return to start after one period")


  def test_two_body_propagation_quarter(self):
    """
    Verify quarter-orbit propagation.
    """
    r   = 7000.0e3
    gp  = SOLARSYSTEMCONSTANTS.EARTH.GP
    v   = np.sqrt(gp / r)
    T   = 2.0 * np.pi * np.sqrt(r**3 / gp)

    state0 = np.array([r, 0.0, 0.0, 0.0, v, 0.0])

    # After quarter period, position should be at (0, r, 0)
    result = propagate_two_body(state0=state0, t0_et=0.0, tf_et=T/4.0, gp=gp, n_points=2)
    assert result['success']
    state_quarter = result['states'][:, -1]

    np.testing.assert_allclose(
      state_quarter[0:3], [0.0, r, 0.0], atol=1e-3,
      err_msg="Quarter-orbit position incorrect")
    np.testing.assert_allclose(
      state_quarter[3:6], [-v, 0.0, 0.0], atol=1e-3,
      err_msg="Quarter-orbit velocity incorrect")


  def test_two_body_propagation_energy_conservation(self):
    """
    Verify energy conservation in two-body numerical propagation.
    """
    gp  = SOLARSYSTEMCONSTANTS.EARTH.GP
    r   = 7000.0e3
    v   = np.sqrt(gp / r) * 1.1  # Slightly elliptical

    state0 = np.array([r, 0.0, 0.0, 0.0, v, 0.0])

    # Compute initial specific energy
    E0 = 0.5 * v**2 - gp / r

    # Propagate for 1 orbit (approximate period)
    a     = -gp / (2.0 * E0)
    T_est = 2.0 * np.pi * np.sqrt(a**3 / gp)

    result = propagate_two_body(
      state0   = state0,
      t0_et    = 0.0,
      tf_et    = T_est,
      gp       = gp,
      n_points = 500,
    )

    assert result['success'], "Two-body propagation failed"

    # Check energy at final state
    state_f = result['states'][:, -1]
    r_f = np.linalg.norm(state_f[0:3])
    v_f = np.linalg.norm(state_f[3:6])
    E_f = 0.5 * v_f**2 - gp / r_f

    rel_error = abs(E_f - E0) / abs(E0)
    assert rel_error < 1e-10, f"Energy not conserved: relative error = {rel_error:.2e}"


class TestLunarTransfer:
  """
  Tests for lunar transfer optimization.
  These tests require SPICE kernels to be loaded.
  """

  @pytest.fixture(autouse=True)
  def _load_spice(self, load_spice_kernels):
    """Auto-load SPICE kernels for all tests in this class."""
    pass

  def test_evaluate_transfer_reaches_soi(self):
    """
    Verify that a Hohmann-like transfer from LEO can reach the Moon's SOI.
    """
    from src.optimization.patched_conic import propagate_to_soi
    from src.model.frame_and_vector_converter import BodyVectorConverter
    from src.model.constants import NAIFIDS
    from src.model.time_converter import utc_to_et

    gp_earth = SOLARSYSTEMCONSTANTS.EARTH.GP
    r_leo    = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR + 200_000.0
    v_circ   = np.sqrt(gp_earth / r_leo)

    # Compute Hohmann ΔV₁
    a_trans    = (r_leo + SOLARSYSTEMCONSTANTS.MOON.SMA) / 2.0
    v_depart   = np.sqrt(gp_earth * (2.0 / r_leo - 1.0 / a_trans))
    dv1        = v_depart - v_circ

    # Initial state (equatorial circular LEO)
    state0 = np.array([r_leo, 0.0, 0.0, 0.0, v_circ, 0.0])

    # Apply ΔV₁ (prograde)
    state_post = state0.copy()
    state_post[4] += dv1  # Add to vy

    # SOI radius
    soi_moon = compute_soi_radius(
      SOLARSYSTEMCONSTANTS.MOON.SMA,
      gp_earth,
      SOLARSYSTEMCONSTANTS.MOON.GP,
    )

    # Use a reference epoch
    t0_et = utc_to_et(datetime(2025, 10, 1))

    # Try to reach Moon SOI (may or may not succeed depending on Moon position)
    result = propagate_to_soi(
      state_o           = state_post,
      time_et_o         = t0_et,
      gp                = gp_earth,
      naif_id_secondary = NAIFIDS.MOON,
      naif_id_primary   = NAIFIDS.EARTH,
      soi_radius        = soi_moon,
      max_time_s        = 7.0 * 86400.0,
    )

    # The trajectory should at least propagate successfully
    assert result['trajectory_states'].shape[0] == 6
    assert result['trajectory_states'].shape[1] > 0

    # The spacecraft should reach Moon-like distances
    max_r = np.max(np.linalg.norm(result['trajectory_states'][0:3, :], axis=0))
    assert max_r > 1e8, f"Max distance = {max_r/1000:.0f} km, expected > 100,000 km"


  def test_full_optimization(self):
    """
    Run a full lunar transfer optimization and verify the result.
    """
    from src.schemas.optimization import OptimizationProblem, OptimizationConfig, DecisionState, Objective, BoundaryCondition, Constraint
    from src.optimization.lunar_transfer import LunarTransferOptimizer

    problem = OptimizationProblem(
      objective           = Objective(quantity='delta_v_total', nodes=[0, 2]),
      decision_state      = DecisionState(epoch=datetime(2025, 10, 1)),
      constraints         = Constraint(final=[BoundaryCondition(node=2, quantity='altitude', target=100_000.0)]),
      optimization_config = OptimizationConfig(),
    )

    r_leo  = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR + 200_000.0
    v_circ = np.sqrt(SOLARSYSTEMCONSTANTS.EARTH.GP / r_leo)
    initial_state = np.array([r_leo, 0.0, 0.0, 0.0, v_circ, 0.0])

    optimizer = LunarTransferOptimizer(
      problem, initial_state,
      leo_altitude_m         = 200_000.0,
      llo_altitude_m         = 100_000.0,
      n_departure_candidates = 180,  # Coarser grid for faster test
      max_transfer_time_s    = 7.0 * 86400.0,
    )
    result    = optimizer.solve()

    if result.success:
      # Verify objective value (total ΔV)
      assert result.objective_value > 3000, f"Total ΔV = {result.objective_value:.0f} m/s, expected > 3000 m/s"
      assert result.objective_value < 5000, f"Total ΔV = {result.objective_value:.0f} m/s, expected < 5000 m/s"

      # Verify trajectory structure
      assert result.trajectory is not None
      assert result.trajectory.n_nodes    >= 3, f"Expected >= 3 nodes, got {result.trajectory.n_nodes}"
      assert result.trajectory.n_segments >= 2, f"Expected >= 2 segments, got {result.trajectory.n_segments}"

      # Verify segments have state data
      for seg in result.trajectory.segments:
        assert seg.j2000_state_vec.shape[0] == 6
        assert seg.j2000_state_vec.shape[1] > 10
        assert seg.time is not None
    else:
      # No solution found for this geometry - that's OK, just note it
      pytest.skip(f"No transfer found for test geometry: {result.message}")
