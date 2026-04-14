"""
Unit Tests for Optimization Module
====================================

Tests for core mathematical functions and decision vector packing.
No SPICE kernels required.

Tests:
------
TestPatchedConicFunctions
  - test_soi_radius_moon                          : verify Moon SOI radius
  - test_circular_velocity                        : verify circular velocity computation
  - test_hohmann_estimates                        : verify Hohmann transfer estimates
  - test_two_body_propagation_period              : verify orbit returns to start after one period
  - test_two_body_propagation_quarter             : verify quarter-orbit propagation
  - test_two_body_propagation_energy_conservation : verify energy conservation

TestDecisionVector
  - test_pack_no_variables                        : all fixed flags → empty array
  - test_pack_unpack_velocity_roundtrip           : variable velocity components pack/unpack correctly

Usage:
------
  python -m pytest src/verification/unit/test_optimization.py -v
"""
import pytest
import numpy as np

from copy     import deepcopy
from datetime import datetime

from src.model.constants             import SOLARSYSTEMCONSTANTS
from src.model.orbital_mechanics     import compute_circular_velocity, compute_hohmann_velocities, compute_soi_radius
from src.propagation.analytical_propagator import propagate_two_body


class TestPatchedConicFunctions:
  """
  Tests for core patched conic mathematical functions.
  """

  def test_soi_radius_moon(self):
    """
    Verify Moon's sphere of influence radius.
    Expected: ~66,183 km (textbook value).
    """
    radius_soi     = compute_soi_radius(
      sma          = SOLARSYSTEMCONSTANTS.MOON.SMA,
      gp_primary   = SOLARSYSTEMCONSTANTS.EARTH.GP,
      gp_secondary = SOLARSYSTEMCONSTANTS.MOON.GP,
    )
    radius_soi__km = radius_soi / 1000.0

    assert 60_000 < radius_soi__km < 70_000, \
      f"Moon SOI = {radius_soi__km:.0f} km, expected ~66,000 km"


  def test_circular_velocity(self):
    """
    Verify circular velocity computation for LEO (200 km altitude).
    """
    pos_mag                = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR + 200_000.0
    vel_mag_circ           = compute_circular_velocity(pos_mag, SOLARSYSTEMCONSTANTS.EARTH.GP)
    vel_mag_circ__km_per_s = vel_mag_circ / 1000.0

    assert 7.5 < vel_mag_circ__km_per_s < 8.0, \
      f"vel_mag_circ = {vel_mag_circ__km_per_s:.3f} km/s, expected ~7.78 km/s"


  def test_hohmann_estimates(self):
    """
    Verify Hohmann transfer estimates from LEO to Moon orbit.
    """
    radius_o  = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR + 200_000.0
    radius_f  = SOLARSYSTEMCONSTANTS.MOON.SMA
    estimates = compute_hohmann_velocities(radius_o, radius_f, SOLARSYSTEMCONSTANTS.EARTH.GP)

    delta_vel_mag_1__km_per_s = estimates['delta_vel_mag_o'] / 1000.0
    assert 2.8 < delta_vel_mag_1__km_per_s < 3.5, \
      f"ΔV₁ = {delta_vel_mag_1__km_per_s:.3f} km/s, expected ~3.1 km/s"

    transfer_time__days = estimates['delta_time_of'] / 86400.0
    assert 4.0 < transfer_time__days < 6.0, \
      f"Transfer time = {transfer_time__days:.2f} days, expected ~5 days"


  def test_two_body_propagation_period(self):
    """
    Verify two-body propagation returns to start after one period (circular orbit).
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


  def test_two_body_propagation_quarter(self):
    """
    Verify quarter-orbit propagation position and velocity.
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


  def test_two_body_propagation_energy_conservation(self):
    """
    Verify energy conservation in two-body numerical propagation.
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


class TestDecisionVector:
  """
  Unit tests for decision vector packing and unpacking.
  """

  def test_pack_no_variables(self):
    """
    All variable flags False → pack returns empty array.
    """
    from src.schemas.optimization            import DecisionState
    from src.optimization.maneuver_optimizer import pack_decision_vector

    ds = DecisionState(epoch=datetime(2025, 10, 1))
    x  = pack_decision_vector(ds)

    assert x.shape == (0,), f"Expected empty array, got shape {x.shape}"


  def test_pack_unpack_velocity_roundtrip(self):
    """
    Variable velocity components pack and unpack back to original values.
    """
    from src.schemas.optimization            import DecisionState
    from src.optimization.maneuver_optimizer import pack_decision_vector, unpack_decision_vector

    vel_vec = np.array([100.0, 200.0, 300.0])
    ds = DecisionState(
      epoch             = datetime(2025, 10, 1),
      velocity          = vel_vec.copy(),
      variable_velocity = np.array([True, True, True]),
    )

    x = pack_decision_vector(ds)
    assert x.shape == (3,), f"Expected 3 variables, got shape {x.shape}"
    np.testing.assert_array_equal(x, vel_vec)

    ds2          = deepcopy(ds)
    ds2.velocity = np.zeros(3)
    unpack_decision_vector(x, ds2)
    np.testing.assert_array_equal(ds2.velocity, vel_vec)
