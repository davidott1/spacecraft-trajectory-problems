"""
Unit Tests for Dynamics Module
==============================

Each test verifies a single function's mathematical correctness.
No ODE integration, no SPICE kernels.

Tests:
------
TestTwoBodyGravity
  - test_sanity_check_point_mass_direction          : acceleration points toward central body
  - test_sanity_check_point_mass_magnitude_scaling  : inverse square law (2x distance = 1/4 acceleration)
  - test_sanity_check_point_mass_magnitude_leo      : ~8-9 m/s² at LEO altitude
  - test_acceleration_magnitude                     : |a| = μ/r² for any position
  - test_acceleration_at_known_values               : hand-computed result for Earth at 7000 km
  - test_acceleration_off_axis                      : correct for non-axis-aligned position
  - test_sanity_check_j2_acc_not_zero               : J2 produces non-zero acceleration for inclined orbits
  - test_sanity_check_j2_zero_when_disabled         : J2=0 produces zero J2 acceleration
  - test_relative_check_j2_smaller_than_point_mass  : J2 << point mass (~1000x smaller)

TestTwoBodyPointMassJacobian
  - test_jacobian_structure                         : upper-right block is I, lower-right is 0
  - test_jacobian_vs_finite_difference              : analytical Jacobian matches numerical
  - test_jacobian_symmetry                          : gravity gradient ∂a/∂r is symmetric

TestSchwarzschildRelativity
  - test_circular_orbit_radial_correction           : GR correction is radial for circular orbit
  - test_magnitude_order                            : GR acceleration << Newtonian for LEO

TestAtmosphericDrag
  - test_sanity_check_drag_opposes_velocity                : drag opposes velocity direction
  - test_relative_check_drag_increases_with_lower_altitude : drag increases at lower altitudes
  - test_sanity_check_drag_zero_at_high_altitude           : drag is negligible at GEO altitude

TestAccelerationCoordinator
  - test_physical_laws_gravity_only                 : gravity-only acceleration matches point mass
  - test_relative_check_drag_adds_to_gravity        : drag modifies total acceleration when enabled

TestGetHarmonicCoefficients
  - test_j2_returns_nonzero                         : requesting J2 returns Earth's J2 value
  - test_unknown_harmonic_ignored                   : unrecognized names leave coefficients at zero
  - test_empty_list_all_zero                        : empty list returns all-zero dict

TestStateTimeDerivative
  - test_output_packs_vel_acc                       : derivative = [vel, acc] for a given state

TestEnergyConservation
  - test_physical_laws_specific_energy_formula      : E = v²/2 - μ/r = -μ/(2a) for circular orbit

Usage:
------
  python -m pytest src/verification/unit/test_dynamics.py -v
"""
import pytest
import numpy as np

from src.model.dynamics    import (
  TwoBodyGravity, GeneralRelativity, AtmosphericDrag,
  AccelerationSTMDot, GeneralStateEquationsOfMotion,
  _get_harmonic_coefficients,
)
from src.model.constants   import SOLARSYSTEMCONSTANTS
from src.schemas.gravity   import GravityModelConfig
from src.schemas.spacecraft import SpacecraftProperties, DragConfig


class TestTwoBodyGravity:
  """
  Unit tests for TwoBodyGravity.point_mass() and oblate_j2().
  """

  def test_sanity_check_point_mass_direction(self):
    """
    Point mass acceleration should point toward central body.
    """
    gravity = TwoBodyGravity(gp=SOLARSYSTEMCONSTANTS.EARTH.GP)

    pos_vec = np.array([7000e3, 0.0, 0.0])
    acc_vec = gravity.point_mass(pos_vec)

    assert acc_vec[0] < 0
    assert np.abs(acc_vec[1]) < 1e-15
    assert np.abs(acc_vec[2]) < 1e-15

  def test_sanity_check_point_mass_magnitude_scaling(self):
    """
    Point mass acceleration magnitude should follow inverse square law.
    """
    gravity = TwoBodyGravity(gp=SOLARSYSTEMCONSTANTS.EARTH.GP)

    pos_vec_1 = np.array([7000e3, 0.0, 0.0])
    pos_vec_2 = np.array([14000e3, 0.0, 0.0])

    acc_mag_1 = np.linalg.norm(gravity.point_mass(pos_vec_1))
    acc_mag_2 = np.linalg.norm(gravity.point_mass(pos_vec_2))

    ratio = acc_mag_1 / acc_mag_2
    assert np.isclose(ratio, 4.0, rtol=1e-10)

  def test_sanity_check_point_mass_magnitude_leo(self):
    """
    Point mass acceleration at LEO altitude should be ~8-9 m/s².
    """
    gravity = TwoBodyGravity(gp=SOLARSYSTEMCONSTANTS.EARTH.GP)

    pos_vec = np.array([6778e3, 0.0, 0.0])
    acc_mag = np.linalg.norm(gravity.point_mass(pos_vec))

    assert 8.0 < acc_mag < 9.0

  def test_acceleration_magnitude(self):
    """
    For any position vector, |a| must equal μ/r².
    """
    gp      = SOLARSYSTEMCONSTANTS.EARTH.GP
    gravity = TwoBodyGravity(gp=gp)
    pos_vec = np.array([7000.0e3, 0.0, 0.0])
    acc_vec = gravity.point_mass(pos_vec)

    pos_mag      = np.linalg.norm(pos_vec)
    expected_mag = gp / pos_mag**2

    assert np.isclose(np.linalg.norm(acc_vec), expected_mag, rtol=1e-14), (
      f"|a| = {np.linalg.norm(acc_vec):.6e}, expected μ/r² = {expected_mag:.6e}"
    )

  def test_acceleration_at_known_values(self):
    """
    Hand-computed: at r = [7000 km, 0, 0], a_x = -μ/r² ≈ -8.1396 m/s².
    """
    gp      = SOLARSYSTEMCONSTANTS.EARTH.GP
    gravity = TwoBodyGravity(gp=gp)
    pos_mag = 7000.0e3
    pos_vec = np.array([pos_mag, 0.0, 0.0])
    acc_vec = gravity.point_mass(pos_vec)

    expected_ax = -gp / pos_mag**2
    assert np.isclose(acc_vec[0], expected_ax, rtol=1e-14)
    assert acc_vec[1] == 0.0
    assert acc_vec[2] == 0.0

  def test_acceleration_off_axis(self):
    """
    For a position not aligned with any axis, acceleration still satisfies
    a = -μ r̂ / r² and each component scales with the position component.
    """
    gp      = SOLARSYSTEMCONSTANTS.EARTH.GP
    gravity = TwoBodyGravity(gp=gp)
    pos_vec = np.array([4000.0e3, 5000.0e3, 3000.0e3])
    acc_vec = gravity.point_mass(pos_vec)

    pos_mag  = np.linalg.norm(pos_vec)
    expected = -gp * pos_vec / pos_mag**3

    np.testing.assert_allclose(acc_vec, expected, rtol=1e-14,
      err_msg="Off-axis acceleration incorrect")

  def test_sanity_check_j2_acc_not_zero(self):
    """
    J2 acceleration should be non-zero for non-equatorial orbits.
    """
    gravity = TwoBodyGravity(
      gp      = SOLARSYSTEMCONSTANTS.EARTH.GP,
      j2      = SOLARSYSTEMCONSTANTS.EARTH.J2,
      pos_ref = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR,
    )

    pos_vec = np.array([5000e3, 3000e3, 4000e3])
    acc_j2  = gravity.oblate_j2(0.0, pos_vec)

    assert np.linalg.norm(acc_j2) > 0

  def test_sanity_check_j2_zero_when_disabled(self):
    """
    J2 acceleration should be zero when J2=0.
    """
    gravity = TwoBodyGravity(
      gp      = SOLARSYSTEMCONSTANTS.EARTH.GP,
      j2      = 0.0,
      pos_ref = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR,
    )

    pos_vec = np.array([7000e3, 0.0, 1000e3])
    acc_j2  = gravity.oblate_j2(0.0, pos_vec)

    assert np.allclose(acc_j2, np.zeros(3))

  def test_relative_check_j2_smaller_than_point_mass(self):
    """
    J2 acceleration should be much smaller than point mass.
    """
    gravity = TwoBodyGravity(
      gp      = SOLARSYSTEMCONSTANTS.EARTH.GP,
      j2      = SOLARSYSTEMCONSTANTS.EARTH.J2,
      pos_ref = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR,
    )

    pos_vec = np.array([7000e3, 0.0, 1000e3])

    acc_pm = np.linalg.norm(gravity.point_mass(pos_vec))
    acc_j2 = np.linalg.norm(gravity.oblate_j2(0.0, pos_vec))

    assert acc_j2 < acc_pm * 0.01


class TestTwoBodyPointMassJacobian:
  """
  Unit tests for TwoBodyGravity.point_mass_jacobian().
  Verifies the 6x6 Jacobian structure and numerical accuracy.
  """

  @pytest.fixture
  def gravity(self):
    return TwoBodyGravity(gp=SOLARSYSTEMCONSTANTS.EARTH.GP)

  def test_jacobian_structure(self, gravity):
    """
    The 6x6 Jacobian must have: upper-left = 0, upper-right = I,
    lower-right = 0. Lower-left = ∂a/∂r (tested separately).
    """
    pos_vec = np.array([7000.0e3, 0.0, 0.0])
    jac     = gravity.point_mass_jacobian(pos_vec)

    assert jac.shape == (6, 6)
    np.testing.assert_array_equal(jac[0:3, 0:3], np.zeros((3, 3)),
      err_msg="Upper-left block should be zero")
    np.testing.assert_array_equal(jac[0:3, 3:6], np.eye(3),
      err_msg="Upper-right block should be identity")
    np.testing.assert_array_equal(jac[3:6, 3:6], np.zeros((3, 3)),
      err_msg="Lower-right block should be zero")

  def test_jacobian_vs_finite_difference(self, gravity):
    """
    Analytical gravity gradient ∂a/∂r must match central finite difference
    to ~1e-6 relative accuracy.
    """
    pos_vec = np.array([4000.0e3, 5000.0e3, 3000.0e3])
    jac     = gravity.point_mass_jacobian(pos_vec)

    daccvec__dposvec = jac[3:6, 0:3]

    h = 1.0
    daccvec__dposvec_fd = np.zeros((3, 3))
    for i in range(3):
      pos_plus     = pos_vec.copy()
      pos_minus    = pos_vec.copy()
      pos_plus[i]  += h
      pos_minus[i] -= h
      daccvec__dposvec_fd[:, i] = (
        gravity.point_mass(pos_plus) - gravity.point_mass(pos_minus)
      ) / (2.0 * h)

    np.testing.assert_allclose(daccvec__dposvec, daccvec__dposvec_fd, rtol=1e-6,
      err_msg="Analytical Jacobian does not match finite difference")

  def test_jacobian_symmetry(self, gravity):
    """
    The gravity gradient ∂a/∂r for a central force must be symmetric.
    """
    pos_vec          = np.array([4000.0e3, 5000.0e3, 3000.0e3])
    jac              = gravity.point_mass_jacobian(pos_vec)
    daccvec__dposvec = jac[3:6, 0:3]

    np.testing.assert_allclose(daccvec__dposvec, daccvec__dposvec.T, atol=1e-20,
      err_msg="Gravity gradient is not symmetric")


class TestSchwarzschildRelativity:
  """
  Unit tests for GeneralRelativity.schwarzschild().
  """

  @pytest.fixture
  def gr_model(self):
    return GeneralRelativity(gp=SOLARSYSTEMCONSTANTS.EARTH.GP)

  def test_circular_orbit_radial_correction(self, gr_model):
    """
    For a circular orbit (r · v = 0), the GR correction simplifies to:
      a_GR = (μ/c²r³)(4μ/r - v²) r
    which is purely radial (parallel to r).
    """
    gp      = SOLARSYSTEMCONSTANTS.EARTH.GP
    pos_mag = 7000.0e3
    vel_mag = np.sqrt(gp / pos_mag)
    pos_vec = np.array([pos_mag, 0.0, 0.0])
    vel_vec = np.array([0.0, vel_mag, 0.0])

    acc_gr = gr_model.schwarzschild(pos_vec, vel_vec)

    assert abs(acc_gr[1]) < 1e-30, f"Non-radial y-component: {acc_gr[1]:.2e}"
    assert abs(acc_gr[2]) < 1e-30, f"Non-radial z-component: {acc_gr[2]:.2e}"
    assert acc_gr[0] != 0.0, "Radial component should be nonzero"

  def test_magnitude_order(self, gr_model):
    """
    For LEO, GR correction should be ~1e-9 m/s², many orders of magnitude
    smaller than Newtonian gravity (~8 m/s²).
    """
    gp      = SOLARSYSTEMCONSTANTS.EARTH.GP
    pos_mag = 7000.0e3
    vel_mag = np.sqrt(gp / pos_mag)
    pos_vec = np.array([pos_mag, 0.0, 0.0])
    vel_vec = np.array([0.0, vel_mag, 0.0])

    acc_gr       = gr_model.schwarzschild(pos_vec, vel_vec)
    acc_gr_mag   = np.linalg.norm(acc_gr)
    acc_newt_mag = gp / pos_mag**2

    ratio = acc_gr_mag / acc_newt_mag
    assert ratio < 1e-8, f"GR/Newtonian ratio = {ratio:.2e}, expected < 1e-8"
    assert ratio > 1e-12, f"GR/Newtonian ratio = {ratio:.2e}, suspiciously small"


class TestAtmosphericDrag:
  """
  Unit tests for AtmosphericDrag.compute().
  """

  def test_sanity_check_drag_opposes_velocity(self):
    """
    Drag acceleration should oppose velocity direction.
    """
    drag_config = DragConfig(enabled=True, cd=2.2, area=10.0)
    drag = AtmosphericDrag(drag_config=drag_config, mass=1000.0)

    pos_vec = np.array([6778e3, 0.0, 0.0])
    vel_vec = np.array([0.0, 7500.0, 0.0])

    acc_drag = drag.compute(pos_vec, vel_vec)

    assert acc_drag[1] < 0

  def test_relative_check_drag_increases_with_lower_altitude(self):
    """
    Drag should increase at lower altitudes.
    """
    drag_config = DragConfig(enabled=True, cd=2.2, area=10.0)
    drag = AtmosphericDrag(drag_config=drag_config, mass=1000.0)

    vel_vec = np.array([0.0, 7500.0, 0.0])

    pos_low  = np.array([6578e3, 0.0, 0.0])
    pos_high = np.array([6978e3, 0.0, 0.0])

    acc_low  = np.linalg.norm(drag.compute(pos_low, vel_vec))
    acc_high = np.linalg.norm(drag.compute(pos_high, vel_vec))

    assert acc_low > acc_high

  def test_sanity_check_drag_zero_at_high_altitude(self):
    """
    Drag should be negligible at high altitudes.
    """
    drag_config = DragConfig(enabled=True, cd=2.2, area=10.0)
    drag = AtmosphericDrag(drag_config=drag_config, mass=1000.0)

    pos_vec = np.array([42164e3, 0.0, 0.0])
    vel_vec = np.array([0.0, 3075.0, 0.0])

    acc_drag = np.linalg.norm(drag.compute(pos_vec, vel_vec))

    assert acc_drag < 1e-20


class TestAccelerationCoordinator:
  """
  Unit tests for AccelerationSTMDot.compute().
  """

  def test_physical_laws_gravity_only(self):
    """
    Gravity-only acceleration should match point mass.
    """
    gravity_config = GravityModelConfig(gp=SOLARSYSTEMCONSTANTS.EARTH.GP)
    spacecraft     = SpacecraftProperties(mass=1000.0)
    accel          = AccelerationSTMDot(gravity_config=gravity_config, spacecraft=spacecraft)

    pos_vec = np.array([7000e3, 0.0, 0.0])
    vel_vec = np.array([0.0, 7500.0, 0.0])

    acc_total = accel.compute(0.0, pos_vec, vel_vec)

    expected = -SOLARSYSTEMCONSTANTS.EARTH.GP * pos_vec / np.linalg.norm(pos_vec)**3
    assert np.allclose(acc_total, expected, rtol=1e-10)

  def test_relative_check_drag_adds_to_gravity(self):
    """
    Drag should modify total acceleration when enabled.
    """
    gravity_config = GravityModelConfig(gp=SOLARSYSTEMCONSTANTS.EARTH.GP)

    spacecraft_no_drag = SpacecraftProperties(
      mass = 1000.0,
      drag = DragConfig(enabled=False),
    )
    spacecraft_with_drag = SpacecraftProperties(
      mass = 1000.0,
      drag = DragConfig(enabled=True, cd=2.2, area=10.0),
    )

    accel_no_drag   = AccelerationSTMDot(gravity_config=gravity_config, spacecraft=spacecraft_no_drag)
    accel_with_drag = AccelerationSTMDot(gravity_config=gravity_config, spacecraft=spacecraft_with_drag)

    pos_vec = np.array([6578e3, 0.0, 0.0])
    vel_vec = np.array([0.0, 7800.0, 0.0])

    acc_no_drag   = accel_no_drag.compute(0.0, pos_vec, vel_vec)
    acc_with_drag = accel_with_drag.compute(0.0, pos_vec, vel_vec)

    assert not np.allclose(acc_no_drag, acc_with_drag)


class TestGetHarmonicCoefficients:
  """
  Unit tests for _get_harmonic_coefficients().
  """

  def test_j2_returns_nonzero(self):
    """Requesting 'J2' must return Earth's known J2 value."""
    coeffs = _get_harmonic_coefficients(['J2'])
    assert coeffs['j2'] == SOLARSYSTEMCONSTANTS.EARTH.J2
    assert coeffs['j2'] > 1e-4, "J2 should be ~1.08e-3"

  def test_unknown_harmonic_ignored(self):
    """Unrecognized harmonic names must leave all coefficients at zero."""
    coeffs = _get_harmonic_coefficients(['FAKE', 'J99'])
    for key, val in coeffs.items():
      assert val == 0.0, f"Expected 0 for '{key}', got {val}"

  def test_empty_list_all_zero(self):
    """Empty list must return all-zero coefficients."""
    coeffs = _get_harmonic_coefficients([])
    for key, val in coeffs.items():
      assert val == 0.0, f"Expected 0 for '{key}', got {val}"


class TestStateTimeDerivative:
  """
  Unit test for GeneralStateEquationsOfMotion.state_time_derivative().
  Verifies the function correctly packs [vel, acc] — no integration.
  """

  def test_output_packs_vel_acc(self):
    """
    state_time_derivative must return [vel_vec, acc_vec] where acc_vec
    comes from AccelerationSTMDot.compute() for the given position.
    """
    gp    = SOLARSYSTEMCONSTANTS.EARTH.GP
    accel = AccelerationSTMDot(
      gravity_config = GravityModelConfig(gp=gp),
      spacecraft     = SpacecraftProperties(mass=1000.0),
    )
    eom = GeneralStateEquationsOfMotion(accel)

    pos_vec   = np.array([7000.0e3, 0.0, 0.0])
    vel_vec   = np.array([0.0, 7546.0, 0.0])
    state_vec = np.concatenate([pos_vec, vel_vec])

    state_dot = eom.state_time_derivative(0.0, state_vec)

    np.testing.assert_array_equal(state_dot[0:3], vel_vec,
      err_msg="Derivative position part should equal velocity")

    expected_acc = accel.compute(0.0, pos_vec, vel_vec)
    np.testing.assert_array_equal(state_dot[3:6], expected_acc,
      err_msg="Derivative velocity part should equal acceleration")


class TestEnergyConservation:
  """
  Unit test for specific orbital energy formula.
  """

  def test_physical_laws_specific_energy_formula(self):
    """
    Verify E = v²/2 - μ/r = -μ/(2a) for circular orbit.
    """
    gp = SOLARSYSTEMCONSTANTS.EARTH.GP

    pos_mag = 7000e3
    vel_mag = np.sqrt(gp / pos_mag)

    pos_vec = np.array([pos_mag, 0.0, 0.0])
    vel_vec = np.array([0.0, vel_mag, 0.0])

    energy          = 0.5 * np.dot(vel_vec, vel_vec) - gp / np.linalg.norm(pos_vec)
    expected_energy = -gp / (2 * pos_mag)

    assert np.isclose(energy, expected_energy, rtol=1e-10)
