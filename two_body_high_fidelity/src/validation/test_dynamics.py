"""
Unit Tests for Dynamics Module
==============================

Tests for gravitational accelerations, atmospheric drag, and SRP models.

Tests:
------
TestTwoBodyGravity
  - test_sanity_check_point_mass_direction               : verify acceleration points toward central body
  - test_sanity_check_point_mass_magnitude_scaling       : verify inverse square law (2x distance = 1/4 acceleration)
  - test_sanity_check_point_mass_magnitude_leo           : verify ~8-9 m/s² at LEO altitude (~400 km)
  - test_sanity_check_j2_acc_not_zero                    : verify J2 produces non-zero acceleration for inclined orbits
  - test_sanity_check_j2_zero_when_disabled              : verify J2=0 produces zero J2 acceleration
  - test_relative_check_j2_smaller_than_point_mass       : verify J2 << point mass (~1000x smaller)

TestAtmosphericDrag
  - test_sanity_check_drag_opposes_velocity                : verify drag opposes velocity direction
  - test_relative_check_drag_increases_with_lower_altitude : verify drag increases at lower altitudes
  - test_sanity_check_drag_zero_at_high_altitude           : verify drag is negligible at GEO altitude

TestAccelerationCoordinator
  - test_physical_laws_gravity_only          : verify gravity-only acceleration matches point mass
  - test_relative_check_drag_adds_to_gravity : verify drag modifies total acceleration when enabled

TestEnergyConservation
  - test_physical_laws_specific_energy_formula : verify E = v²/2 - μ/r = -μ/(2a) for circular orbit

Usage:
------
  python -m pytest src/validation/test_dynamics.py -v
"""
import pytest
import numpy as np

from src.model.dynamics  import TwoBodyGravity, AtmosphericDrag, Acceleration
from src.model.constants import SOLARSYSTEMCONSTANTS


class TestTwoBodyGravity:
  """
  Tests for two-body gravitational acceleration.
  """
  
  def test_sanity_check_point_mass_direction(self):
    """
    Point mass acceleration should point toward central body.
    """
    gravity = TwoBodyGravity(gp=SOLARSYSTEMCONSTANTS.EARTH.GP)
    
    # Satellite at +X direction
    pos_vec = np.array([7000e3, 0.0, 0.0])
    acc_vec = gravity.point_mass(pos_vec)
    
    # Acceleration should point in -X direction
    assert acc_vec[0] < 0
    assert np.abs(acc_vec[1]) < 1e-15
    assert np.abs(acc_vec[2]) < 1e-15
  
  def test_sanity_check_point_mass_magnitude_scaling(self):
    """
    Point mass acceleration magnitude should follow inverse square law.
    """
    gravity = TwoBodyGravity(gp=SOLARSYSTEMCONSTANTS.EARTH.GP)
    
    pos_vec_1 = np.array([7000e3, 0.0, 0.0])
    pos_vec_2 = np.array([14000e3, 0.0, 0.0])  # Double the distance
    
    acc_mag_1 = np.linalg.norm(gravity.point_mass(pos_vec_1))
    acc_mag_2 = np.linalg.norm(gravity.point_mass(pos_vec_2))
    
    # At double distance, acceleration should be 1/4
    ratio = acc_mag_1 / acc_mag_2
    assert np.isclose(ratio, 4.0, rtol=1e-10)
  
  def test_sanity_check_point_mass_magnitude_leo(self):
    """
    Point mass acceleration at LEO altitude should be ~8-9 m/s².
    """
    gravity = TwoBodyGravity(gp=SOLARSYSTEMCONSTANTS.EARTH.GP)
    
    # ~400 km altitude (ISS orbit)
    pos_vec = np.array([6778e3, 0.0, 0.0])
    acc_mag = np.linalg.norm(gravity.point_mass(pos_vec))
    
    # Should be slightly less than 9.81 m/s² (surface gravity)
    assert 8.0 < acc_mag < 9.0
  
  def test_sanity_check_j2_acc_not_zero(self):
    """
    J2 acceleration should be non-zero for non-equatorial orbits.
    """
    gravity = TwoBodyGravity(
      gp      = SOLARSYSTEMCONSTANTS.EARTH.GP,
      j2      = SOLARSYSTEMCONSTANTS.EARTH.J2,
      pos_ref = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR,
    )
    
    # Position with Z component (inclined orbit)
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
    
    # J2 should be ~1000x smaller than point mass
    assert acc_j2 < acc_pm * 0.01


class TestAtmosphericDrag:
  """
  Tests for atmospheric drag model.
  """
  
  def test_sanity_check_drag_opposes_velocity(self):
    """
    Drag acceleration should oppose velocity direction.
    """
    drag = AtmosphericDrag(cd=2.2, area=10.0, mass=1000.0)
    
    pos_vec = np.array([6778e3, 0.0, 0.0])  # LEO altitude
    vel_vec = np.array([0.0, 7500.0, 0.0])  # Prograde velocity
    
    acc_drag = drag.compute(pos_vec, vel_vec)
    
    # Drag should have negative Y component (opposing velocity)
    # Note: includes Earth rotation effect, so not exactly opposite
    assert acc_drag[1] < 0
  
  def test_relative_check_drag_increases_with_lower_altitude(self):
    """
    Drag should increase at lower altitudes.
    """
    drag = AtmosphericDrag(cd=2.2, area=10.0, mass=1000.0)
    
    vel_vec = np.array([0.0, 7500.0, 0.0])
    
    pos_low  = np.array([6578e3, 0.0, 0.0])  # 200 km altitude
    pos_high = np.array([6978e3, 0.0, 0.0])  # 600 km altitude
    
    acc_low  = np.linalg.norm(drag.compute(pos_low, vel_vec))
    acc_high = np.linalg.norm(drag.compute(pos_high, vel_vec))
    
    assert acc_low > acc_high
  
  def test_sanity_check_drag_zero_at_high_altitude(self):
    """
    Drag should be negligible at high altitudes.
    """
    drag = AtmosphericDrag(cd=2.2, area=10.0, mass=1000.0)
    
    pos_vec = np.array([42164e3, 0.0, 0.0])  # GEO altitude
    vel_vec = np.array([0.0, 3075.0, 0.0])
    
    acc_drag = np.linalg.norm(drag.compute(pos_vec, vel_vec))
    
    # Should be essentially zero
    assert acc_drag < 1e-20


class TestAccelerationCoordinator:
  """
  Tests for the Acceleration coordinator class.
  """
  
  def test_physical_laws_gravity_only(self):
    """
    Test acceleration with only gravity enabled.
    """
    accel = Acceleration(
      gp          = SOLARSYSTEMCONSTANTS.EARTH.GP,
      enable_drag = False,
      enable_srp  = False,
    )
    
    pos_vec = np.array([7000e3, 0.0, 0.0])
    vel_vec = np.array([0.0, 7500.0, 0.0])
    
    acc_total = accel.compute(0.0, pos_vec, vel_vec)
    
    # Should be approximately point mass gravity
    expected = -SOLARSYSTEMCONSTANTS.EARTH.GP * pos_vec / np.linalg.norm(pos_vec)**3
    assert np.allclose(acc_total, expected, rtol=1e-10)
  
  def test_relative_check_drag_adds_to_gravity(self):
    """
    Test that drag is added to gravity when enabled.
    """
    accel_no_drag = Acceleration(
      gp          = SOLARSYSTEMCONSTANTS.EARTH.GP,
      enable_drag = False,
    )
    
    accel_with_drag = Acceleration(
      gp          = SOLARSYSTEMCONSTANTS.EARTH.GP,
      enable_drag = True,
      cd          = 2.2,
      area_drag   = 10.0,
      mass        = 1000.0,
    )
    
    pos_vec = np.array([6578e3, 0.0, 0.0])  # Low altitude for drag
    vel_vec = np.array([0.0, 7800.0, 0.0])
    
    acc_no_drag   = accel_no_drag.compute(0.0, pos_vec, vel_vec)
    acc_with_drag = accel_with_drag.compute(0.0, pos_vec, vel_vec)
    
    # With drag, total acceleration should be different
    assert not np.allclose(acc_no_drag, acc_with_drag)


class TestEnergyConservation:
  """
  Tests for energy conservation in conservative systems.
  """
  
  def test_physical_laws_specific_energy_formula(self):
    """
    Verify specific orbital energy calculation.
    """
    gp = SOLARSYSTEMCONSTANTS.EARTH.GP
    
    # Circular orbit
    pos_mag = 7000e3
    vel_mag = np.sqrt(gp / pos_mag)
    
    pos_vec = np.array([pos_mag, 0.0, 0.0])
    vel_vec = np.array([0.0, vel_mag, 0.0])
    
    # Specific energy: E = v²/2 - μ/r
    energy = 0.5 * np.dot(vel_vec, vel_vec) - gp / np.linalg.norm(pos_vec)
    
    # For circular orbit: E = -μ/(2a) = -μ/(2r)
    expected_energy = -gp / (2 * pos_mag)
    
    assert np.isclose(energy, expected_energy, rtol=1e-10)


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
